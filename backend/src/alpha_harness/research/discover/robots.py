"""``robots.txt``, asked before a page is read by anything on this machine's behalf.

The file is fetched through :mod:`..netfetch`, so it gets the same address checks and byte
limit as any page. A site that cannot be asked is treated as having said no: an unreachable
or forbidden ``robots.txt`` is not permission. Answers are kept for a few hours per origin so
a job that reads ten pages of one site asks once.

This covers pages read directly and pages handed to an extraction service. It does not make
the service itself behave; that is the service's own policy.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .. import netfetch

AGENT = "AlphaHarness-ResearchImporter"
TTL_SECONDS = 6 * 3600.0
MAX_ROBOTS_BYTES = 512 * 1024

Fetch = Callable[..., Awaitable[netfetch.Fetched]]


@dataclass(slots=True)
class Verdict:
    allowed: bool
    #: Why, in words for the person: shown on a result that was not read.
    reason: str


@dataclass(slots=True)
class _Entry:
    parser: RobotFileParser | None
    verdict: Verdict | None
    expires: float


class RobotsPolicy:
    def __init__(self, fetch: Fetch = netfetch.fetch, *, ttl: float = TTL_SECONDS) -> None:
        self._fetch = fetch
        self._ttl = ttl
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def check(self, url: str) -> Verdict:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            entry = self._entries.get(origin)
            if entry is None or entry.expires < time.monotonic():
                entry = await self._load(origin)
                self._entries[origin] = entry
        if entry.verdict is not None or entry.parser is None:
            return entry.verdict or Verdict(False, "robots.txt could not be checked.")
        if entry.parser.can_fetch(AGENT, url):
            return Verdict(True, "Allowed by robots.txt.")
        return Verdict(
            False, f"{parts.hostname} asks automated tools not to read this page (robots.txt)."
        )

    async def _load(self, origin: str) -> _Entry:
        expires = time.monotonic() + self._ttl
        try:
            fetched = await self._fetch(
                f"{origin}/robots.txt",
                accept=("text/plain", "text/html", "application/octet-stream"),
                max_bytes=MAX_ROBOTS_BYTES,
            )
        except netfetch.FetchError as exc:
            return _Entry(None, _verdict_for(exc), expires)
        parser = RobotFileParser()
        parser.parse(fetched.data.decode("utf-8", errors="replace").splitlines())
        return _Entry(parser, None, expires)


def _verdict_for(exc: netfetch.FetchError) -> Verdict:
    """What a failure to read ``robots.txt`` means. A site with none (404) imposes nothing;
    every other failure is not permission."""
    if exc.code == "not_found":
        return Verdict(True, "The site has no robots.txt.")
    if exc.code == "access_denied":
        return Verdict(False, "The site's robots.txt is behind a login, so it cannot be checked.")
    return Verdict(
        False, "The site's robots.txt could not be read, so the page was not opened to be safe."
    )
