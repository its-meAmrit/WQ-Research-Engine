"""Fetching a URL the person typed, or one a web page named, without letting it reach inward.

A link in a search result or a landing page's ``citation_pdf_url`` is chosen by someone else.
Followed naively it can point this machine at a router, a database or a cloud metadata
address. So every hop here:

1. must be ``http(s)`` to a public-looking host (:func:`~..integrations.base.safe_public_url`);
2. has its name resolved *here*, and is refused if any answer is private, loopback,
   link-local, reserved or multicast;
3. is connected to the address that was checked, not looked up again (the name stays in the
   ``Host`` header and in TLS' server name), so a name that answers differently the second
   time gains nothing;
4. has redirects followed by hand, each target going through 1-3 again;
5. is read as a stream and stopped at a byte limit, so a large or endless body is never held.

Credentials are never sent: these are anonymous reads of public documents.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx2

from ..integrations.base import safe_public_url

USER_AGENT = "AlphaHarness-ResearchImporter (local research tool)"
MAX_REDIRECTS = 4
TOTAL_SECONDS = 60.0
CONNECT_SECONDS = 10.0
READ_SECONDS = 25.0
#: Refused before the body is read when the server announces more.
DEFAULT_MAX_BYTES = 30 * 1024 * 1024

Resolver = Callable[[str, int], Awaitable[list[str]]]


class FetchError(Exception):
    """A fetch this module refused or could not finish, with a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class Fetched:
    url: str
    content_type: str
    data: bytes
    status: int


async def resolve(host: str, port: int) -> list[str]:
    """Every address ``host`` answers with."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise FetchError("unresolvable", f"Could not find the server {host}.") from exc
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


async def _pinned(url: str, resolver: Resolver) -> tuple[str, str, dict[str, str]]:
    """``(url to request, host name, extensions)``: ``url`` with its host replaced by the
    checked address."""
    safe = safe_public_url(url)
    if safe is None:
        raise FetchError(
            "blocked_url", "That link is not a public web address, so it was not opened."
        )
    parts = urlsplit(safe)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = await resolver(host, port)
    if not addresses or not all(public(a) for a in addresses):
        raise FetchError(
            "blocked_address",
            f"{host} points at a private or reserved address, so it was not opened.",
        )
    ip = addresses[0]
    literal = f"[{ip}]" if ":" in ip else ip
    netloc = literal if parts.port is None else f"{literal}:{parts.port}"
    pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
    return pinned, host, {"sni_hostname": host}


def _type(header: str) -> str:
    return header.split(";", 1)[0].strip().lower()


async def fetch(
    url: str,
    *,
    accept: tuple[str, ...] = ("application/pdf", "text/html", "application/xhtml+xml"),
    max_bytes: int = DEFAULT_MAX_BYTES,
    resolver: Resolver = resolve,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> Fetched:
    """GET ``url`` under the rules above. ``accept`` lists the content types worth reading;
    ``application/octet-stream`` passes too, because servers send PDFs under it, and the caller
    checks the bytes."""
    try:
        async with asyncio.timeout(TOTAL_SECONDS):
            return await _fetch(url, accept, max_bytes, resolver, transport)
    except TimeoutError as exc:
        raise FetchError("timeout", "The server did not finish answering in time.") from exc


async def _fetch(
    url: str,
    accept: tuple[str, ...],
    max_bytes: int,
    resolver: Resolver,
    transport: httpx2.AsyncBaseTransport | None,
) -> Fetched:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": ", ".join(accept) + ", */*;q=0.1",
        "Accept-Encoding": "identity",
    }
    timeout = httpx2.Timeout(READ_SECONDS, connect=CONNECT_SECONDS)
    current = url
    async with httpx2.AsyncClient(
        timeout=timeout, follow_redirects=False, transport=transport, trust_env=False
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            target, host, extensions = await _pinned(current, resolver)
            try:
                async with client.stream(
                    "GET", target, headers={**headers, "Host": host}, extensions=extensions
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError(
                                "bad_redirect", "The server redirected without saying where."
                            )
                        current = urljoin(current, location)
                        continue
                    if response.status_code in {401, 402, 403}:
                        raise FetchError(
                            "access_denied",
                            "The server requires a login or payment for that document, so it "
                            "was not read. Upload the PDF yourself if you have access.",
                        )
                    if response.status_code == 404:
                        raise FetchError("not_found", "There is nothing at that address.")
                    if response.status_code >= 400:
                        raise FetchError(
                            "http_error",
                            f"The server answered with an error ({response.status_code}).",
                        )
                    kind = _type(response.headers.get("content-type", ""))
                    if kind and kind != "application/octet-stream" and kind not in accept:
                        raise FetchError(
                            "unsupported_type",
                            f"That address is {kind or 'an unknown type'}, not a paper or a page.",
                        )
                    announced = response.headers.get("content-length", "")
                    if announced.isdigit() and int(announced) > max_bytes:
                        raise FetchError(
                            "too_large",
                            f"That file is {int(announced) / 1_048_576:.1f} MB; the limit is "
                            f"{max_bytes // 1_048_576} MB.",
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise FetchError(
                                "too_large",
                                f"That file is larger than the {max_bytes // 1_048_576} MB limit.",
                            )
                    return Fetched(current, kind, bytes(body), response.status_code)
            except httpx2.TimeoutException as exc:
                raise FetchError("timeout", "The server did not answer in time.") from exc
            except httpx2.TransportError as exc:
                raise FetchError(
                    "network_error", f"Could not reach the server ({type(exc).__name__})."
                ) from exc
    raise FetchError("too_many_redirects", "That link redirects too many times.")
