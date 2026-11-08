"""Current evidence for a run, from a search provider the user configured.

A chat model has no browser. Telling it to "search the web" in the prompt does nothing unless
the provider runs a search, so research here is a separate step: queries go to a search API,
the results are normalised, dated and screened, and the text is handed to the model with the
date it was retrieved. Nothing is retrieved on the model's behalf and nothing is claimed that
was not.

**Configuration.** Add a Tavily or Brave Search key under API Integration › Web Research. The
service that applies is the user's enabled stored key; failing that, the older
``ALPHA_HARNESS_SPC_SEARCH`` / ``ALPHA_HARNESS_SPC_SEARCH_KEY`` environment variables; failing
that, research is reported as not configured and a run may still go ahead without it, saying so.
The adapters, the keys and the budget live in :mod:`..integrations`; this module is the research
*step*: what to ask, and what to do with the answers.

**What is fetched.** Only the provider's own fixed endpoint. This build never downloads the
pages the results name, so there is no server-side fetch of an arbitrary URL to turn against
the machine's network. The result URLs are still screened (see :func:`safe_public_url`)
because they are shown and stored, and a link to a private address has no place there.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx2
import structlog

from ..integrations import legacy
from ..integrations.base import SearchProvider, safe_public_url
from ..integrations.brave import BraveSearch
from ..integrations.errors import IntegrationError
from ..integrations.legacy import ENV_KEY, ENV_PROVIDER, SETUP_HELP
from ..integrations.tavily import TavilySearch
from .errors import SpcError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = structlog.get_logger(__name__)

#: Seconds a search request may take.
TIMEOUT_SECONDS = 15.0
#: Searches in flight at once.
CONCURRENCY = 3
RESULTS_PER_QUERY = 6
MAX_SOURCES = 18
#: Evidence older than this is marked stale unless the prompt asked for something else.
DEFAULT_MAX_AGE_DAYS = 90

#: Re-exported: the legacy environment variables and the setup message live in
#: :mod:`..integrations.legacy`, beside the code that reads them.
__all__ = [
    "ENV_KEY",
    "ENV_PROVIDER",
    "SETUP_HELP",
    "BraveSearch",
    "SearchProvider",
    "TavilySearch",
    "safe_public_url",
]


class ResearchNotConfiguredError(SpcError):
    code = "research_not_configured"


class ResearchFailedError(SpcError):
    code = "research_failed"


@dataclass(slots=True)
class Source:
    """One retrieved source. A date is only ever the one the source gave."""

    url: str
    title: str
    snippet: str
    query: str
    source_type: str
    published_at: str | None
    retrieved_at: datetime
    #: Older than the freshness window. Not set when the date is unknown: that is its own
    #: state, neither fresh nor stale.
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "query": self.query,
            "sourceType": self.source_type,
            "publishedAt": self.published_at,
            "retrievedAt": self.retrieved_at.isoformat(),
            "stale": self.stale,
        }


@dataclass(slots=True)
class ResearchResult:
    provider: str
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    #: What did not go to plan, in words: a failed query, a screened-out URL.
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "status": "done",
            "provider": self.provider,
            "queries": self.queries,
            "sources": len(self.sources),
            "stale": sum(1 for s in self.sources if s.stale),
            "undated": sum(1 for s in self.sources if s.published_at is None),
            "notes": self.notes,
        }


# --- URLs -------------------------------------------------------------------------------


_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("regulatory_filing", ("sec.gov", "edgar", "sedar", "esma.europa.eu", "companieshouse.gov.uk")),
    (
        "exchange_disclosure",
        ("nasdaq.com", "nyse.com", "lseg.com", "jpx.co.jp", "hkexnews.hk", "sgx.com", "asx.com.au"),
    ),
    (
        "government_statistics",
        (
            ".gov",
            "bls.gov",
            "census.gov",
            "imf.org",
            "worldbank.org",
            "oecd.org",
            "ecb.europa.eu",
            "federalreserve.gov",
        ),
    ),
    (
        "financial_news",
        (
            "reuters.com",
            "ft.com",
            "bloomberg.com",
            "wsj.com",
            "cnbc.com",
            "barrons.com",
            "economist.com",
            "nikkei.com",
        ),
    ),
)


def classify(url: str) -> str:
    """A heuristic label from the domain alone. A label, not an endorsement."""
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith(("investor.", "investors.", "ir.")) or ".investor" in host:
        return "investor_relations"
    for kind, markers in _TYPES:
        if any(host == m.lstrip(".") or host.endswith(m) or m in host for m in markers):
            return kind
    return "web"


_AGO = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?\s+ago", re.IGNORECASE)
_UNIT_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}


def parse_published(value: object, *, now: datetime) -> datetime | None:
    """A source's own date, or ``None`` when it gave none or one that cannot be read."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    ago = _AGO.search(text)
    if ago:
        return now - timedelta(days=int(ago.group(1)) * _UNIT_DAYS[ago.group(2).lower()])
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --- providers --------------------------------------------------------------------------


def configured_provider(env: Mapping[str, str] | None = None) -> SearchProvider | None:
    """The provider the *legacy environment* names, or ``None``. An unknown name is not guessed at.

    Only the environment: a key stored in the application is resolved by
    :class:`~..integrations.service.WebResearchService`, which falls back to this.
    """
    config = legacy.read(env)
    if config is None or not config.named or not config.key:
        return None
    if config.provider is None:
        log.warning("spc.search_provider_unknown", name=config.named)
        return None
    return BraveSearch(config.key) if config.provider == "brave" else TavilySearch(config.key)


def status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What the UI shows about research from the legacy environment alone: configured or not,
    and how to fix it. The application's own status, which also knows about stored keys, is
    ``WebResearchService.research_status``; this is what an SPC service built without one shows."""
    provider = configured_provider(env)
    if provider is not None:
        return {"configured": True, "provider": provider.name, "help": None}
    config = legacy.read(env)
    if config is not None and config.named and config.provider is None:
        return {
            "configured": False,
            "provider": None,
            "help": f"{config.named!r} is not a search provider here; use 'brave' or 'tavily'.",
        }
    return {"configured": False, "provider": None, "help": SETUP_HELP}


# --- the step ---------------------------------------------------------------------------


def build_queries(settings: dict[str, Any] | None, prompt_text: str, *, now: datetime) -> list[str]:
    """Search queries from the Prompt Studio options, or the prompt's opening when it was
    written by hand. A handful, because each one spends the search quota."""
    when = now.strftime("%B %Y")
    s = settings or {}
    sector = str(s.get("sector") or "").strip()
    universe = str(s.get("universe") or "").strip()
    thesis = str(s.get("thesis") or "").strip()
    topic = " ".join(p for p in (sector, universe) if p) or thesis[:80]
    if not topic:
        first = next(
            (line.strip() for line in prompt_text.splitlines() if len(line.strip()) > 20), ""
        )
        topic = first[:100]
    if not topic:
        return []
    queries = [
        f"{topic} earnings outlook {when}",
        f"{topic} company news {when}",
        f"{topic} valuation analyst expectations",
    ]
    if thesis and thesis[:80] != topic:
        queries.append(f"{thesis[:100]} {when}")
    return queries[:4]


def evidence_block(sources: Sequence[Source], *, retrieved: datetime) -> str:
    """The evidence as text for the model, dated, with its gaps named rather than hidden."""
    lines = [
        f"RESEARCH EVIDENCE (retrieved {retrieved.date().isoformat()}; may be incomplete)",
        "Use it with your own checks. A source marked undated has no known publication date; "
        "one marked stale is older than the freshness window.",
    ]
    for n, s in enumerate(sources, 1):
        when = s.published_at or "undated"
        flag = ", stale" if s.stale else ""
        lines.append(f"[{n}] {s.title} ({when}{flag}) {s.url}\n    {s.snippet[:400]}")
    return "\n".join(lines)


async def gather(
    provider: SearchProvider,
    queries: Sequence[str],
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: datetime,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> ResearchResult:
    """Run the queries with bounded concurrency and a timeout, and normalise what comes back.

    A query that fails is noted and the rest carry on; all of them failing is an error, never
    an empty success.
    """
    result = ResearchResult(provider=provider.name, queries=list(queries))
    gate = asyncio.Semaphore(CONCURRENCY)
    async with httpx2.AsyncClient(
        timeout=httpx2.Timeout(TIMEOUT_SECONDS), follow_redirects=False, transport=transport
    ) as client:

        async def one(query: str) -> list[dict[str, Any]] | str:
            async with gate:
                try:
                    return await provider.search(
                        client, query, count=RESULTS_PER_QUERY, freshness_days=max_age_days
                    )
                except (ResearchFailedError, IntegrationError) as exc:
                    # Typed and key-free: built from the service's name and an HTTP status.
                    return str(exc)
                except httpx2.TransportError as exc:
                    return f"{provider.name} could not be reached ({type(exc).__name__})."
                except ValueError:
                    return f"{provider.name} sent an answer that could not be read."

        answers = await asyncio.gather(*(one(q) for q in queries))

    seen: set[str] = set()
    failures = 0
    for query, answer in zip(queries, answers, strict=True):
        if isinstance(answer, str):
            failures += 1
            result.notes.append(f"Query {query!r}: {answer}")
            continue
        for raw in answer:
            url = safe_public_url(str(raw.get("url") or ""))
            if url is None:
                result.notes.append("A result with an unsafe or unusable URL was dropped.")
                continue
            if url in seen or len(result.sources) >= MAX_SOURCES:
                continue
            seen.add(url)
            published = parse_published(raw.get("published"), now=now)
            result.sources.append(
                Source(
                    url=url,
                    title=str(raw.get("title") or url)[:300],
                    snippet=str(raw.get("snippet") or "")[:1000],
                    query=query,
                    source_type=classify(url),
                    published_at=published.date().isoformat() if published else None,
                    retrieved_at=now,
                    stale=published is not None and now - published > timedelta(days=max_age_days),
                )
            )
    if queries and failures == len(queries):
        raise ResearchFailedError(" ".join(["Every search failed.", *result.notes[:2]]))
    return result
