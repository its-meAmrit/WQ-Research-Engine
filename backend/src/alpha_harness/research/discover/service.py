"""Discovery: find public documents, read the ones the person picks, and save their ideas.

Two jobs, both written to ``research_job`` before they start so a restart can say where they
stopped and a screen can follow them by reading the row:

``discover``
    Searches the sources the person chose. Reads *search results only*: titles, snippets,
    dates. Nothing is fetched from the found pages and no model is called. Every result is kept
    as a ``research_discovery_hit`` with its attribution and the time it was retrieved.

``extract``
    Reads the hits the person picked, one at a time, and only if the site's robots.txt allows
    it. Paywalled, login-gated and CAPTCHA-gated pages are not read past: they are kept as
    metadata only. Text that is read is handed to the model between markers as data; whatever
    instructions it contains are ignored, and the model's answer is validated and grounded in
    the page before anything is saved.

Nothing here starts a simulation, signs in to BRAIN, or reads the person's own alphas.
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from typing import Any, Literal
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

import structlog
from pydantic import Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from ...db.models import (
    ResearchDiscoveryHit,
    ResearchDocument,
    ResearchIdea,
    ResearchIdeaLink,
    ResearchJob,
    ResearchRegistryEntry,
    ResearchSearchCache,
    ResearchSource,
    utcnow,
)
from ...integrations import registry as web_registry
from ...integrations import tavily
from ...integrations.base import (
    DocumentQuery,
    DocumentSearch,
    IntegrationError,
    safe_public_url,
)
from ...integrations.errors import RequestBudgetExhaustedError
from ...integrations.service import Resolved, WebResearchService
from ...llm.keys import LLMError
from ...schemas import Out
from .. import extraction as ex
from .. import netfetch
from ..identifiers import canonical_url
from ..scholarly import PaperMeta, ScholarlyError
from ..service import ResearchError, ResearchService
from . import dedupe, htmltext, queries, ranking, webextract
from .registry import Registry, RegistryError
from .robots import RobotsPolicy

log = structlog.get_logger(__name__)

#: Most results one request may ask for, whatever the provider would allow.
MAX_RESULTS = 20
#: Searches one job may send. A cap on cost and on time, not on how many sources exist.
MAX_REQUESTS_PER_JOB = 12
SEARCH_CONCURRENCY = 2
FETCH_CONCURRENCY = 3
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
CACHE_TTL = timedelta(hours=6)
MAX_EXTRACT_PER_JOB = 10
MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_FEED_BYTES = 2 * 1024 * 1024
TEMPERATURE = 0.1

#: A hit's life: found -> extracted | rejected | restricted | failed | duplicate | imported.
HIT_STATUSES = ("found", "duplicate", "restricted", "extracted", "rejected", "failed", "imported")
FEED_TYPES = (
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
    "text/html",
)

Sleep = Callable[[float], Awaitable[None]]


class DiscoverRequest(Out):
    """What the Discover screen sends."""

    topic: str = Field(default="", max_length=300)
    #: The searches to run, as the person edited them. Empty means the topic itself.
    queries: list[str] = Field(default_factory=list, max_length=queries.MAX_QUERIES)
    market: str = Field(default="", max_length=120)
    #: Registry ids. Empty means every enabled search source.
    sources: list[str] = Field(default_factory=list, max_length=20)
    provider: str | None = Field(default=None, max_length=32)
    max_results: int = Field(default=10, ge=1, le=MAX_RESULTS)
    depth: Literal["basic", "advanced"] = "basic"
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    include_domains: list[str] = Field(default_factory=list, max_length=20)
    urls: list[str] = Field(default_factory=list, max_length=MAX_EXTRACT_PER_JOB)


@dataclass(slots=True)
class Found:
    """One search result before it is stored."""

    url: str
    title: str
    snippet: str
    published: str | None
    score: float | None
    provider: str
    registry_id: str
    query: str
    from_cache: bool = False
    #: Set for scholarly results: the paper pipeline imports these, not the page reader.
    paper: dict[str, Any] | None = None


@dataclass(slots=True)
class Plan:
    """A request after it has been checked: what will be searched, where, with which key."""

    request: DiscoverRequest
    queries: list[str]
    entries: list[ResearchRegistryEntry]
    resolved: Resolved | None
    provider_id: str | None
    searches: list[tuple[ResearchRegistryEntry, str]]
    key: str
    estimate: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Outcome:
    """How one hit's extraction ended."""

    hit_id: str
    status: str
    detail: str
    source_id: str | None = None
    idea_ids: list[str] = field(default_factory=list)
    linked: list[dict[str, Any]] = field(default_factory=list)


class Retryable(Exception):
    """A failure worth trying again after a wait."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _iso(value: Any) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _in_range(published: str | None, start: str | None, end: str | None) -> bool:
    """Whether a publication date falls inside the window. An undated result is kept: dropping
    it would hide it for a reason the person cannot see."""
    if not published or not (start or end):
        return True
    stamp = published[:10]
    if start and stamp < start[: len(stamp)]:
        return False
    return not (end and stamp > end[: len(stamp)])


class DiscoveryService:
    def __init__(
        self,
        db: Any,
        research: ResearchService,
        web: WebResearchService,
        registry: Registry,
        *,
        robots: RobotsPolicy | None = None,
        fetch: Callable[..., Awaitable[netfetch.Fetched]] = netfetch.fetch,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.db = db
        self.research = research
        self.web = web
        self.registry = registry
        self._fetch = fetch
        self.robots = robots or RobotsPolicy(fetch)
        self._sleep = sleep
        self._search_gate = asyncio.Semaphore(SEARCH_CONCURRENCY)
        self._fetch_gate = asyncio.Semaphore(FETCH_CONCURRENCY)

    # --- planning -------------------------------------------------------------------------

    async def _provider(self, wanted: str | None) -> Resolved | None:
        """The key a web search will use, or ``None`` when nothing is set up."""
        if wanted:
            web_registry.spec_for(wanted)
            row = await self.web.store.get(wanted)
            if row is None or not row.enabled:
                raise ResearchError(
                    "no_search_provider",
                    f"{web_registry.spec_for(wanted).label} has no enabled key. Add one under "
                    "API Integration, or choose another service.",
                    422,
                )
            key = await self.web.store.secret(wanted)
            return Resolved(web_registry.build(wanted, key), "store", wanted, row.id)
        return await self.web.resolve("search")

    async def plan(self, request: DiscoverRequest) -> Plan:
        topic = " ".join(request.topic.split())
        try:
            if request.queries:
                texts = queries.validate(request.queries)
            elif topic:
                text = f"{topic} {request.market}".strip() if request.market else topic
                texts = queries.validate([text])
            else:
                texts = []
        except queries.QueryError as exc:
            raise ResearchError("bad_query", str(exc), 422) from exc
        urls = [u.strip() for u in request.urls if u.strip()]
        if not texts and not urls:
            raise ResearchError("bad_query", "Enter a topic or a URL to discover from.", 422)

        enabled = await self.registry.entries(enabled=True)
        if request.sources:
            by_id = {e.id: e for e in enabled}
            missing = [s for s in request.sources if s not in by_id]
            if missing:
                raise ResearchError(
                    "source_unavailable",
                    f"{missing[0]} is not an enabled source. Switch it on in the source list.",
                    422,
                )
            entries = [by_id[s] for s in request.sources]
        else:
            entries = [e for e in enabled if e.adapter in {"scholarly", "tavily"}]
        if urls and not any(e.adapter == "direct_url" for e in entries):
            direct = [e for e in enabled if e.adapter == "direct_url"]
            if not direct:
                raise ResearchError(
                    "source_unavailable", "The URL source is switched off in the source list.", 422
                )
            entries.append(direct[0])
        if not entries:
            raise ResearchError("source_unavailable", "No source is enabled to search.", 422)

        notes: list[str] = []
        resolved: Resolved | None = None
        needs_key = texts and any(e.adapter == "tavily" for e in entries)
        if needs_key:
            resolved = await self._provider(request.provider)
            if resolved is None or not isinstance(resolved.provider, DocumentSearch):
                raise ResearchError(
                    "no_search_provider",
                    "Web discovery needs a Tavily or Brave key, and none is set up. Add one "
                    "under API Integration. Academic indexes and feeds work without one.",
                    422,
                )
        searches: list[tuple[ResearchRegistryEntry, str]] = []
        for entry in entries:
            if entry.adapter == "direct_url":
                continue
            searches.extend((entry, text) for text in texts)
        if len(searches) > MAX_REQUESTS_PER_JOB:
            notes.append(
                f"{len(searches)} searches were asked for; the first {MAX_REQUESTS_PER_JOB} "
                "will run."
            )
            searches = searches[:MAX_REQUESTS_PER_JOB]
        provider_id = resolved.provider_id if resolved else None
        estimate = self._estimate(request, searches, resolved)
        key = _hash(
            {
                "q": texts,
                "e": sorted(e.id for e in entries),
                "p": provider_id,
                "n": request.max_results,
                "d": request.depth,
                "s": request.start_date,
                "u": request.end_date,
                "i": sorted(request.include_domains),
                "urls": sorted(urls),
            }
        )
        return Plan(request, texts, entries, resolved, provider_id, searches, key, estimate, notes)

    @staticmethod
    def _estimate(
        request: DiscoverRequest,
        searches: list[tuple[ResearchRegistryEntry, str]],
        resolved: Resolved | None,
    ) -> dict[str, Any]:
        """What the job will spend. Search and extraction are separate, and extraction happens
        only for the results the person picks afterwards."""
        web_searches = sum(1 for e, _ in searches if e.adapter == "tavily")
        provider = resolved.provider_id if resolved else None
        per = tavily.search_credits(request.depth) if provider == "tavily" else 1.0
        extract = (
            tavily.extract_credits(request.max_results, "basic") if provider == "tavily" else None
        )
        return {
            "provider": provider,
            "requests": web_searches,
            "free": sum(1 for e, _ in searches if e.adapter != "tavily"),
            "searchCredits": round(web_searches * per, 2),
            "extractCreditsIfAllRead": extract,
            "maxResults": request.max_results,
            "note": (
                "Searching reads results only. Reading the pages is a separate step you start "
                "yourself, with its own cost. Credit figures are estimates; your provider's "
                "dashboard has the real balance."
            ),
        }

    async def estimate(self, request: DiscoverRequest) -> dict[str, Any]:
        plan = await self.plan(request)
        return plan.estimate | {"queries": plan.queries, "notes": plan.notes}

    # --- discover job -----------------------------------------------------------------------

    async def start(self, request: DiscoverRequest) -> dict[str, Any]:
        plan = await self.plan(request)
        running = await self._in_flight("discover", plan.key)
        if running:
            return {"jobId": running, "existing": True, "message": "That search is running."}
        job_id = await self.research.new_job(
            kind="discover",
            input_ref=(plan.queries[0] if plan.queries else "URLs")[:200],
            cache_key=plan.key,
            provider=plan.provider_id,
            params={
                "request": request.model_dump(mode="json"),
                "queries": plan.queries,
                "sources": [e.id for e in plan.entries],
                "estimate": plan.estimate,
                "notes": plan.notes,
            },
            message="Queued.",
        )

        async def work(job_id: str) -> None:
            await self._discover(job_id, plan)

        self.research.launch(job_id, f"Discovering: {(plan.queries or ['URLs'])[0][:50]}", work)
        return {"jobId": job_id, "existing": False, "message": "Discovery started."}

    async def _in_flight(self, kind: str, key: str) -> str | None:
        async with self.db.session() as session:
            return await session.scalar(
                select(ResearchJob.id).where(
                    ResearchJob.kind == kind,
                    ResearchJob.cache_key == key,
                    ResearchJob.status.in_(["queued", "running"]),
                )
            )

    async def _discover(self, job_id: str, plan: Plan) -> None:
        request = plan.request
        spent = {"requests": 0, "credits": 0.0, "cached": 0}
        log_rows: list[dict[str, Any]] = []
        found: list[list[Found]] = []
        exhausted = False
        total = max(1, len(plan.searches) + (1 if request.urls else 0))
        done = 0

        async def run(entry: ResearchRegistryEntry, text: str) -> None:
            nonlocal exhausted, done
            line: dict[str, Any] = {
                "registryId": entry.id,
                "label": entry.label,
                "query": text,
                "provider": plan.provider_id if entry.adapter == "tavily" else entry.adapter,
                "hits": 0,
                "fromCache": False,
                "error": None,
            }
            try:
                async with self._search_gate:
                    if exhausted:
                        raise RequestBudgetExhaustedError("Skipped: the budget is used up.")
                    results, cached, spend, note = await self._search_one(plan, entry, text)
                found.append(results)
                line |= {"hits": len(results), "fromCache": cached, "note": note}
                spent["credits"] += spend
                spent["cached"] += 1 if cached else 0
                spent["requests"] += 0 if cached or entry.adapter != "tavily" else 1
                await self.registry.record(entry.id, error=None)
            except RequestBudgetExhaustedError as exc:
                exhausted = True
                line["error"] = {"code": exc.code, "message": str(exc)}
            except (IntegrationError, ScholarlyError, netfetch.FetchError, Retryable) as exc:
                code = getattr(exc, "code", "failed")
                line["error"] = {"code": code, "message": str(exc)}
                await self.registry.record(entry.id, error=str(exc))
            log_rows.append(line)
            done += 1
            await self.research.progress(
                job_id, 0.1 + 0.7 * done / total, f"Searched {done}/{total}."
            )

        await self.research.progress(job_id, 0.05, "Searching.")
        await asyncio.gather(*(run(e, t) for e, t in plan.searches))
        if request.urls:
            found.append(await self._direct(request.urls))
            done += 1

        results = [r for batch in found for r in batch]
        if not results and log_rows and all(row["error"] for row in log_rows):
            first = log_rows[0]["error"]
            await self.research.fail(job_id, first["code"], first["message"])
            await self.research.set_job(job_id, result={"searches": log_rows})
            return
        await self.research.progress(job_id, 0.85, "Removing duplicates.")
        counts = await self._store_hits(job_id, plan, found)
        await self.research.set_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"{counts['shown']} result(s) found."
            if counts["shown"]
            else "The search finished with no results.",
            usage=spent,
            result={
                **counts,
                "searches": log_rows,
                "notes": plan.notes,
                "budgetExhausted": exhausted,
            },
            finished_at=utcnow(),
        )

    async def _search_one(
        self, plan: Plan, entry: ResearchRegistryEntry, text: str
    ) -> tuple[list[Found], bool, float, str | None]:
        """``(results, from_cache, credits_spent, note)`` for one source and one query."""
        request = plan.request
        limit = min(request.max_results, int(entry.retrieval_policy.get("maxPerJob", MAX_RESULTS)))
        domains = tuple(dict.fromkeys([*entry.config.get("domains", []), *request.include_domains]))
        provider = plan.provider_id if entry.adapter == "tavily" else entry.adapter
        cache_key = _hash(
            {
                "adapter": entry.adapter,
                "id": entry.id,
                "provider": provider,
                "q": text,
                "n": limit,
                "depth": request.depth,
                "s": request.start_date,
                "u": request.end_date,
                "d": list(domains),
                "feed": entry.config.get("feedUrl"),
            }
        )
        ttl = (
            timedelta(seconds=entry.min_interval_seconds)
            if entry.min_interval_seconds
            else CACHE_TTL
        )
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return [self._found(h, entry, text, True) for h in cached["hits"]], True, 0.0, None
        if entry.adapter == "rss" and entry.last_success_at is not None:
            wait = entry.last_success_at + timedelta(seconds=entry.min_interval_seconds)
            if wait > utcnow():
                return (
                    [],
                    False,
                    0.0,
                    f"Read recently; this feed is read at most every "
                    f"{entry.min_interval_seconds // 3600} h.",
                )
        if entry.adapter == "tavily":
            hits, spend, note = await self._web_search(plan, entry, text, limit, domains)
        elif entry.adapter == "scholarly":
            hits, spend, note = await self._scholarly_search(plan, entry, text, limit)
        elif entry.adapter == "rss":
            hits, spend, note = await self._feed(plan, entry, text, limit)
        else:
            return [], False, 0.0, None
        await self._cache_put(cache_key, provider or entry.adapter, entry.adapter, hits, spend, ttl)
        return hits, False, spend, note

    async def _web_search(
        self,
        plan: Plan,
        entry: ResearchRegistryEntry,
        text: str,
        limit: int,
        domains: tuple[str, ...],
    ) -> tuple[list[Found], float, str | None]:
        resolved = plan.resolved
        assert resolved is not None  # noqa: S101 - plan() refuses a web search without a key
        adapter = resolved.provider
        assert isinstance(adapter, DocumentSearch)  # noqa: S101
        limit = min(limit, adapter.max_results_cap)
        query = DocumentQuery(
            query=text,
            max_results=limit,
            depth=plan.request.depth,
            start_date=plan.request.start_date,
            end_date=plan.request.end_date,
            include_domains=domains,
        )

        async def call() -> Any:
            await self.web.reserve(resolved, 1)
            try:
                async with self.web.http_client() as client:
                    page = await adapter.search_documents(client, query)
            except IntegrationError as exc:
                await self.web.record(resolved, error=exc)
                raise
            await self.web.record(resolved, error=None)
            return page

        page = await self._retry(call)
        hits = [
            Found(
                url=h.url,
                title=h.title,
                snippet=h.snippet,
                published=h.published,
                score=h.score,
                provider=resolved.provider_id or "",
                registry_id=entry.id,
                query=text,
            )
            for h in page.hits
        ]
        spend = (
            page.credits
            if page.credits is not None
            else (
                tavily.search_credits(plan.request.depth)
                if resolved.provider_id == "tavily"
                else 1.0
            )
        )
        return hits, float(spend), "; ".join(page.notes) or None

    async def _scholarly_search(
        self, plan: Plan, entry: ResearchRegistryEntry, text: str, limit: int
    ) -> tuple[list[Found], float, str | None]:
        client = self.research.scholarly_client()
        try:
            papers: list[PaperMeta] = await self._retry(lambda: client.search(text, limit=limit))
        finally:
            await client.aclose()
        out: list[Found] = []
        for meta in papers:
            if not _in_range(meta.published, plan.request.start_date, plan.request.end_date):
                continue
            url = meta.url or (f"https://doi.org/{meta.doi}" if meta.doi else "")
            if not url:
                continue
            out.append(
                Found(
                    url=url,
                    title=meta.title,
                    snippet=(meta.abstract or "")[:700],
                    published=meta.published,
                    score=None,
                    provider=meta.provider,
                    registry_id=entry.id,
                    query=text,
                    paper={
                        "doi": meta.doi,
                        "arxivId": meta.arxiv_id,
                        "authors": meta.authors[:8],
                        "venue": meta.venue,
                        "openCopy": bool(meta.pdf_url),
                    },
                )
            )
        return out, 0.0, "Free public indexes: no credits are used."

    async def _feed(
        self, plan: Plan, entry: ResearchRegistryEntry, text: str, limit: int
    ) -> tuple[list[Found], float, str | None]:
        url = str(entry.config.get("feedUrl", ""))
        if (verdict := await self.robots.check(url)) and not verdict.allowed:
            raise netfetch.FetchError("robots_disallowed", verdict.reason)
        fetched = await self._retry(
            lambda: self._fetch(url, accept=FEED_TYPES, max_bytes=MAX_FEED_BYTES)
        )
        items = parse_feed(fetched.data.decode("utf-8", errors="replace"))
        words = set(dedupe.tokens(text))
        need = max(1, (len(words) + 1) // 2)
        out: list[Found] = []
        for item in items:
            if len(out) >= limit:
                break
            if len(words & set(dedupe.tokens(f"{item['title']} {item['summary']}"))) < need:
                continue
            if not _in_range(item["published"], plan.request.start_date, plan.request.end_date):
                continue
            out.append(
                Found(
                    url=item["link"],
                    title=item["title"],
                    snippet=item["summary"][:700],
                    published=item["published"],
                    score=None,
                    provider="rss",
                    registry_id=entry.id,
                    query=text,
                )
            )
        return out, 0.0, f"{len(items)} feed item(s) checked against the query."

    @staticmethod
    async def _direct(urls: list[str]) -> list[Found]:
        """URLs the person typed. Each is a result as it stands; it is read only on Extract."""
        return [
            Found(
                url=raw.strip(),
                title="",
                snippet="",
                published=None,
                score=None,
                provider="direct",
                registry_id="user_urls",
                query="",
            )
            for raw in urls
        ]

    # --- retry, cache ---------------------------------------------------------------------

    async def _retry(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """``call()``, again after a wait when the failure can pass: a rate limit, a timeout,
        a busy server. Auth failures and an exhausted quota are not retried, since trying again
        cannot help and a rate-limited key only gets more limited. At most
        :data:`MAX_ATTEMPTS` tries."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await call()
            except RequestBudgetExhaustedError:
                raise
            except IntegrationError as exc:
                if not exc.retryable or attempt == MAX_ATTEMPTS:
                    raise
                wait = exc.retry_after
            except netfetch.FetchError as exc:
                if exc.code not in {"timeout", "network_error"} or attempt == MAX_ATTEMPTS:
                    raise
                wait = None
            except ScholarlyError as exc:
                if getattr(exc, "code", "") not in {"timeout", "rate_limited", "unavailable"} or (
                    attempt == MAX_ATTEMPTS
                ):
                    raise
                wait = None
            delay = min(
                wait if wait is not None else BACKOFF_SECONDS * 2 ** (attempt - 1),
                MAX_BACKOFF_SECONDS,
            )
            log.info("research.discover_retry", attempt=attempt, wait=delay)
            await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _cache_get(self, key: str) -> dict[str, Any] | None:
        async with self.db.session() as session:
            row = await session.get(ResearchSearchCache, key)
        if row is None or row.expires_at <= utcnow():
            return None
        return row.response

    async def _cache_put(
        self,
        key: str,
        provider: str,
        kind: str,
        hits: list[Found],
        spend: float,
        ttl: timedelta,
    ) -> None:
        body = {
            "hits": [
                {
                    "url": h.url,
                    "title": h.title,
                    "snippet": h.snippet,
                    "published": h.published,
                    "score": h.score,
                    "provider": h.provider,
                    "paper": h.paper,
                }
                for h in hits
            ]
        }
        async with self.db.session() as session:
            row = await session.get(ResearchSearchCache, key)
            expires = utcnow() + ttl
            if row is None:
                session.add(
                    ResearchSearchCache(
                        key=key,
                        provider=provider[:32],
                        kind=kind[:12],
                        request={},
                        response=body,
                        credits=spend,
                        expires_at=expires,
                    )
                )
            else:
                row.response, row.credits, row.expires_at = body, spend, expires

    @staticmethod
    def _found(raw: dict[str, Any], entry: ResearchRegistryEntry, text: str, cached: bool) -> Found:
        return Found(
            url=raw["url"],
            title=raw.get("title", ""),
            snippet=raw.get("snippet", ""),
            published=raw.get("published"),
            score=raw.get("score"),
            provider=raw.get("provider", ""),
            registry_id=entry.id,
            query=text,
            from_cache=cached,
            paper=raw.get("paper"),
        )

    # --- storing results ------------------------------------------------------------------

    async def _store_hits(
        self, job_id: str, plan: Plan, batches: list[list[Found]]
    ) -> dict[str, int]:
        """Interleave the sources, drop repeats, cap, and save each result with its status."""
        order: list[Found] = []
        for rank in range(max((len(b) for b in batches), default=0)):
            order.extend(b[rank] for b in batches if rank < len(b))
        counts = {"found": 0, "duplicates": 0, "restricted": 0, "shown": 0}
        seen: dict[str, str] = {}
        async with self.db.session() as session:
            sources = (
                await session.execute(
                    select(
                        ResearchSource.id,
                        ResearchSource.canonical_url,
                        ResearchSource.doi,
                        ResearchSource.arxiv_id,
                    )
                )
            ).all()
            by_url = {s.canonical_url: s.id for s in sources if s.canonical_url}
            by_doi = {s.doi: s.id for s in sources if s.doi}
            by_arxiv = {s.arxiv_id: s.id for s in sources if s.arxiv_id}
            kept = 0
            for item in order:
                status, detail, duplicate_of, source_id = "found", "", None, None
                safe = safe_public_url(item.url)
                canon = canonical_url(item.url) if safe else None
                if safe is None or canon is None:
                    status, detail = (
                        "restricted",
                        "Not a public web address, so it will not be opened.",
                    )
                    counts["restricted"] += 1
                elif canon in seen:
                    status, detail, duplicate_of = (
                        "duplicate",
                        "Already in these results.",
                        seen[canon],
                    )
                    counts["duplicates"] += 1
                else:
                    paper = item.paper or {}
                    known = (
                        by_url.get(canon)
                        or by_doi.get(str(paper.get("doi") or ""))
                        or by_arxiv.get(str(paper.get("arxivId") or ""))
                    )
                    if known:
                        status, detail, source_id = "duplicate", "Already in your library.", known
                        counts["duplicates"] += 1
                if status == "found":
                    if kept >= plan.request.max_results:
                        continue
                    kept += 1
                    counts["found"] += 1
                hit = ResearchDiscoveryHit(
                    job_id=job_id,
                    query=item.query,
                    rank=len(seen),
                    provider=item.provider,
                    registry_id=item.registry_id,
                    url=item.url,
                    canonical_url=canon,
                    domain=_host(item.url) or None,
                    title=item.title[:500],
                    snippet=item.snippet,
                    published=item.published[:32] if item.published else None,
                    score=item.score,
                    status=status,
                    status_detail=detail,
                    source_id=source_id,
                    duplicate_of=duplicate_of,
                    from_cache=item.from_cache,
                )
                if item.paper:
                    hit.status_detail = detail or "Academic paper: opens in the paper pipeline."
                session.add(hit)
                await session.flush()
                if canon and canon not in seen:
                    seen[canon] = hit.id
                counts["shown"] += 1
        return counts

    @staticmethod
    def hit_view(hit: ResearchDiscoveryHit) -> dict[str, Any]:
        return {
            "id": hit.id,
            "jobId": hit.job_id,
            "query": hit.query,
            "rank": hit.rank,
            "provider": hit.provider,
            "registryId": hit.registry_id,
            "url": hit.url,
            "canonicalUrl": hit.canonical_url,
            "domain": hit.domain,
            "title": hit.title,
            "snippet": hit.snippet,
            "published": hit.published,
            "score": hit.score,
            "retrievedAt": _iso(hit.retrieved_at),
            "status": hit.status,
            "statusDetail": hit.status_detail,
            "sourceId": hit.source_id,
            "duplicateOf": hit.duplicate_of,
            "fromCache": hit.from_cache,
            "isPaper": hit.provider in {"openalex", "arxiv", "crossref", "semantic_scholar"},
        }

    async def hits(self, job_id: str) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(ResearchDiscoveryHit)
                        .where(ResearchDiscoveryHit.job_id == job_id)
                        .order_by(ResearchDiscoveryHit.rank, ResearchDiscoveryHit.retrieved_at)
                    )
                )
                .scalars()
                .all()
            )
        return [self.hit_view(h) for h in rows]

    # --- extract job ----------------------------------------------------------------------

    async def start_extract(
        self,
        hit_ids: list[str],
        *,
        model_ref: str | None,
        scope: Any = None,
        depth: Literal["basic", "advanced"] = "basic",
    ) -> dict[str, Any]:
        if not hit_ids:
            raise ResearchError("no_hits", "Choose at least one result to read.", 422)
        if len(hit_ids) > MAX_EXTRACT_PER_JOB:
            raise ResearchError(
                "too_many", f"Read at most {MAX_EXTRACT_PER_JOB} results at a time.", 422
            )
        async with self.db.session() as session:
            hits = list(
                (
                    await session.execute(
                        select(ResearchDiscoveryHit).where(ResearchDiscoveryHit.id.in_(hit_ids))
                    )
                )
                .scalars()
                .all()
            )
        if len(hits) != len(set(hit_ids)):
            raise ResearchError("hit_not_found", "One of those results no longer exists.", 404)
        try:
            model = self.research.llm.model_for(model_ref)
        except LLMError as exc:
            raise ResearchError("no_model", str(exc), 422) from exc
        if not [m for m in await self.research.llm.runnable() if m["ref"] == model.ref]:
            raise ResearchError(
                "no_key", f"{model.ref} has no enabled API key. Add one under API Integration.", 422
            )
        imports: list[dict[str, Any]] = []
        pages: list[ResearchDiscoveryHit] = []
        for hit in hits:
            if hit.status in {"duplicate", "restricted", "extracted", "imported"}:
                continue
            if self.hit_view(hit)["isPaper"]:
                imports.append(await self._import_paper(hit))
            else:
                pages.append(hit)
        if not pages:
            return {
                "jobId": None,
                "existing": False,
                "imports": imports,
                "message": "Nothing to read: those results are papers, duplicates or blocked.",
            }
        key = _hash(
            {
                "hits": sorted(h.id for h in pages),
                "model": model.ref,
                "v": webextract.PROMPT_VERSION,
            }
        )
        if running := await self._in_flight("extract", key):
            return {
                "jobId": running,
                "existing": True,
                "imports": imports,
                "message": "Those results are already being read.",
            }
        extract_credits = None
        if await self.web.resolve("extract") is not None:
            extract_credits = tavily.extract_credits(len(pages), depth)
        job_id = await self.research.new_job(
            kind="extract",
            input_ref=f"{len(pages)} page(s)",
            cache_key=key,
            model_ref=model.ref,
            provider=model.provider,
            model_id=model.id,
            params={
                "hitIds": [h.id for h in pages],
                "depth": depth,
                "scope": scope.model_dump() if scope is not None else None,
                "estimate": {
                    "pages": len(pages),
                    "extractCredits": extract_credits,
                    "modelCalls": f"about one per page ({len(pages)}+)",
                },
            },
            message="Queued.",
        )
        ids = [h.id for h in pages]

        async def work(job_id: str) -> None:
            await self._extract(job_id, ids, model.ref, scope, depth)

        self.research.launch(job_id, f"Reading {len(pages)} page(s)", work)
        return {
            "jobId": job_id,
            "existing": False,
            "imports": imports,
            "message": "Reading started.",
        }

    async def _import_paper(self, hit: ResearchDiscoveryHit) -> dict[str, Any]:
        """A scholarly result goes through the paper pipeline, which already knows DOIs, open
        copies and the PDF reader."""
        try:
            started = await self.research.import_reference(hit.url)
        except ResearchError as exc:
            await self._set_hit(hit.id, "failed", str(exc))
            return {"hitId": hit.id, "error": str(exc)}
        await self._set_hit(
            hit.id,
            "imported",
            "Imported as a paper. Analyse it from the paper's page.",
            source_id=started.source_id,
        )
        return {"hitId": hit.id, "sourceId": started.source_id, "jobId": started.job_id}

    async def _set_hit(
        self, hit_id: str, status: str, detail: str, *, source_id: str | None = None
    ) -> None:
        values: dict[str, Any] = {"status": status, "status_detail": detail[:600]}
        if source_id:
            values["source_id"] = source_id
        async with self.db.session() as session:
            await session.execute(
                update(ResearchDiscoveryHit)
                .where(ResearchDiscoveryHit.id == hit_id)
                .values(**values)
            )

    async def _extract(
        self, job_id: str, hit_ids: list[str], model_ref: str, scope: Any, depth: str
    ) -> None:
        async with self.db.session() as session:
            hits = list(
                (
                    await session.execute(
                        select(ResearchDiscoveryHit).where(ResearchDiscoveryHit.id.in_(hit_ids))
                    )
                )
                .scalars()
                .all()
            )
        order = {h: n for n, h in enumerate(hit_ids)}
        hits.sort(key=lambda h: order[h.id])
        usage = {"promptTokens": 0, "outputTokens": 0, "thinkingTokens": 0, "totalTokens": 0}
        outcomes: list[Outcome] = []
        await self.research.progress(job_id, 0.05, "Checking what each site allows.")
        allowed: list[ResearchDiscoveryHit] = []
        for hit in hits:
            verdict = await self.robots.check(hit.url)
            if verdict.allowed:
                allowed.append(hit)
                continue
            outcomes.append(await self._restrict(job_id, hit, verdict.reason, "robots.txt"))
        texts = await self._texts(allowed, depth, outcomes)
        for number, hit in enumerate(allowed, 1):
            await self.research.progress(
                job_id,
                0.2 + 0.75 * (number - 1) / max(1, len(allowed)),
                f"Reading {number} of {len(allowed)}.",
            )
            got = texts.get(hit.id)
            if got is None:
                continue
            outcomes.append(await self._one(job_id, hit, got, model_ref, scope, usage))
            await self.research.set_job(job_id, usage=usage)
        ideas = [i for o in outcomes for i in o.idea_ids]
        await self.research.set_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"{len(ideas)} idea(s) saved from {len(outcomes)} page(s)."
            if ideas
            else "No ideas were saved. See the reason on each result.",
            usage=usage,
            result={
                "ideas": len(ideas),
                "ideaIds": ideas,
                "outcomes": [
                    {
                        "hitId": o.hit_id,
                        "status": o.status,
                        "detail": o.detail,
                        "sourceId": o.source_id,
                        "ideaIds": o.idea_ids,
                        "linked": o.linked,
                    }
                    for o in outcomes
                ],
            },
            finished_at=utcnow(),
        )

    # -- getting a page's text -------------------------------------------------------------

    async def _texts(
        self,
        hits: list[ResearchDiscoveryHit],
        depth: str,
        outcomes: list[Outcome],
    ) -> dict[str, tuple[htmltext.ParsedPage, str]]:
        """Page text for each allowed hit: ``(parsed page, how it was read)``. A hit that could
        not be read is settled here and left out."""
        out: dict[str, tuple[htmltext.ParsedPage, str]] = {}
        provider_hits = [
            h
            for h in hits
            if h.provider != "direct" and await self._policy(h, "fullText") == "provider_extract"
        ]
        if provider_hits and await self.web.resolve("extract") is not None:
            for start in range(0, len(provider_hits), 5):
                batch = provider_hits[start : start + 5]
                try:
                    _, result = await self._retry(
                        lambda batch=batch: self.web.extract(
                            [h.url for h in batch],
                            depth=depth,  # pyright: ignore[reportArgumentType]
                        )
                    )
                except (IntegrationError, Retryable) as exc:
                    for h in batch:
                        message = f"The page could not be read: {exc}"
                        await self._set_hit(h.id, "failed", message)
                        outcomes.append(Outcome(h.id, "failed", message))
                    continue
                pages = {canonical_url(p.url) or p.url: p for p in result.pages}
                reasons = {canonical_url(u) or u: r for u, r in result.failed}
                for h in batch:
                    key = h.canonical_url or h.url
                    page = pages.get(key)
                    if page is None:
                        message = "The service could not read this page: " + reasons.get(
                            key, "no content was returned."
                        )
                        await self._set_hit(h.id, "failed", message)
                        outcomes.append(Outcome(h.id, "failed", message))
                        continue
                    parsed = htmltext.parse_markdown(page.content)
                    parsed.title = parsed.title or h.title
                    out[h.id] = (parsed, "provider extract")
        async with asyncio.TaskGroup() as group:
            tasks = {
                h.id: group.create_task(self._direct_read(h))
                for h in hits
                if h.id not in out and not any(o.hit_id == h.id for o in outcomes)
            }
        for hit_id, task in tasks.items():
            result = task.result()
            if isinstance(result, str):
                await self._set_hit(hit_id, "failed", result)
                outcomes.append(Outcome(hit_id, "failed", result))
            else:
                out[hit_id] = (result, "direct fetch")
        return out

    async def _policy(self, hit: ResearchDiscoveryHit, name: str) -> str:
        if not hit.registry_id:
            return ""
        try:
            entry = await self.registry.get(hit.registry_id)
        except RegistryError:
            return ""
        return str(entry.retrieval_policy.get(name, ""))

    async def _direct_read(self, hit: ResearchDiscoveryHit) -> htmltext.ParsedPage | str:
        """A page read straight from its site, or the reason it was not."""
        async with self._fetch_gate:
            try:
                fetched = await self._retry(
                    lambda: self._fetch(
                        hit.url,
                        accept=("text/html", "application/xhtml+xml"),
                        max_bytes=MAX_HTML_BYTES,
                    )
                )
            except netfetch.FetchError as exc:
                return f"{exc}"
        parsed = htmltext.parse_html(fetched.data.decode("utf-8", errors="replace"))
        parsed.title = parsed.title or hit.title
        return parsed

    # -- one page --------------------------------------------------------------------------

    async def _restrict(
        self, job_id: str, hit: ResearchDiscoveryHit, reason: str, by: str
    ) -> Outcome:
        """Keep what the search told us (title, address, date) and nothing more."""
        source_id = await self._metadata_source(job_id, hit, reason, by)
        await self._set_hit(hit.id, "restricted", reason, source_id=source_id)
        return Outcome(hit.id, "restricted", reason, source_id)

    async def _metadata_source(
        self, job_id: str, hit: ResearchDiscoveryHit, reason: str, by: str
    ) -> str | None:
        async with self.db.session() as session:
            existing = await session.scalar(
                select(ResearchSource.id).where(ResearchSource.canonical_url == hit.canonical_url)
            )
            if existing:
                return existing
            source = ResearchSource(
                source_type="web_article",
                import_method="search",
                title=hit.title or hit.url,
                canonical_url=hit.canonical_url,
                published_at=hit.published,
                access_status="metadata_only",
                access_detail=reason,
                extraction_status="not_applicable",
                domain=hit.domain,
                registry_id=hit.registry_id,
                provenance={
                    "hitId": hit.id,
                    "jobId": job_id,
                    "retrievedAt": _iso(hit.retrieved_at),
                    "restrictedBy": by,
                    "provider": hit.provider,
                    "query": hit.query,
                },
            )
            session.add(source)
            try:
                await session.flush()
            except IntegrityError:
                return None
            return source.id

    async def _one(
        self,
        job_id: str,
        hit: ResearchDiscoveryHit,
        got: tuple[htmltext.ParsedPage, str],
        model_ref: str,
        scope: Any,
        usage: dict[str, int],
    ) -> Outcome:
        page, how = got
        text = page.text.strip()
        reason = htmltext.restriction(text)
        if reason:
            return await self._restrict(job_id, hit, reason, "page content")
        content_hash = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        async with self.db.session() as session:
            clash = await session.scalar(
                select(ResearchSource.id).where(
                    (ResearchSource.content_hash == content_hash)
                    | (ResearchSource.canonical_url == hit.canonical_url)
                )
            )
        if clash:
            await self._set_hit(hit.id, "duplicate", "Already in your library.", source_id=clash)
            return Outcome(hit.id, "duplicate", "Already in your library.", clash)
        source_id = await self._store_page(job_id, hit, page, text, content_hash, how)
        if source_id is None:
            await self._set_hit(hit.id, "duplicate", "Already in your library.")
            return Outcome(hit.id, "duplicate", "Already in your library.")
        try:
            rows, classification, summary, reasons = await self._ask(
                hit, page, text, model_ref, scope, usage
            )
        except LLMError as exc:
            await self._set_hit(hit.id, "failed", str(exc), source_id=source_id)
            return Outcome(hit.id, "failed", str(exc), source_id)
        except ex.ModelOutputError as exc:
            message = f"{exc} Nothing was saved from this page. Try again or choose another model."
            await self._set_hit(hit.id, "failed", message, source_id=source_id)
            await self._provenance(source_id, {"invalidModelOutput": exc.problems[:5]})
            return Outcome(hit.id, "failed", message, source_id)
        await self._classify(source_id, classification, summary)
        if classification == "not_research" or not rows:
            why = " ".join(reasons)[:500] or (
                "The page is not research."
                if classification == "not_research"
                else "No testable idea was found."
            )
            await self._set_hit(hit.id, "rejected", why, source_id=source_id)
            return Outcome(hit.id, "rejected", why, source_id)
        ids, linked = await self._save(job_id, source_id, rows, scope)
        detail = f"{len(ids)} idea(s) saved" + (
            f", {len(linked)} matched to known ideas." if linked else "."
        )
        await self._set_hit(hit.id, "extracted", detail, source_id=source_id)
        return Outcome(hit.id, "extracted", detail, source_id, ids, linked)

    async def _store_page(
        self,
        job_id: str,
        hit: ResearchDiscoveryHit,
        page: htmltext.ParsedPage,
        text: str,
        content_hash: str,
        how: str,
    ) -> str | None:
        markers = htmltext.injection_markers(text)
        async with self.db.session() as session:
            entry = await session.get(ResearchRegistryEntry, hit.registry_id or "")
            source = ResearchSource(
                source_type="web_article",
                import_method="search",
                title=(page.title or hit.title or hit.url)[:500],
                authors=page.authors,
                abstract=page.description,
                canonical_url=hit.canonical_url,
                published_at=(page.published or hit.published or "")[:32] or None,
                access_status="full_text",
                access_detail=f"Public page, read by {how} after robots.txt allowed it.",
                content_hash=content_hash,
                extraction_status="parsed",
                page_count=1,
                domain=hit.domain,
                registry_id=hit.registry_id,
                metadata_={"siteName": page.site_name, "truncated": page.truncated},
                provenance={
                    "hitId": hit.id,
                    "jobId": job_id,
                    "provider": hit.provider,
                    "query": hit.query,
                    "retrievedAt": _iso(utcnow()),
                    "searchRetrievedAt": _iso(hit.retrieved_at),
                    "readBy": how,
                    "robots": "allowed",
                    "registryTier": (entry.quality.get("tier") if entry else None),
                    "injectionMarkers": markers,
                },
            )
            session.add(source)
            try:
                await session.flush()
            except IntegrityError:
                return None
            session.add(
                ResearchDocument(
                    source_id=source.id,
                    pages=[{"page": 1, "text": text}],
                    sections=page.sections or [_whole(text)],
                    tables=[],
                    equations=[],
                    stats={"chars": len(text), "injectionMarkers": len(markers)},
                    parser="web-text-v1",
                )
            )
            return source.id

    async def _ask(
        self,
        hit: ResearchDiscoveryHit,
        page: htmltext.ParsedPage,
        text: str,
        model_ref: str,
        scope: Any,
        usage: dict[str, int],
    ) -> tuple[list[dict[str, Any]], str, str, list[str]]:
        sections = page.sections or [_whole(text)]
        chunks, _ = ex.chunk_document(
            text, sections, [(1, 0, len(text))], budget=ex.DEFAULT_CHAR_BUDGET
        )
        index = ex.PageIndex.build([{"page": 1, "text": text}])
        label = (
            "not chosen"
            if scope is None
            else f"{scope.instrument_type} {scope.region} delay {scope.delay} {scope.universe}"
        )
        rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        classification, summary = "", ""
        async with self.research.llm_gate:
            for chunk in chunks:
                answer = await self.research.llm.generate(
                    system=webextract.SYSTEM,
                    user=webextract.build_prompt(
                        title=page.title or hit.title,
                        url=hit.url,
                        scope=label,
                        text=chunk.text,
                        part=(chunk.index, len(chunks)),
                    ),
                    model_ref=model_ref,
                    response_schema=webextract.SCHEMA,
                    temperature=TEMPERATURE,
                )
                for k, v in answer.usage.items():
                    usage[k] = usage.get(k, 0) + int(v)
                parsed = webextract.parse_answer(answer.text)
                classification = classification or parsed.source_class
                summary = summary or parsed.summary
                if parsed.no_ideas_reason:
                    reasons.append(parsed.no_ideas_reason)
                if parsed.classification_reason:
                    reasons.append(parsed.classification_reason)
                rows.extend(webextract.web_idea_row(i, index=index) for i in parsed.ideas)
        return ex.merge_ideas(rows), classification or "unverified_claim", summary, reasons

    async def _classify(self, source_id: str, classification: str, summary: str) -> None:
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            if source is None:
                return
            source.source_class = classification
            source.provenance = {**source.provenance, "summary": summary[:600]}

    async def _provenance(self, source_id: str, extra: dict[str, Any]) -> None:
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            if source is not None:
                source.provenance = {**source.provenance, **extra}

    # -- the library -----------------------------------------------------------------------

    async def _save(
        self, job_id: str, source_id: str, rows: list[dict[str, Any]], scope: Any
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Add each idea to the library, or link its source to the idea it restates.

        Two ideas from one page stay two ideas. An idea another source already reports is not
        copied: this page's source is linked to it, with the passages it contributed, so both
        references survive.
        """
        created: list[str] = []
        linked: list[dict[str, Any]] = []
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            others = list(
                (
                    await session.execute(
                        select(ResearchIdea).where(ResearchIdea.source_id != source_id)
                    )
                )
                .scalars()
                .all()
            )
        if source is None:
            return created, linked
        entry_tier = (source.provenance or {}).get("registryTier")
        source_view = {
            "authors": source.authors,
            "published": source.published_at,
            "doi": source.doi,
            "venue": source.venue,
            "sourceClass": source.source_class,
            "registryTier": entry_tier,
        }
        pool = [(o, dedupe.view_row(o)) for o in others]
        for row in rows:
            print_ = dedupe.fingerprint(row)
            best: tuple[ResearchIdea | None, float, str] = (None, 0.0, "")
            for idea, mine in pool:
                if idea.fingerprint == print_:
                    best = (idea, 1.0, "fingerprint")
                    break
                score = dedupe.similarity(row, mine)
                if score > best[1]:
                    best = (idea, score, "similarity")
            match, score, how = best
            if match is not None and score >= dedupe.SAME:
                if await self._link(match.id, source_id, row, score, how, job_id):
                    linked.append(
                        {
                            "ideaId": match.id,
                            "title": match.title,
                            "similarity": round(score, 2),
                            "matchedBy": how,
                        }
                    )
                continue
            ranked = ranking.rank(
                row,
                source=source_view,
                assessment=None,
                others=[v for _, v in pool],
            )
            values = {k: v for k, v in row.items() if k != "ranking"}
            async with self.db.session() as session:
                idea = ResearchIdea(
                    source_id=source_id,
                    job_id=job_id,
                    status="new",
                    fingerprint=print_,
                    ranking=ranked,
                    **values,
                )
                session.add(idea)
                await session.flush()
                idea_id = idea.id
            created.append(idea_id)
            if scope is not None:
                await self._assess_and_rank(idea_id, scope, source_view, [v for _, v in pool])
            if match is not None and score >= dedupe.POSSIBLE:
                linked.append(
                    {
                        "ideaId": match.id,
                        "title": match.title,
                        "similarity": round(score, 2),
                        "matchedBy": "possible",
                    }
                )
        return created, [x for x in linked if x["matchedBy"] != "possible"]

    async def _link(
        self,
        idea_id: str,
        source_id: str,
        row: dict[str, Any],
        score: float,
        how: str,
        job_id: str,
    ) -> bool:
        async with self.db.session() as session:
            try:
                session.add(
                    ResearchIdeaLink(
                        idea_id=idea_id,
                        source_id=source_id,
                        matched_by=how,
                        similarity=round(score, 3),
                        evidence=[e for e in row.get("evidence", []) if e.get("verified")][:6],
                        their_title=str(row.get("title", ""))[:300],
                        job_id=job_id,
                    )
                )
                await session.flush()
            except IntegrityError:
                return False
        return True

    async def _assess_and_rank(
        self, idea_id: str, scope: Any, source_view: dict[str, Any], others: list[dict[str, Any]]
    ) -> None:
        try:
            view = await self.research.assess(idea_id, scope)
        except Exception:
            log.warning("research.discover_assess_failed", idea_id=idea_id, exc_info=True)
            return
        async with self.db.session() as session:
            idea = await session.get(ResearchIdea, idea_id)
            if idea is None:
                return
            row = {
                "required_variables": idea.required_variables,
                "claims": idea.claims,
                "methodology": idea.methodology,
                "evidence": idea.evidence,
                "evidence_scope": idea.evidence_scope,
                "support": idea.support,
                "universe": idea.universe,
                "holding_period": idea.holding_period,
                "lookback": idea.lookback,
                "signal_definition": idea.signal_definition,
                "title": idea.title,
            }
            idea.ranking = ranking.rank(
                row, source=source_view, assessment=view.get("brainAssessment"), others=others
            )

    # --- registry, topics ------------------------------------------------------------------

    @staticmethod
    def topics() -> list[str]:
        return list(queries.SUGGESTED_TOPICS)

    @staticmethod
    def propose(topic: str, market: str) -> list[dict[str, str]]:
        try:
            return [q.to_dict() for q in queries.propose(topic, market=market)]
        except queries.QueryError as exc:
            raise ResearchError("bad_query", str(exc), 422) from exc

    async def providers(self) -> list[dict[str, Any]]:
        """Which search services can run a discovery search, with their state."""
        out = []
        status = await self.web.status()
        for p in status["providers"]:
            reg = web_registry.REGISTRY[p["id"]]
            out.append(
                {
                    "id": p["id"],
                    "label": p["label"],
                    "configured": p["configured"],
                    "enabled": p["enabled"],
                    "isDefault": p["is_preferred"],
                    "documentSearch": isinstance(reg.build("placeholder-key"), DocumentSearch),
                    "usage": p["usage"],
                }
            )
        return [o for o in out if o["documentSearch"]]


def _whole(text: str) -> dict[str, Any]:
    return {
        "id": "s1",
        "heading": "Article",
        "level": 1,
        "start": 0,
        "end": len(text),
        "page_start": 1,
        "page_end": 1,
    }


def parse_feed(xml: str) -> list[dict[str, Any]]:
    """Items of an RSS or Atom document: title, link, summary, date. A document with a doctype
    or entity declaration is refused, since expansion tricks need one."""
    if "<!DOCTYPE" in xml[:4096].upper() or "<!ENTITY" in xml.upper()[:50_000]:
        raise netfetch.FetchError("bad_feed", "The feed contained unexpected markup.")
    try:
        root = ET.fromstring(xml)  # noqa: S314
    except ET.ParseError as exc:
        raise netfetch.FetchError("bad_feed", "The feed could not be read.") from exc
    out: list[dict[str, Any]] = []

    def text(node: ET.Element, *names: str) -> str:
        for name in names:
            for found in node.iter():
                if found.tag.rsplit("}", 1)[-1] == name and (found.text or "").strip():
                    return " ".join((found.text or "").split())
        return ""

    for item in root.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag not in {"item", "entry"}:
            continue
        link = text(item, "link")
        if not link:
            for child in item:
                if child.tag.rsplit("}", 1)[-1] == "link" and child.get("href"):
                    link = child.get("href") or ""
                    break
        if not link:
            continue
        published = text(item, "published", "updated", "pubDate", "date")
        out.append(
            {
                "title": text(item, "title"),
                "link": link,
                "summary": text(item, "summary", "description"),
                "published": _feed_date(published),
            }
        )
    return out


def _feed_date(raw: str) -> str | None:
    if not raw:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except TypeError, ValueError:
        return raw[:10] if raw[:4].isdigit() else None


__all__ = [
    "DiscoverRequest",
    "DiscoveryService",
    "ResearchError",
    "parse_feed",
]
