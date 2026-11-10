"""Discover: find public research, read what the person picks, and save its ideas.

Searching reads results only. Reading a page is a separate request the person makes, and
neither reaches BRAIN or starts a simulation. Progress and cancellation use the research
job routes (``/api/research/jobs``).
"""

from fastapi import APIRouter

from ..catalog.queries import Tuple4
from ..research.discover import queries as discover_queries
from ..research.discover import registry as source_registry
from ..research.discover import service as discover_service
from ..research.discover.service import DiscoverRequest
from ..research.discover.views import (
    DiscoverLimits,
    DiscoverOptions,
    DiscoverStarted,
    EstimateOut,
    ExtractIn,
    ExtractStarted,
    FeedIn,
    HitOut,
    QueryProposal,
    QueryProposalIn,
    RegistryEntryOut,
    RegistryPatch,
    SearchProviderOut,
)
from ..research.service import ResearchError
from ..research.views import ModelChoice
from .deps import State, refuse

router = APIRouter(prefix="/api/research/discover", tags=["research"])


def _refused(exc: ResearchError | source_registry.RegistryError) -> Exception:
    return refuse(exc.status, exc.code, str(exc))


@router.get("/options")
async def options(state: State) -> DiscoverOptions:
    """Suggested topics, the source registry, search services that can run, and the limits."""
    service = state.discovery
    providers = await service.providers()
    return DiscoverOptions(
        topics=service.topics(),
        sources=[
            RegistryEntryOut.model_validate(source_registry.view(e))
            for e in await state.registry.entries()
        ],
        providers=[
            SearchProviderOut.model_validate(
                {
                    "id": p["id"],
                    "label": p["label"],
                    "configured": p["configured"],
                    "enabled": p["enabled"],
                    "isDefault": p["isDefault"],
                    "usage": p["usage"],
                }
            )
            for p in providers
        ],
        models=[
            ModelChoice.model_validate(m | {"remaining_today": m["remainingToday"]})
            for m in await state.llm.runnable()
        ],
        limits=DiscoverLimits(
            max_results=discover_service.MAX_RESULTS,
            max_queries=discover_queries.MAX_QUERIES,
            max_requests_per_job=discover_service.MAX_REQUESTS_PER_JOB,
            max_extract_per_job=discover_service.MAX_EXTRACT_PER_JOB,
        ),
    )


@router.post("/queries")
async def propose_queries(body: QueryProposalIn, state: State) -> list[QueryProposal]:
    """Focused search queries for a topic. The person edits them before anything is sent."""
    try:
        return [
            QueryProposal.model_validate(q)
            for q in state.discovery.propose(body.topic, body.market)
        ]
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/estimate")
async def estimate(body: DiscoverRequest, state: State) -> EstimateOut:
    """What a search would cost and which services it would use. Nothing is sent."""
    try:
        return EstimateOut.model_validate(await state.discovery.estimate(body))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("", status_code=202)
async def start(body: DiscoverRequest, state: State) -> DiscoverStarted:
    """Start a discovery search. Returns at once; follow the job."""
    try:
        return DiscoverStarted.model_validate(await state.discovery.start(body))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.get("/jobs/{job_id}/hits")
async def hits(job_id: str, state: State) -> list[HitOut]:
    return [HitOut.model_validate(h) for h in await state.discovery.hits(job_id)]


@router.post("/extract", status_code=202)
async def extract(body: ExtractIn, state: State) -> ExtractStarted:
    """Read the chosen results (where their sites allow it) and extract ideas from them."""
    scope = (
        Tuple4(
            instrument_type=body.scope.instrument_type,
            region=body.scope.region,
            delay=body.scope.delay,
            universe=body.scope.universe,
        )
        if body.scope
        else None
    )
    try:
        return ExtractStarted.model_validate(
            await state.discovery.start_extract(
                body.hit_ids, model_ref=body.model_ref, scope=scope, depth=body.depth
            )
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


# --- source registry ------------------------------------------------------------------------


@router.get("/sources")
async def list_sources(state: State) -> list[RegistryEntryOut]:
    return [
        RegistryEntryOut.model_validate(source_registry.view(e))
        for e in await state.registry.entries()
    ]


@router.patch("/sources/{entry_id}")
async def update_source(entry_id: str, body: RegistryPatch, state: State) -> RegistryEntryOut:
    try:
        row = await state.registry.update(
            entry_id,
            enabled=body.enabled,
            min_interval_seconds=body.min_interval_seconds,
            domains=body.domains,
            quality_note=body.quality_note,
        )
    except source_registry.RegistryError as exc:
        raise _refused(exc) from exc
    return RegistryEntryOut.model_validate(source_registry.view(row))


@router.post("/sources", status_code=201)
async def add_feed(body: FeedIn, state: State) -> RegistryEntryOut:
    """Add an RSS or Atom feed. Read through the same public-address checks as any page."""
    try:
        row = await state.registry.add_feed(
            body.label, body.feed_url, min_interval_seconds=body.min_interval_seconds
        )
    except source_registry.RegistryError as exc:
        raise _refused(exc) from exc
    return RegistryEntryOut.model_validate(source_registry.view(row))


@router.delete("/sources/{entry_id}", status_code=204)
async def remove_source(entry_id: str, state: State) -> None:
    try:
        await state.registry.remove(entry_id)
    except source_registry.RegistryError as exc:
        raise _refused(exc) from exc
