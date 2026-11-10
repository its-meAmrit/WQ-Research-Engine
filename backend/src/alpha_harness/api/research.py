"""Research Intelligence: papers in, source-grounded ideas out.

Independent of the simulation routes and of SPC. Nothing here reaches BRAIN or starts a
simulation; an analysis spends LLM requests, a search spends a registry request (and a
Tavily request when asked for), and a hand-off writes a local draft.
"""

from typing import Annotated

from fastapi import APIRouter, Query, Request

from ..catalog.queries import Tuple4
from ..research import extraction, pdf
from ..research.service import ResearchError, Started
from ..research.views import (
    AnalyzeRequest,
    CandidateImport,
    CandidateOut,
    IdeaDetail,
    IdeaOut,
    IdeaPatch,
    ImportRequest,
    JobOut,
    Limits,
    ModelChoice,
    PageOut,
    PreviewOut,
    ResearchOptions,
    ScopeIn,
    ScopeRequest,
    SearchOut,
    SourceDetail,
    SourceOut,
    StartedOut,
)
from .deps import State, refuse

router = APIRouter(prefix="/api/research", tags=["research"])

TOPICS = [
    "value", "momentum", "quality", "growth", "size", "volatility",
    "liquidity", "sentiment", "analyst", "event", "macro", "other",
]  # fmt: skip


def _refused(exc: ResearchError) -> Exception:
    return refuse(exc.status, exc.code, str(exc))


def _scope(scope: ScopeIn) -> Tuple4:
    return Tuple4(
        instrument_type=scope.instrument_type,
        region=scope.region,
        delay=scope.delay,
        universe=scope.universe,
    )


def _started(started: Started) -> StartedOut:
    return StartedOut(
        source_id=started.source_id,
        job_id=started.job_id,
        existing=started.existing,
        message=started.message,
    )


@router.get("/options")
async def options(state: State) -> ResearchOptions:
    """Models that can answer, the limits that apply, and whether web search is set up."""
    limits = state.research.limits
    web = False
    try:
        web = await state.web.resolve_search() is not None
    except Exception:  # noqa: BLE001 - an option, not a requirement
        web = False
    return ResearchOptions(
        models=[
            ModelChoice.model_validate(m | {"remaining_today": m["remainingToday"]})
            for m in await state.llm.runnable()
        ],
        limits=Limits(
            max_pdf_mb=limits.max_bytes // 1_048_576,
            max_pages=limits.max_pages,
            default_char_budget=extraction.DEFAULT_CHAR_BUDGET,
            max_ideas=extraction.MAX_IDEAS,
        ),
        web_search=web,
        topics=TOPICS,
    )


# --- papers ---------------------------------------------------------------------------------


@router.post("/papers/upload", status_code=202)
async def upload(
    request: Request, state: State, filename: Annotated[str, Query(max_length=255)] = "paper.pdf"
) -> StartedOut:
    """A PDF as the raw request body (``Content-Type: application/pdf``).

    Read as a stream and stopped at the size limit, so an oversized upload is refused without
    being held in memory.
    """
    limit = state.research.limits.max_bytes
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise refuse(
            413, "too_large", f"The file is larger than the {limit // 1_048_576} MB limit."
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise refuse(
                413, "too_large", f"The file is larger than the {limit // 1_048_576} MB limit."
            )
    try:
        return _started(await state.research.import_pdf(filename, bytes(body)))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/papers/import", status_code=202)
async def import_paper(body: ImportRequest, state: State) -> StartedOut:
    """A DOI, an arXiv id or a link. Registries supply the details; full text only when open."""
    try:
        return _started(await state.research.import_reference(body.reference))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.get("/papers")
async def list_papers(
    state: State,
    search: Annotated[str | None, Query(max_length=200)] = None,
    access: Annotated[
        str | None, Query(pattern="^(full_text|abstract_only|metadata_only)$")
    ] = None,
    source: Annotated[str | None, Query(pattern="^(upload|doi|arxiv|url|search)$")] = None,
    topic: str | None = None,
    readiness: Annotated[str | None, Query(pattern="^(ready|adaptation|blocked)$")] = None,
    saved: bool | None = None,
    published_from: Annotated[str | None, Query(alias="publishedFrom", max_length=10)] = None,
) -> list[SourceOut]:
    rows = await state.research.list_sources(
        search=search,
        access=access,
        imported_from=source,
        topic=topic,
        readiness=readiness,
        saved=saved,
        published_from=published_from,
    )
    return [SourceOut.model_validate(r) for r in rows]


@router.get("/papers/{source_id}")
async def get_paper(source_id: str, state: State) -> SourceDetail:
    try:
        return SourceDetail.model_validate(await state.research.get_source(source_id))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.delete("/papers/{source_id}", status_code=204)
async def delete_paper(source_id: str, state: State) -> None:
    """Remove a paper with its stored text, ideas and jobs."""
    try:
        await state.research.delete_source(source_id)
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.get("/papers/{source_id}/pages/{page}")
async def get_page(source_id: str, page: int, state: State) -> PageOut:
    """The stored text of one page, to read a cited passage in its place."""
    try:
        return PageOut.model_validate(await state.research.get_pages(source_id, page))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.get("/papers/{source_id}/preview")
async def preview(
    source_id: str,
    state: State,
    char_budget: Annotated[int | None, Query(alias="charBudget", ge=2_000, le=200_000)] = None,
) -> PreviewOut:
    """What an analysis would send and cost. Nothing is sent."""
    try:
        return PreviewOut.model_validate(
            await state.research.preview(source_id, budget=char_budget)
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/papers/{source_id}/analyze", status_code=202)
async def analyze(source_id: str, body: AnalyzeRequest, state: State) -> StartedOut:
    """Start the model-assisted analysis. Returns at once; follow the job."""
    try:
        return _started(
            await state.research.analyze(
                source_id,
                scope=_scope(body.scope) if body.scope else None,
                model_ref=body.model_ref,
                budget=body.char_budget,
                force=body.force,
            )
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


# --- search ---------------------------------------------------------------------------------


@router.get("/search")
async def search(
    q: Annotated[str, Query(min_length=3, max_length=300)],
    state: State,
    semantic: bool = False,
    web: bool = False,
) -> SearchOut:
    """Academic keyword search (OpenAlex, arXiv; Semantic Scholar if asked). ``web`` adds
    Tavily results as supplementary reading, kept apart from the papers."""
    try:
        return SearchOut.model_validate(
            await state.research.search(q, semantic=semantic, include_web=web)
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/search/import", status_code=202)
async def import_result(body: CandidateImport, state: State) -> StartedOut:
    """Import a search result by its DOI, arXiv id or link, in that order of preference."""
    reference = body.doi or body.arxiv_id or body.url
    if not reference:
        raise refuse(
            422, "bad_identifier", "That result has no DOI, arXiv id or link to import by."
        )
    try:
        return _started(await state.research.import_reference(reference))
    except ResearchError as exc:
        raise _refused(exc) from exc


# --- jobs -----------------------------------------------------------------------------------


@router.get("/jobs")
async def list_jobs(
    state: State, source_id: Annotated[str | None, Query(alias="sourceId")] = None
) -> list[JobOut]:
    return [JobOut.model_validate(j) for j in await state.research.list_jobs(source_id=source_id)]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, state: State) -> JobOut:
    try:
        return JobOut.model_validate(await state.research.get_job(job_id))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, state: State) -> JobOut:
    try:
        return JobOut.model_validate(await state.research.cancel_job(job_id))
    except ResearchError as exc:
        raise _refused(exc) from exc


# --- ideas ----------------------------------------------------------------------------------


@router.get("/ideas")
async def list_ideas(
    state: State,
    source_id: Annotated[str | None, Query(alias="sourceId")] = None,
    status: Annotated[str | None, Query(pattern="^(new|saved|rejected)$")] = None,
    readiness: Annotated[str | None, Query(pattern="^(ready|adaptation|blocked)$")] = None,
    topic: str | None = None,
    support: Annotated[
        str | None, Query(pattern="^(source_backed|partially_supported|speculative)$")
    ] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    domain: Annotated[str | None, Query(max_length=255)] = None,
    source_class: Annotated[str | None, Query(alias="sourceClass", max_length=24)] = None,
    source_type: Annotated[str | None, Query(alias="sourceType", max_length=24)] = None,
    kind: Annotated[str | None, Query(max_length=24)] = None,
    favorite: bool | None = None,
    reviewed: bool | None = None,
) -> list[IdeaOut]:
    rows = await state.research.list_ideas(
        domain=domain,
        source_class=source_class,
        source_type=source_type,
        kind=kind,
        favorite=favorite,
        reviewed=reviewed,
        source_id=source_id,
        status=status,
        readiness=readiness,
        topic=topic,
        support=support,
        search=search,
    )
    return [IdeaOut.model_validate(r) for r in rows]


@router.get("/ideas/{idea_id}")
async def get_idea(idea_id: str, state: State) -> IdeaDetail:
    try:
        return IdeaDetail.model_validate(await state.research.get_idea(idea_id))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.patch("/ideas/{idea_id}")
async def update_idea(idea_id: str, body: IdeaPatch, state: State) -> IdeaDetail:
    """Save, reject or annotate an idea."""
    try:
        return IdeaDetail.model_validate(
            await state.research.update_idea(
                idea_id,
                status=body.status,
                note=body.note,
                favorite=body.favorite,
                reviewed=body.reviewed,
            )
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/ideas/{idea_id}/assess")
async def assess(idea_id: str, body: ScopeRequest, state: State) -> IdeaDetail:
    """Check an idea's data and operators against one BRAIN scope. Reads the local catalogue
    and the cached operator list; nothing is sent to BRAIN."""
    try:
        return IdeaDetail.model_validate(await state.research.assess(idea_id, _scope(body.scope)))
    except ResearchError as exc:
        raise _refused(exc) from exc


@router.post("/ideas/{idea_id}/handoff", status_code=201)
async def hand_off(idea_id: str, body: ScopeRequest, state: State) -> CandidateOut:
    """Record a draft for the Alpha Builder. A draft only: no expression, no simulation."""
    try:
        return CandidateOut.model_validate(
            await state.research.hand_off(idea_id, _scope(body.scope))
        )
    except ResearchError as exc:
        raise _refused(exc) from exc


_ = pdf  # the limits live on the service; imported so a change there is seen by this module
