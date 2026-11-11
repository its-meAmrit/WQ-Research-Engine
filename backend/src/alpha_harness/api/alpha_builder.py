"""Alpha Builder: an approved research idea, written as reviewable alpha candidates.

Nothing here simulates. Designing, generating, validating, saving and previewing read local
metadata and call the configured model; creating a task stores an *idle* Paste Alphas task that
the person starts themselves in Tasks. Prediction-challenge prompts are a separate product
and are not reachable from here.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import Field

from ..alpha_builder import design as dsn
from ..research.service import ResearchError
from ..research.views import ScopeIn
from ..schemas import Out
from .deps import State
from .research import refused, scope_of

router = APIRouter(prefix="/api/research/builder", tags=["alpha-builder"])


class BuilderOptions(Out):
    #: BRAIN's own settings schema, as cached: the choices for every setting.
    settings_schema: dict[str, Any]
    #: Markets whose fields are downloaded: the only scopes a candidate can be built in.
    synced_scopes: list[ScopeIn]
    max_variants: int
    default_variants: int = dsn.DEFAULT_VARIANTS
    variant_kinds: list[str] = Field(default_factory=lambda: list(dsn.VARIANT_KINDS))


class BuilderDesign(Out):
    id: str
    idea_id: str
    scope: dict[str, Any]
    design: dict[str, Any]
    #: One entry per variable: status, selected field, candidates, rationale, approximation.
    mapping: list[dict[str, Any]]
    substitutions: dict[str, str]
    accepted_proxies: list[str]
    model_ref: str | None
    job_id: str | None
    created_at: str | None
    #: Variables nothing usable stands for yet, with why.
    blocked: list[dict[str, Any]]


class BuilderCandidate(Out):
    id: str
    idea_id: str
    expression: str | None
    scope: dict[str, Any]
    validation_outcome: str
    validation_detail: dict[str, Any] | None
    version: int
    status: str
    provenance: dict[str, Any]
    created_at: str | None
    design_id: str | None
    parent_id: str | None
    variant_kind: str | None
    settings: dict[str, Any] | None
    explanation: dict[str, Any] | None
    changes: list[str]
    limitations: list[str]
    fields_used: list[str]
    operators_used: list[str]
    saved_at: str | None
    study_id: int | None
    note: str


class BuilderJobStarted(Out):
    job_id: str
    message: str


class BuilderFieldHit(Out):
    field_id: str
    dataset_id: str | None
    dataset_name: str | None
    description: str
    field_type: str | None
    coverage: float | None
    category: str | None
    point_in_time: str


class BuilderDesignRequest(Out):
    scope: ScopeIn
    model_ref: str | None = None


class BuilderMappingPatch(Out):
    #: ``{variable name: field id}``; an empty id removes a substitution. Omit to leave as is.
    substitutions: dict[str, str] | None = None
    accepted_proxies: list[str] | None = None


class BuilderVariantsRequest(Out):
    #: Every setting, complete. Nothing is filled in on the server.
    settings: dict[str, Any]
    #: The person confirmed these settings on screen; required.
    settings_confirmed: bool = False
    count: int = Field(default=dsn.DEFAULT_VARIANTS, ge=1, le=dsn.MAX_VARIANTS)
    kinds: list[str] | None = None
    model_ref: str | None = None


class BuilderEditRequest(Out):
    expression: str | None = Field(default=None, max_length=4000)
    settings: dict[str, Any] | None = None
    note: str = Field(default="", max_length=2000)


class BuilderFieldSearch(Out):
    scope: ScopeIn
    text: str = Field(min_length=1, max_length=120)


class BuilderRefineRequest(Out):
    model_ref: str | None = None


class BuilderPreviewRequest(Out):
    candidate_ids: list[str] = Field(min_length=1, max_length=20)
    cores: int | None = Field(default=None, ge=1, le=64)


class BuilderTaskRequest(BuilderPreviewRequest):
    #: One per click of the button; a repeat with the same key returns the first task.
    idempotency_key: str = Field(min_length=8, max_length=64)
    #: Must be true: the person confirmed the preview.
    confirm: bool = False
    acknowledge_review: bool = False


class BuilderPreviewItem(Out):
    candidate_id: str
    version: int
    expression: str | None
    scope: dict[str, Any]
    settings: dict[str, Any] | None
    labels: dict[str, str]
    outcome: str
    needs_review: bool
    review_reasons: list[str]
    blockers: list[str]
    fields: list[str]
    runnable: bool


class BuilderPreview(Out):
    items: list[BuilderPreviewItem]
    simulations: int
    batches: int
    problems: list[str]
    needs_review: bool
    max_cores: int | None
    quota: str
    #: Always false: a preview never starts anything.
    starts: bool


class BuilderTask(Out):
    task_id: int
    name: str
    candidate_ids: list[str]
    existing: bool
    #: Always false: the task is stored idle and started by the person in Tasks.
    started: bool


class BuilderResults(Out):
    candidate_id: str
    version: int
    idea_id: str
    settings: dict[str, Any] | None
    expression: str | None
    #: no_task | not_started | waiting | failed | complete
    state: str
    message: str
    task_id: int | None = None
    task_status: str | None = None
    trials: list[dict[str, Any]]
    #: BRAIN's metrics, as stored; null for a number BRAIN did not report.
    metrics: dict[str, Any] | None
    checks: list[dict[str, Any]]
    settings_mismatch: list[str] = Field(default_factory=list)
    #: Rule-based readings of BRAIN's numbers, each naming the number it reads.
    diagnostics: list[dict[str, Any]]
    source: str | None = None


class BuilderLineage(Out):
    idea_id: str
    versions: int
    expressions: int
    tested: list[dict[str, Any]]
    multiple_testing_warning: str | None
    note: str


@router.get("/options")
async def options(state: State) -> BuilderOptions:
    """BRAIN's settings choices and the markets that are downloaded."""
    found = await state.builder.options()
    return BuilderOptions.model_validate(
        {
            "settingsSchema": found["settingsSchema"],
            "syncedScopes": [
                {
                    "instrumentType": s["instrumentType"],
                    "region": s["region"],
                    "delay": s["delay"],
                    "universe": s["universe"],
                }
                for s in found["syncedScopes"]
            ],
            "maxVariants": found["maxVariants"],
        }
    )


@router.get("/ideas/{idea_id}/designs")
async def designs(idea_id: str, state: State) -> list[BuilderDesign]:
    return [BuilderDesign.model_validate(d) for d in await state.builder.list_designs(idea_id)]


@router.get("/ideas/{idea_id}/candidates")
async def candidates(idea_id: str, state: State) -> list[BuilderCandidate]:
    return [
        BuilderCandidate.model_validate(c) for c in await state.builder.list_candidates(idea_id)
    ]


@router.post("/ideas/{idea_id}/design", status_code=202)
async def start_design(idea_id: str, body: BuilderDesignRequest, state: State) -> BuilderJobStarted:
    """Ask the model for a signal design for this idea in this exact scope. Spends one LLM
    request; sends nothing to BRAIN. Follow the job under ``/api/research/jobs``."""
    try:
        return BuilderJobStarted.model_validate(
            await state.builder.start_design(idea_id, scope_of(body.scope), body.model_ref)
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.get("/designs/{design_id}")
async def get_design(design_id: str, state: State) -> BuilderDesign:
    try:
        row = await state.builder.design_row(design_id)
        return BuilderDesign.model_validate(await state.builder.design_view(row))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.patch("/designs/{design_id}/mapping")
async def patch_mapping(design_id: str, body: BuilderMappingPatch, state: State) -> BuilderDesign:
    """Substitute a field or accept a proxy. A field outside the scope is refused."""
    try:
        return BuilderDesign.model_validate(
            await state.builder.update_mapping(design_id, body.substitutions, body.accepted_proxies)
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/fields/search")
async def search_fields(body: BuilderFieldSearch, state: State) -> list[BuilderFieldHit]:
    """Fields in exactly this scope, to choose a substitution from."""
    try:
        return [
            BuilderFieldHit.model_validate(h)
            for h in await state.builder.search_fields(scope_of(body.scope), body.text)
        ]
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/designs/{design_id}/variants", status_code=202)
async def start_variants(
    design_id: str, body: BuilderVariantsRequest, state: State
) -> BuilderJobStarted:
    """Write candidate expressions from a reviewed design. Drafts only; one LLM request."""
    try:
        return BuilderJobStarted.model_validate(
            await state.builder.start_variants(
                design_id,
                settings=body.settings,
                settings_confirmed=body.settings_confirmed,
                count=body.count,
                kinds=body.kinds,
                model_ref=body.model_ref,
            )
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str, state: State) -> BuilderCandidate:
    try:
        return BuilderCandidate.model_validate(await state.builder.get_candidate(candidate_id))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/candidates/{candidate_id}/check")
async def check(candidate_id: str, body: BuilderEditRequest, state: State) -> dict[str, Any]:
    """Validate an edit without keeping it. Free; nothing is stored."""
    try:
        return await state.builder.check_edit(candidate_id, body.expression, body.settings)
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/candidates/{candidate_id}/revalidate")
async def revalidate(candidate_id: str, state: State) -> BuilderCandidate:
    try:
        return BuilderCandidate.model_validate(await state.builder.revalidate(candidate_id))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/candidates/{candidate_id}/revise", status_code=201)
async def revise(candidate_id: str, body: BuilderEditRequest, state: State) -> BuilderCandidate:
    """An edit is a new version; the candidate it came from is left as it was."""
    try:
        return BuilderCandidate.model_validate(
            await state.builder.revise(
                candidate_id, expression=body.expression, settings=body.settings, note=body.note
            )
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/candidates/{candidate_id}/save")
async def save(candidate_id: str, state: State) -> BuilderCandidate:
    try:
        return BuilderCandidate.model_validate(await state.builder.save(candidate_id))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/preview")
async def preview(body: BuilderPreviewRequest, state: State) -> BuilderPreview:
    """The task exactly as it would be made. Read-only: simulates nothing, stores nothing."""
    try:
        return BuilderPreview.model_validate(await state.builder.preview(body.candidate_ids))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/tasks", status_code=201)
async def create_task(body: BuilderTaskRequest, state: State) -> BuilderTask:
    """Store an idle task for saved candidates. It starts only when the person starts it."""
    try:
        return BuilderTask.model_validate(
            await state.builder.create_task(
                body.candidate_ids,
                key=body.idempotency_key,
                confirm=body.confirm,
                acknowledge_review=body.acknowledge_review,
                cores=body.cores,
            )
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.get("/candidates/{candidate_id}/results")
async def results(candidate_id: str, state: State) -> BuilderResults:
    try:
        return BuilderResults.model_validate(await state.builder.results(candidate_id))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.get("/candidates/{candidate_id}/lineage")
async def lineage(candidate_id: str, state: State) -> BuilderLineage:
    try:
        return BuilderLineage.model_validate(await state.builder.lineage(candidate_id))
    except ResearchError as exc:
        raise refused(exc) from exc


@router.post("/candidates/{candidate_id}/refine", status_code=202)
async def refine(candidate_id: str, body: BuilderRefineRequest, state: State) -> BuilderJobStarted:
    """Ask for refinement drafts from a simulated candidate's real results."""
    try:
        return BuilderJobStarted.model_validate(
            await state.builder.start_refinement(candidate_id, body.model_ref)
        )
    except ResearchError as exc:
        raise refused(exc) from exc


@router.get("/candidates/{candidate_id}/export")
async def export(candidate_id: str, state: State) -> dict[str, Any]:
    try:
        return await state.builder.export(candidate_id)
    except ResearchError as exc:
        raise refused(exc) from exc
