"""SPC Research Lab: prompts for WorldQuant's Systematic Prediction Challenge.

Independent of the simulation routes: nothing here reaches BRAIN, and nothing here submits
anything to WorldQuant. A run spends LLM requests (and search requests, when configured), never
a simulation slot.
"""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import Field

from ..schemas import Out
from ..spc.errors import SpcConflictError, SpcError, SpcInvalidError, SpcNotFoundError
from ..spc.prompt_generation import GenerationOptions, PromptCandidate, generate
from ..spc.views import (
    Analytics,
    ExportOut,
    Overlap,
    PromptCreate,
    PromptDetail,
    PromptOut,
    PromptPatch,
    RunBrief,
    RunDetail,
    RunRequest,
    SpcOptions,
    VersionComparison,
)
from .deps import State, refuse

router = APIRouter(prefix="/api/spc", tags=["spc"])

_STATUS: dict[type[SpcError], int] = {
    SpcNotFoundError: 404,
    SpcConflictError: 409,
    SpcInvalidError: 422,
}


def _refused(exc: SpcError) -> Exception:
    return refuse(_STATUS.get(type(exc), 400), exc.code, str(exc))


class OverlapRequest(Out):
    run_ids: list[int] = Field(min_length=2, max_length=10)


class Generated(Out):
    candidates: list[PromptCandidate]


@router.get("/options")
async def options(state: State) -> SpcOptions:
    """Models you can run, the documented limits, research status and schema notes."""
    return await state.spc.options()


@router.post("/generate")
async def generate_prompts(body: GenerationOptions) -> Generated:
    """Candidate prompts from an idea and options. Free: no model call, nothing saved."""
    try:
        return Generated(candidates=generate(body))
    except ValueError as exc:
        raise refuse(422, "spc_invalid", str(exc)) from exc


@router.get("/prompts")
async def list_prompts(state: State, archived: bool = False) -> list[PromptOut]:
    return await state.spc.list_prompts(archived=archived)


@router.post("/prompts", status_code=201)
async def create_prompt(body: PromptCreate, state: State) -> PromptDetail:
    try:
        return await state.spc.create_prompt(body)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: int, state: State) -> PromptDetail:
    try:
        return await state.spc.get_prompt(prompt_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.patch("/prompts/{prompt_id}")
async def update_prompt(prompt_id: int, body: PromptPatch, state: State) -> PromptDetail:
    """Edit details, or save a new text as a new version. Earlier versions never change."""
    try:
        return await state.spc.update_prompt(prompt_id, body)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/prompts/{prompt_id}/duplicate", status_code=201)
async def duplicate_prompt(prompt_id: int, state: State) -> PromptDetail:
    try:
        return await state.spc.duplicate_prompt(prompt_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/prompts/{prompt_id}/versions/{number}/restore", status_code=201)
async def restore_version(prompt_id: int, number: int, state: State) -> PromptDetail:
    try:
        return await state.spc.restore_version(prompt_id, number)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.get("/prompts/{prompt_id}/compare")
async def compare_versions(
    prompt_id: int,
    state: State,
    a: Annotated[int, Query(ge=1)],
    b: Annotated[int, Query(ge=1)],
) -> VersionComparison:
    try:
        return await state.spc.compare(prompt_id, a, b)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/prompts/{prompt_id}/test", status_code=202)
async def test_prompt(prompt_id: int, body: RunRequest, state: State) -> RunDetail:
    """Queue a real run and return it at once. Follow it with ``GET /runs/{id}``."""
    try:
        return await state.spc.start_run(prompt_id, body)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.get("/prompts/{prompt_id}/export")
async def export_prompt(
    prompt_id: int, state: State, run_id: Annotated[int | None, Query()] = None
) -> ExportOut:
    try:
        return await state.spc.export(prompt_id, run_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.get("/runs")
async def list_runs(
    state: State,
    prompt_id: Annotated[int | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RunBrief]:
    return await state.spc.list_runs(prompt_id=prompt_id, limit=limit)


@router.get("/runs/{run_id}")
async def get_run(run_id: int, state: State) -> RunDetail:
    try:
        return await state.spc.get_run(run_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: int, state: State) -> RunDetail:
    try:
        return await state.spc.cancel_run(run_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/runs/{run_id}/correct", status_code=202)
async def correct_run(run_id: int, body: RunRequest, state: State) -> RunDetail:
    """A new run that shows the model its earlier answer's errors. The original is kept."""
    try:
        return await state.spc.correct_run(run_id, body)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.get("/runs/{run_id}/analytics")
async def run_analytics(run_id: int, state: State) -> Analytics:
    try:
        return await state.spc.analytics(run_id)
    except SpcError as exc:
        raise _refused(exc) from exc


@router.post("/overlap")
async def run_overlap(body: OverlapRequest, state: State) -> Overlap:
    try:
        return await state.spc.overlap(body.run_ids)
    except SpcError as exc:
        raise _refused(exc) from exc
