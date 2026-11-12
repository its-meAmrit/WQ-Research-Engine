"""Super Alpha Research (Advanced Mode): the operator catalog, task creation, and a task's
candidates, coverage and dashboard. Start, pause, resume and stop are the Tasks API's own."""

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import Field

from ..brain.platform import (
    MULTI_SIMULATION_SIZE,
    REGION_AGNOSTIC_REGION,
    SUPER_ALPHA_MIN_SELECTION,
    SUPER_ALPHA_SLOTS,
)
from ..db.models import Study, utcnow
from ..labs.launch import AddedTask, add_study
from ..labs.params import SUPER_RESEARCH, SuperResearchParams, params_of
from ..labs.super_research import coverage, engine, report, scoring, templates
from ..labs.super_research.checker import check
from ..labs.super_research.registry import Ctx
from ..schemas import Out
from .deps import State, refuse

router = APIRouter(prefix="/api/super-lab/advanced", tags=["super-lab"])

Strategy = Literal["templates", "grammar", "evolve", "mixed"]
Objective = Literal["balanced", "sharpe", "fitness", "robust"]


class ResearchTask(Out):
    region: str
    delay: int
    universe: str
    neutralization: str
    selection_limit: int = Field(default=30, ge=SUPER_ALPHA_MIN_SELECTION, le=200)
    activation: Literal["IS", "OS", "BOTH"] = "BOTH"
    decays: list[int] = Field(default_factory=lambda: [4, 10], min_length=1, max_length=6)
    truncations: list[float] = Field(default_factory=lambda: [0.05], min_length=1, max_length=4)
    #: The simulation budget: how many SuperAlpha simulations the task may spend.
    simulations: int = Field(default=60, ge=1, le=2000)
    cores: int = Field(default=SUPER_ALPHA_SLOTS, ge=1, le=SUPER_ALPHA_SLOTS)
    objective: Objective = "balanced"
    strategy: Strategy = "mixed"
    max_depth: int = Field(default=5, ge=1, le=8)
    max_operators: int = Field(default=14, ge=3, le=30)
    max_variables: int = Field(default=3, ge=0, le=6)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    exploration: float = Field(default=0.5, ge=0, le=1)
    mutation_probability: float = Field(default=0.6, ge=0, le=1)
    disabled_operators: list[str] = Field(default_factory=list, max_length=200)
    disabled_families: list[str] = Field(default_factory=list, max_length=50)
    preferred_families: list[str] = Field(default_factory=list, max_length=50)
    patience: int = Field(default=4, ge=1, le=20)
    relaxed_scope: bool = False
    run: bool = False

    def checked(self) -> None:
        for decay in self.decays:
            if not 0 <= decay <= 512:
                raise refuse(422, "decay_range", "Decay must be between 0 and 512.")
        for value in self.truncations:
            if not 0 <= value <= 1:
                raise refuse(422, "truncation_range", "Truncation must be between 0 and 1.")
        if self.region == REGION_AGNOSTIC_REGION:
            raise refuse(422, "region_agnostic", "SuperAlphas combine Alphas of one region.")


class Proposal(Out):
    selection: str = Field(min_length=1, max_length=8000)
    combo: str = Field(min_length=1, max_length=8000)
    note: str = Field(default="", max_length=500)
    decay: int | None = Field(default=None, ge=0, le=512)
    truncation: float | None = Field(default=None, ge=0, le=1)
    activation: Literal["IS", "OS"] | None = None


class Validate(Out):
    selection: str | None = None
    combo: str | None = None
    relaxed_scope: bool = False


def _activations(choice: str) -> list[str]:
    return ["IS", "OS"] if choice == "BOTH" else [choice]


def _run_of(body: ResearchTask) -> SuperResearchParams:
    return SuperResearchParams(
        region=body.region,
        delay=body.delay,
        cores=body.cores,
        universe=body.universe,
        neutralization=body.neutralization,
        selection_limit=body.selection_limit,
        activation=_activations(body.activation),
        decays=body.decays,
        truncations=body.truncations,
        objective=body.objective,
        strategy=body.strategy,
        max_depth=body.max_depth,
        max_operators=body.max_operators,
        max_variables=body.max_variables,
        seed=body.seed,
        exploration=body.exploration,
        mutation_probability=body.mutation_probability,
        disabled_operators=sorted(set(body.disabled_operators)),
        disabled_families=sorted(set(body.disabled_families)),
        preferred_families=sorted(set(body.preferred_families)),
        patience=body.patience,
        relaxed_scope=body.relaxed_scope,
    )


@router.get("/operators")
async def operators(
    state: State,
    relaxed_scope: bool = False,
    disabled: str = "",
    families_off: str = "",
    examples: bool = False,
) -> dict[str, Any]:
    """The full catalog: every documented operator, its context, signature, validation status
    and whether and why the generator can use it. Verified against the operators BRAIN last
    published to this account, when they have been read."""
    run = SuperResearchParams(region="", delay=0, relaxed_scope=relaxed_scope)
    registry = await engine.registry_for(state.optimizer, run)
    off = frozenset(x for x in disabled.split(",") if x)
    fams = frozenset(x for x in families_off.split(",") if x)
    rows = coverage.build(registry, [], disabled=off, families_off=fams, examples=examples)
    brain = await state.optimizer.metadata.cached_operators()
    return {
        "operators": coverage.rows_as_dicts(rows),
        "summary": coverage.summary(rows),
        "families": sorted({r.family for r in rows}),
        "verifiedAgainstBrain": bool(brain),
        "note": (
            None
            if brain
            else "BRAIN's operator list has not been read yet, so only operators the Classic "
            "recipes already run are verified. Sign in and sync operators to verify the rest."
        ),
    }


@router.post("/validate")
async def validate(body: Validate, state: State) -> dict[str, Any]:
    """Run the local checker on an expression. Local acceptance is not BRAIN's."""
    run = SuperResearchParams(region="", delay=0, relaxed_scope=body.relaxed_scope)
    registry = await engine.registry_for(state.optimizer, run)
    out: dict[str, Any] = {}
    for label, text, ctx in (
        ("selection", body.selection, Ctx.SELECTION),
        ("combo", body.combo, Ctx.COMBO),
    ):
        if text is None:
            continue
        r = check(text, ctx, registry)
        out[label] = {
            "ok": r.ok,
            "problems": r.problems,
            "operators": r.operators,
            "text": r.text,
            "operatorCount": r.operator_count,
            "depth": r.depth,
            "type": r.ty.value if r.ty else None,
        }
    return out


@router.post("/plan")
async def plan(body: ResearchTask, state: State) -> dict[str, Any]:
    """What a task with these settings could do, without running anything."""
    body.checked()
    run = _run_of(body)
    registry = await engine.registry_for(state.optimizer, run)
    config = engine.config_of(run)
    cover = coverage.build(
        registry,
        [],
        disabled=config.controls.disabled_operators,
        families_off=config.controls.disabled_families,
    )
    from ..labs.super_research.planner import Planner

    planner = Planner(registry, config, [], generation=0)
    warnings = [
        "Component Alphas' real operator counts are unknown until BRAIN selects them: the "
        f"selection caps each at {config.operator_cap} (8,000 / {run.selection_limit}), and BRAIN "
        "may truncate a selection past 8,000 operators in all.",
        "BRAIN offers no selection preview here, so a selection's content is first seen in its "
        "simulation.",
    ]
    if not state_has_operators(cover):
        warnings.append("Few operators are verified until BRAIN's operator list is read.")
    return {
        "simulations": body.simulations,
        "concurrent": min(body.cores, SUPER_ALPHA_SLOTS),
        "operatorSummary": coverage.summary(cover),
        "comboTemplates": [t.key for t in planner.available_templates(Ctx.COMBO)],
        "selectionTemplates": [t.key for t in planner.available_templates(Ctx.SELECTION)],
        "unavailableTemplates": [
            {
                "key": t.key,
                "needs": [
                    n
                    for n in t.needs
                    if not planner.registry.eligibility(
                        n, Ctx.COMBO if t.ctx == "combo" else Ctx.SELECTION
                    ).eligible
                ],
            }
            for t in (*templates.COMBO_TEMPLATES, *templates.SELECTION_TEMPLATES)
            if not planner.usable(t, Ctx.COMBO if t.ctx == "combo" else Ctx.SELECTION)
        ],
        "operatorCap": config.operator_cap,
        "objectives": sorted(scoring.OBJECTIVES),
        "turnoverRange": [scoring.TURNOVER_LOW, scoring.TURNOVER_HIGH],
        "warnings": warnings,
    }


def state_has_operators(rows: list[coverage.OperatorCoverage]) -> bool:
    return sum(r.signature_verified for r in rows) > 12


@router.post("/tasks", status_code=201)
async def add_task(body: ResearchTask, state: State) -> AddedTask:
    """Create a research task, and queue it when ``run``. No candidates are written yet: the
    first generation is written when the scheduler gives the task its cores."""
    body.checked()
    run = _run_of(body)
    registry = await engine.registry_for(state.optimizer, run)
    config = engine.config_of(run)
    cover = coverage.build(
        registry,
        [],
        disabled=config.controls.disabled_operators,
        families_off=config.controls.disabled_families,
    )
    if not any(r.eligible for r in cover):
        raise refuse(
            422, "no_operators", "No verified operator can be generated with these settings."
        )
    return await add_study(
        state,
        now=utcnow(),
        sampler=SUPER_RESEARCH,
        params=run,
        simulations=body.simulations,
        batch_size=body.cores * MULTI_SIMULATION_SIZE,
        template_source="1",
        template_name=f"Super Alpha Research · {body.region} D{body.delay} {body.universe}",
        run=body.run,
        seeds=None,
    )


async def _task(state: Any, task_id: int) -> Study:
    async with state.db.session() as session:
        row = await session.get(Study, task_id)
    if row is None or row.sampler != SUPER_RESEARCH:
        raise refuse(404, "not_found", "No such Super Alpha Research task.")
    return row


@router.get("/tasks/{task_id}")
async def detail(task_id: int, state: State) -> dict[str, Any]:
    """Everything the research screen shows for one task, computed from its trials."""
    row = await _task(state, task_id)
    run = params_of(row, SuperResearchParams)
    rows = await engine.rows_of(state.optimizer, task_id)
    candidates = report.evaluate(rows, run)
    registry = await engine.registry_for(state.optimizer, run)
    config = engine.config_of(run)
    cover = coverage.build(
        registry,
        report.used_of(candidates),
        disabled=config.controls.disabled_operators,
        families_off=config.controls.disabled_families,
    )
    return {
        "id": row.id,
        "name": row.name,
        "status": str(row.status),
        "message": row.message,
        "params": run.dump(),
        "dashboard": report.dashboard(candidates, row.max_trials),
        "candidates": candidates,
        "coverage": coverage.rows_as_dicts(cover),
        "coverageSummary": coverage.summary(cover),
        "avoided": coverage.avoided(cover),
    }


@router.post("/tasks/{task_id}/proposals")
async def propose(task_id: int, body: Proposal, state: State) -> dict[str, Any]:
    """Queue an edited or hand-written candidate on a task. It must pass the local checker, and
    it is refused when the task already ran that exact SuperAlpha."""
    row = await _task(state, task_id)
    run = params_of(row, SuperResearchParams)
    registry = await engine.registry_for(state.optimizer, run)
    config = engine.config_of(run)
    problems: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for label, text, ctx in (
        ("selection", body.selection, Ctx.SELECTION),
        ("combo", body.combo, Ctx.COMBO),
    ):
        r = check(text, ctx, registry, config.controls.limits())
        if not r.ok:
            problems[label] = r.problems
        texts[label] = r.text
    if problems:
        raise refuse(
            422, "invalid_expression", "; ".join(f"{k}: {p[0]}" for k, p in problems.items())
        )
    from ..labs.super_research.planner import proposal_key

    decay = body.decay if body.decay is not None else config.decays[0]
    trunc = body.truncation if body.truncation is not None else config.truncations[0]
    act = body.activation or config.activations[0]
    key = proposal_key(texts["selection"], texts["combo"], decay, trunc, act)
    rows = await engine.rows_of(state.optimizer, task_id)
    if any(r.params.get("key") == key for r in rows):
        raise refuse(409, "duplicate", "This task already ran that SuperAlpha at those settings.")
    queued = [p for p in run.proposals if p.get("key") == key]
    if queued:
        raise refuse(409, "duplicate", "That SuperAlpha is already queued on this task.")
    run.proposals.append({**body.model_dump(exclude_none=True), "key": key, "source": "edited"})
    async with state.db.session() as session:
        stored = await session.get(Study, task_id)
        if stored is None:
            raise refuse(404, "not_found", "No such Super Alpha Research task.")
        stored.sampler_params = run.dump()
        await session.commit()
    return {"queued": True, "key": key}
