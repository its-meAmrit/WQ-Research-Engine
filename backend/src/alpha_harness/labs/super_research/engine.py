"""The refill loop for Advanced Mode: written a generation at a time, from the results.

``scheduler.advance`` calls :func:`refill` whenever a Super Alpha Research task has room.
Candidates become parked trials (``Waiting for cores.``) exactly as the Classic lab's are, so
sending, pausing, resuming after a restart, and the three-at-a-time limit are the scheduler's
and not repeated here. Nothing is sent to BRAIN that did not pass the local checker.
"""

from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select

from ...brain.platform import SUPER_ALPHA_MIN_SELECTION, SUPER_ALPHA_OPERATOR_LIMIT
from ...brain.schemas import FULL_MODE, SimulationSettings
from ...db.models import Study, StudyStatus, Trial, TrialState
from .. import scheduler
from ..fastexpr import parse
from ..params import SuperResearchParams, params_of
from . import coverage, report
from .canon import canonical
from .checker import check
from .grammar import Controls
from .planner import Config, Planner, Proposal, proposal_key
from .registry import BASELINE, Ctx, Registry

if TYPE_CHECKING:
    from ..study import Optimizer

log = structlog.get_logger(__name__)

PENDING_SEND = scheduler.PENDING_SEND


def config_of(run: SuperResearchParams) -> Config:
    cap = SUPER_ALPHA_OPERATOR_LIMIT // max(run.selection_limit, SUPER_ALPHA_MIN_SELECTION)
    return Config(
        controls=Controls(
            max_depth=run.max_depth,
            max_operators=run.max_operators,
            max_variables=run.max_variables,
            seed=run.seed,
            exploration=run.exploration,
            disabled_operators=frozenset(run.disabled_operators),
            disabled_families=frozenset(run.disabled_families),
            preferred_families=frozenset(run.preferred_families),
        ),
        strategy=run.strategy,
        mutation_probability=run.mutation_probability,
        decays=tuple(run.decays) or (4,),
        truncations=tuple(run.truncations) or (0.05,),
        activations=tuple(run.activation) or ("IS",),
        selection_limit=run.selection_limit,
        operator_cap=cap,
        patience=run.patience,
    )


async def registry_for(optimizer: Optimizer, run: SuperResearchParams) -> Registry:
    """The registry, verified against the operators BRAIN last published to this account.
    Without them, only what the Classic recipes already prove is verified."""
    operators = await optimizer.metadata.cached_operators() or []
    if not operators:
        return BASELINE
    registry, _ = BASELINE.reconciled(operators, strict_scope=not run.relaxed_scope)
    return registry


def settings_for(run: SuperResearchParams, p: Proposal) -> dict[str, Any]:
    return SimulationSettings(
        region=run.region,
        delay=run.delay,
        universe=run.universe,
        neutralization=run.neutralization,
        decay=p.decay,
        truncation=p.truncation,
        selection_handling="POSITIVE",
        selection_limit=run.selection_limit,
        component_activation=p.activation,
        simulation_mode=FULL_MODE,
    ).model_dump(by_alias=True, exclude_none=True)


def trial_of(
    study_id: int,
    number: int,
    generation: int,
    run: SuperResearchParams,
    p: Proposal,
    *,
    baseline: bool,
) -> Trial:
    return Trial(
        study_id=study_id,
        number=number,
        generation=generation,
        params={
            "recipe": p.name,
            "selection": p.selection,
            "combo": p.combo,
            "key": p.key,
            "family": p.family,
            "origin": p.origin,
            "explanation": p.explanation,
            "operators": p.operators,
            "parents": p.parents,
            "local": p.local,
            "baseline": baseline,
            "advanced": True,
            "seed": run.seed,
        },
        distributions={},
        expression=f"{p.family} · {p.origin.split(' {')[0]} · Decay {p.decay} · {p.activation}",
        settings=settings_for(run, p),
        state=TrialState.PRUNED,
        message=PENDING_SEND,
    )


def baselines_for(
    proposals: list[Proposal], known_selections: set[str], seen: set[str] | None = None
) -> list[Proposal]:
    """An equal-weight run on each selection that has none yet: what its combo must beat.

    A proposal that is itself equal-weight already counts for its selection.
    """
    out: list[Proposal] = []
    covered = set(known_selections)
    taken = seen if seen is not None else set()
    for p in proposals:
        if p.combo == "1":
            covered.add(canonical(parse(p.selection)))
    for p in proposals:
        sel = canonical(parse(p.selection))
        if sel in covered:
            continue
        covered.add(sel)
        key = proposal_key(p.selection, "1", p.decay, p.truncation, p.activation)
        if key in taken:
            continue
        taken.add(key)
        out.append(
            Proposal(
                selection=p.selection,
                combo="1",
                decay=p.decay,
                truncation=p.truncation,
                activation=p.activation,
                family="equal_weight",
                origin="baseline",
                explanation=(
                    "Every selected Alpha weighted equally: what a smarter combo must beat "
                    "on this selection."
                ),
                operators=[],
                key=key,
                local={"depth": 1, "operators": 0},
            )
        )
    return out


def assemble(
    user: list[Proposal],
    planned: list[Proposal],
    known: set[str],
    seen: set[str],
    want: int,
) -> list[Proposal]:
    """The generation to write: the consultant's own first, then each new selection's
    equal-weight baseline, then the planned candidates, all within ``want``."""
    planned = list(planned)
    while True:
        bases = baselines_for([*user, *planned], known, set(seen))
        if len(user) + len(bases) + len(planned) <= want or not planned:
            break
        planned.pop()
    final_bases = baselines_for([*user, *planned], known, seen)
    return ([*user, *final_bases, *planned])[:want]


def proposals_from(
    run: SuperResearchParams, registry: Registry, config: Config, seen: set[str]
) -> list[Proposal]:
    """Expressions the consultant or an LLM proposed: only those that pass the same checker."""
    out: list[Proposal] = []
    for item in run.proposals:
        sel, combo = str(item.get("selection", "")), str(item.get("combo", ""))
        a = check(sel, Ctx.SELECTION, registry, config.controls.limits())
        b = check(combo, Ctx.COMBO, registry, config.controls.limits())
        if not (a.ok and b.ok):
            continue
        decay = int(item.get("decay") or config.decays[0])
        trunc = float(item.get("truncation") or config.truncations[0])
        act = str(item.get("activation") or config.activations[0])
        key = proposal_key(a.text, b.text, decay, trunc, act)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Proposal(
                selection=a.text,
                combo=b.text,
                decay=decay,
                truncation=trunc,
                activation=act,
                family="proposed",
                origin=str(item.get("source") or "proposed"),
                explanation=str(item.get("note") or "Proposed by the consultant."),
                operators=sorted(set(a.operators) | set(b.operators)),
                key=key,
                local={"depth": b.depth, "operators": b.operator_count},
            )
        )
    return out


async def rows_of(optimizer: Optimizer, study_id: int) -> list[report.Row]:
    async with optimizer.db.session() as session:
        found = (
            await session.execute(
                select(
                    Trial.id,
                    Trial.number,
                    Trial.state,
                    Trial.params,
                    Trial.settings,
                    Trial.result,
                    Trial.message,
                    Trial.generation,
                    Trial.alpha_id,
                )
                .where(Trial.study_id == study_id)
                .order_by(Trial.number)
            )
        ).all()
    return [
        report.Row(
            id=r.id,
            number=r.number,
            state=r.state,
            params=r.params or {},
            settings=r.settings or {},
            result=r.result,
            message=r.message,
            generation=r.generation,
            alpha_id=r.alpha_id,
        )
        for r in found
    ]


async def refill(optimizer: Optimizer, row: Study, want: int, waiting: bool) -> int:
    """Send what is parked; once nothing is, write the next generation from the results."""
    run = params_of(row, SuperResearchParams)
    parked = (
        Trial.study_id == row.id,
        Trial.state == TrialState.PRUNED,
        Trial.message == PENDING_SEND,
    )
    async with optimizer.db.session() as session:
        batch = (
            (
                await session.scalars(
                    select(Trial).where(*parked).order_by(Trial.number).limit(want)
                )
            ).all()
            if want > 0
            else []
        )
        sent = await scheduler.send_parked(optimizer, row, batch) if batch else 0
        left = await session.scalar(select(func.count()).select_from(Trial).where(*parked))
    if sent or left:
        return sent
    if want <= 0:
        return 0

    rows = await rows_of(optimizer, row.id)
    candidates = report.evaluate(rows, run)
    if report.stalled(candidates, run.patience):
        if not waiting:
            await scheduler.finish(
                optimizer,
                row.id,
                StudyStatus.COMPLETE,
                f"Stopped early: {run.patience} generations without a new best.",
            )
        return 0

    registry = await registry_for(optimizer, run)
    config = config_of(run)
    generation = max((r.generation or 0 for r in rows), default=-1) + 1
    cover = coverage.build(
        registry,
        report.used_of(candidates),
        disabled=config.controls.disabled_operators,
        families_off=config.controls.disabled_families,
        probe_operators=False,
    )
    avoid = frozenset(coverage.avoided(cover))
    planner = Planner(
        registry, config, report.history_of(candidates), generation=generation, avoid=avoid
    )
    for item in cover:
        if item.eligible or item.signature_verified:
            planner.used[item.name] = item.included
            planner.grammar.uses[item.name] = item.included
    # The operators least explored so far get a slot of their own each generation.
    untouched = sorted(c.name for c in cover if c.eligible and c.included == 0)
    focus = untouched[: max(1, want // 10)]
    known = {canonical(parse(str(c["selection"]))) for c in candidates if c["baseline"]}
    user = proposals_from(run, registry, config, planner.seen)[:want]
    planned = planner.plan(max(want - len(user), 0), focus=focus)
    chosen = assemble(user, planned, known, planner.seen, want)
    if not chosen:
        if not waiting:
            await scheduler.finish(
                optimizer,
                row.id,
                StudyStatus.COMPLETE,
                "No new valid candidates could be built from the verified operators.",
            )
        return 0
    async with optimizer.db.session() as session:
        start = await session.scalar(select(func.max(Trial.number)).where(Trial.study_id == row.id))
        number = -1 if start is None else int(start)
        new = []
        for p in chosen:
            number += 1
            new.append(
                trial_of(
                    row.id,
                    number,
                    generation,
                    run,
                    p,
                    baseline=p.origin == "baseline" or p.combo == "1",
                )
            )
        session.add_all(new)
        await session.flush()
        sent = await scheduler.send_parked(optimizer, row, new)
        await session.commit()
    log.info("super_research_generation", study=row.id, generation=generation, written=len(chosen))
    return sent
