"""Advanced Mode of the Super Alpha Lab. No test here talks to BRAIN: the operator list a
"published" registry is reconciled against is built from the registry's own drafts, so those
tests exercise the reconciliation mechanics and say nothing about BRAIN's real signatures."""

import random
from collections import Counter
from typing import Any

import pytest
from sqlalchemy import select

from alpha_harness.brain.platform import SUPER_ALPHA_SLOTS
from alpha_harness.brain.schemas import SimulationType
from alpha_harness.db.models import Study, StudyStatus, Trial, TrialState
from alpha_harness.labs import super_alpha
from alpha_harness.labs.fastexpr import parse, render
from alpha_harness.labs.params import SUPER_RESEARCH, SuperResearchParams
from alpha_harness.labs.scheduler import PENDING_SEND, request_of
from alpha_harness.labs.super_research import coverage, engine, mutate, report, scoring
from alpha_harness.labs.super_research import registry as reg
from alpha_harness.labs.super_research import templates as tpl
from alpha_harness.labs.super_research.canon import canonical
from alpha_harness.labs.super_research.checker import check
from alpha_harness.labs.super_research.grammar import Controls, Grammar, accepted
from alpha_harness.labs.super_research.planner import Config, Done, Planner
from alpha_harness.labs.super_research.registry import BASELINE, Ctx, Status

COMBO, SEL = Ctx.COMBO, Ctx.SELECTION


def published(registry: reg.Registry = BASELINE, *, scope: bool = True) -> list[dict[str, Any]]:
    """An operator list in BRAIN's shape, derived from the registry's drafts."""
    out = []
    for name, spec in registry.specs.items():
        if spec.params is None:
            continue
        parts = [
            p.name if p.required else f"{p.name} = {p.default}"
            for p in spec.params
            if not p.variadic
        ]
        if any(p.variadic for p in spec.params):
            parts.append("..")
        item: dict[str, Any] = {
            "name": name,
            "category": spec.family,
            "definition": f"{name}({', '.join(parts)})",
            "scope": [c.value.upper() for c in spec.contexts] if scope else ["REGULAR"],
        }
        out.append(item)
    return out


@pytest.fixture(scope="module")
def full() -> reg.Registry:
    """Every signed operator promoted, as if BRAIN's list agreed with each draft."""
    registry, _ = BASELINE.reconciled(published())
    return registry


# --- the registry ------------------------------------------------------------------------


def test_exactly_161_unique_documented_operators() -> None:
    names = reg.BOTH_OPERATORS + reg.SELECTION_ONLY_OPERATORS + reg.COMBO_ONLY_OPERATORS
    assert (
        len(reg.BOTH_OPERATORS),
        len(reg.SELECTION_ONLY_OPERATORS),
        len(reg.COMBO_ONLY_OPERATORS),
    ) == (42, 1, 118)
    assert len(names) == len(set(names)) == 161
    assert set(BASELINE.specs) == set(names)
    assert len(BASELINE) == 161


def test_context_permissions_match_the_inventory() -> None:
    for name in reg.BOTH_OPERATORS:
        assert BASELINE.specs[name].contexts == reg.BOTH
    assert BASELINE.specs["universe_size"].contexts == reg.SEL
    for name in reg.COMBO_ONLY_OPERATORS:
        assert BASELINE.specs[name].contexts == reg.CMB


def test_registered_is_not_generatable() -> None:
    counts = BASELINE.counts()
    assert counts["registered"] == 161
    assert counts["eligible"] == counts[Status.VERIFIED] < 161
    assert BASELINE.eligibility("zscore", COMBO).eligible is False
    assert "confirmed" in BASELINE.eligibility("zscore", COMBO).reason.lower()
    assert BASELINE.eligibility("group_rank", COMBO).reason  # unsigned: says why


def test_every_verified_operator_has_a_tested_signature(full: reg.Registry) -> None:
    for registry in (BASELINE, full):
        for name, spec in registry.specs.items():
            if spec.status != Status.VERIFIED:
                continue
            assert spec.params, name
            assert spec.signature and spec.signature.startswith(f"{name}(")
            ctxs = [c for c in spec.contexts if registry.eligibility(name, c).eligible]
            assert ctxs, f"{name} is verified but nothing can build it"
            assert any(coverage.probe(registry, name, c) for c in ctxs), name


def test_reconcile_promotes_demotes_and_never_trusts_a_name(full: reg.Registry) -> None:
    assert full.specs["ts_mean"].status == Status.VERIFIED
    assert full.specs["ts_mean"].verified_by == "brain-operators"
    assert full.specs["ts_ir"].verified_by == "classic-recipes"
    # Arity disagreement: BRAIN's ts_mean takes one more input than the draft.
    wrong = [
        {**o, "definition": "ts_mean(x, d, extra)"} if o["name"] == "ts_mean" else o
        for o in published()
    ]
    registry, report_ = BASELINE.reconciled(wrong)
    assert registry.specs["ts_mean"].status == Status.PENDING
    assert "ts_mean" in report_.mismatched and registry.specs["ts_mean"].note
    # Scope disagreement: BRAIN says a combo operator is selection-only.
    scoped = [{**o, "scope": ["SELECTION"]} if o["name"] == "ts_mean" else o for o in published()]
    registry, report_ = BASELINE.reconciled(scoped)
    assert registry.specs["ts_mean"].status == Status.INCOMPATIBLE
    # Not published at all: stays unverified.
    registry, report_ = BASELINE.reconciled([o for o in published() if o["name"] != "ts_mean"])
    assert registry.specs["ts_mean"].status != Status.VERIFIED and "ts_mean" in report_.missing
    # Operators BRAIN lists for REGULAR only cannot have their super scope confirmed.
    registry, report_ = BASELINE.reconciled(published(scope=False))
    assert registry.specs["ts_mean"].status != Status.VERIFIED and report_.unconfirmed_scope
    relaxed, _ = BASELINE.reconciled(published(scope=False), strict_scope=False)
    assert relaxed.specs["ts_mean"].verified_by == "brain-operators (arity only)"


def test_unsigned_operators_stay_registered_and_ungenerated(full: reg.Registry) -> None:
    unsigned = [n for n, s in full.specs.items() if s.params is None]
    assert unsigned and all(not full.eligibility(n, COMBO).eligible for n in unsigned)
    assert all(full.specs[n].note for n in unsigned)


# --- the checker -------------------------------------------------------------------------

STATS = "stats = generate_stats(alpha); "


@pytest.mark.parametrize(
    ("text", "ctx", "fragment"),
    [
        ("universe_size(universe) + 1", COMBO, "Selection-only"),
        ("ts_ir(operator_count, 5)", SEL, "Combo-only"),
        (STATS + "ts_ir(stats.returns)", COMBO, "missing"),
        (STATS + "ts_ir(stats.returns, 25, 7)", COMBO, "at most"),
        (STATS + "ts_ir(stats.returns, 25, foo=1)", COMBO, "no parameter named foo"),
        (STATS + "ts_ir(stats.returns, 0)", COMBO, "between 1 and 512"),
        (STATS + "ts_ir(stats.returns, 2.5)", COMBO, "whole number"),
        (STATS + "reduce_max(stats.returns)", COMBO, "matrix"),
        (STATS + "ts_ir(self_corr(stats.returns, 50), 5)", COMBO, "one value per Alpha"),
        (STATS + "self_corr(stats.returns, 50) + stats.returns", COMBO, "not compatible"),
        (STATS + "stats.nonsense", COMBO, "not a verified statistic"),
        (STATS + "ts_ir(stats, 5)", COMBO, "statistics"),
        ("group_rank(a, b)", COMBO, "no verified signature"),
        ("zscore(turnover)", SEL, "Combo-only"),
        ("abs(turnover)", SEL, "pending"),
        ("turnover / 0", SEL, "zero"),
        ("mystery_op(turnover)", SEL, "not a documented"),
        ("long_count > 3", COMBO, "not available in a combo"),
        ("universe_size(universe)", SEL, "must end in one score per Alpha"),
        (STATS + "stats", COMBO, "must end in"),
        ("a = 1;", COMBO, "last statement"),
        ("(", COMBO, "Syntax"),
    ],
)
def test_invalid_expressions_are_refused_before_simulation(
    text: str, ctx: Ctx, fragment: str
) -> None:
    result = check(text, ctx, BASELINE)
    assert not result.ok
    assert any(fragment.lower() in p.lower() for p in result.problems), result.problems


def test_limits_are_enforced() -> None:
    deep = STATS + "ts_ir(" * 12 + "stats.returns" + ", 5)" * 12
    assert any("deep" in p for p in check(deep, COMBO, BASELINE).problems)
    assert any(
        "operators" in p
        for p in check(deep, COMBO, BASELINE, reg_limits(max_depth=99, max_operators=3)).problems
    )


def reg_limits(**kw: int) -> Any:
    from alpha_harness.labs.super_research.checker import Limits

    return Limits(**kw)


def test_classic_recipes_pass_the_checker() -> None:
    for recipe in super_alpha.COMBOS:
        assert check(recipe.code, COMBO, BASELINE).ok, recipe.name
    for recipe in super_alpha.selections(40):
        assert check(recipe.code, SEL, BASELINE).ok, recipe.name


def test_complex_expression_serializes_and_round_trips() -> None:
    text = (
        STATS + "m = self_corr(stats.returns, 250); ic = if_else(m == 1.0, nan, m); "
        "max(ts_ir(stats.returns, 60), 0) * max(1 - reduce_max(ic), 0.05)"
    )
    result = check(text, COMBO, BASELINE)
    assert result.ok and result.ty == reg.Ty.VEC
    assert result.operators == [
        "generate_stats",
        "ts_ir",
        "max",
        "self_corr",
        "if_else",
        "reduce_max",
    ] or set(result.operators) == {
        "generate_stats",
        "ts_ir",
        "max",
        "self_corr",
        "if_else",
        "reduce_max",
    }
    assert parse(result.text) == parse(text)


# --- generation --------------------------------------------------------------------------


def test_templates_that_build_pass_and_the_rest_say_what_they_need(full: reg.Registry) -> None:
    rng = random.Random(0)
    for registry in (BASELINE, full):
        for t in (*tpl.COMBO_TEMPLATES, *tpl.SELECTION_TEMPLATES):
            ctx = COMBO if t.ctx == "combo" else SEL
            node = t.build(t.sample(rng))
            ok = check(node, ctx, registry).ok
            assert ok == all(registry.eligibility(n, ctx).eligible for n in t.needs), t.key
    # With every signed operator verified, every template is usable.
    assert all(check(t.build(t.sample(rng)), COMBO, full).ok for t in tpl.COMBO_TEMPLATES)


def test_grammar_output_is_valid_nontrivial_and_in_context(full: reg.Registry) -> None:
    controls = Controls(seed=11)
    grammar = Grammar(full, controls, random.Random(11))
    combos, sels = set(), set()
    for _ in range(600):
        if (r := accepted(grammar.combo(), COMBO, full, controls)) is not None:
            assert not set(r.operators) & set(reg.SELECTION_ONLY_OPERATORS)
            assert r.operator_count <= controls.max_operators + 2
            combos.add(r.text)
        if (r := accepted(grammar.selection(266), SEL, full, controls)) is not None:
            assert all(full.specs[o].contexts & {SEL} for o in r.operators)
            assert "operator_count <= 266" in r.text  # the 8,000-operator gate
            sels.add(r.text)
    assert len(combos) > 100 and len(sels) > 100


def test_every_eligible_operator_can_appear_in_a_generated_expression(full: reg.Registry) -> None:
    """No hardcoded shortlist: each verified, constructible operator is reachable."""
    rows = coverage.build(full, [])
    eligible = [r for r in rows if r.eligible]
    assert len(eligible) > 100
    for row in eligible:
        assert row.locally_validated, row.name
    # and the rest are explained
    assert all(r.reason for r in rows if not r.eligible)


def test_generation_is_deterministic_for_a_seed(full: reg.Registry) -> None:
    cfg = Config(controls=Controls(seed=5))
    a = [p.key for p in Planner(full, cfg, [], generation=0).plan(15)]
    b = [p.key for p in Planner(full, cfg, [], generation=0).plan(15)]
    c = [p.key for p in Planner(full, Config(controls=Controls(seed=6)), [], generation=0).plan(15)]
    assert a == b and a != c


def test_deduplication_collapses_equivalent_expressions() -> None:
    a = parse(STATS + "max(ts_ir(stats.returns, 60), 0) + 0")
    b = parse("stats = generate_stats(alpha); max(0, ts_ir(stats.returns, 60))")
    assert canonical(a) == canonical(b)
    assert canonical(parse("max(a, b)")) == canonical(parse("max(b, a)"))
    assert canonical(parse("1 * turnover")) == canonical(parse("turnover"))
    cfg = Config(controls=Controls(seed=2))
    first = Planner(BASELINE, cfg, [], generation=0).plan(20)
    hist = [
        Done(p.key, p.selection, p.combo, p.family, 1.0, (p.decay, p.truncation, p.activation))
        for p in first
    ]
    again = Planner(BASELINE, cfg, hist, generation=1).plan(20)
    assert not {p.key for p in again} & {p.key for p in first}
    assert len({p.key for p in again}) == len(again)


def test_mutations_preserve_type_correctness(full: reg.Registry) -> None:
    rng = random.Random(3)
    controls = Controls(seed=3)
    grammar = Grammar(full, controls, rng)
    parents = [t.build(t.sample(rng)) for t in tpl.COMBO_TEMPLATES]
    kinds: Counter[str] = Counter()
    for i in range(400):
        out = mutate.mutate(
            rng, parents[i % len(parents)], COMBO, grammar, partner=parents[(i * 7) % len(parents)]
        )
        if out is None:
            continue
        child, kind = out
        result = check(child, COMBO, full, controls.limits())
        # A child can exceed a growth limit and be refused, but never with a type error.
        type_errors = [
            p
            for p in result.problems
            if "deep" not in p and "operators;" not in p and "variables" not in p
        ]
        assert not type_errors, (kind, render(child), type_errors)
        kinds[kind] += 1
    assert set(kinds) == {"parameter", "operator", "subtree", "crossover"}


def test_operator_substitution_keeps_signatures(full: reg.Registry) -> None:
    rng = random.Random(1)
    grammar = Grammar(full, Controls(seed=1), rng)
    base = parse(STATS + "max(ts_mean(stats.returns, 60), 0)")
    seen = set()
    for _ in range(60):
        out = mutate.substitute_operator(
            rng, mutate.inline(base) if hasattr(mutate, "inline") else base, COMBO, grammar
        )
        if out is not None:
            seen.add(render(out))
            assert check(program_of(out), COMBO, full).ok
    assert len(seen) > 3


def program_of(tree: Any) -> Any:
    from alpha_harness.labs.super_research.build import program

    return program(tree)


def test_growth_is_bounded(full: reg.Registry) -> None:
    cfg = Config(controls=Controls(seed=9, max_depth=3, max_operators=8))
    for p in Planner(full, cfg, [], generation=0).plan(40):
        assert p.local["operators"] <= 8 + 2 and p.local["depth"] <= 3 + 3 + 1


def test_selection_generation_keeps_the_operator_gate(full: reg.Registry) -> None:
    cfg = Config(controls=Controls(seed=4), operator_cap=200)
    for p in Planner(full, cfg, [], generation=0).plan(30):
        assert "operator_count <= 200" in p.selection


def test_baseline_equal_weight_is_retained() -> None:
    props = Planner(BASELINE, Config(controls=Controls(seed=1)), [], generation=0).plan(10)
    bases = engine.baselines_for(props, set())
    assert bases and all(b.combo == "1" for b in bases)
    assert len({canonical(parse(b.selection)) for b in bases}) == len(bases)
    # no second baseline for a selection that has one
    assert not engine.baselines_for(props, {canonical(parse(p.selection)) for p in props})


# --- scoring and coverage ----------------------------------------------------------------


def test_turnover_range_and_score_terms() -> None:
    ok = {"stats": {"sharpe": 2.0, "fitness": 1.0, "turnover": 0.2, "drawdown": 0.05}}
    low = {"stats": {"sharpe": 2.0, "fitness": 1.0, "turnover": 0.01, "drawdown": 0.05}}
    assert scoring.turnover_in_range(ok) and not scoring.turnover_in_range(low)
    assert scoring.score(ok)[0] > scoring.score(low)[0]
    value, parts = scoring.score(ok, baseline_sharpe=1.0)
    assert value == pytest.approx(sum(parts.values())) and parts["vs_equal_weight"] > 0
    assert scoring.score({})[0] == scoring.FAILURE_SCORE


def test_coverage_counts_come_from_outcomes() -> None:
    used = [
        coverage.Used(["ts_ir", "max"], "accepted", 2.0),
        coverage.Used(["ts_ir"], "rejected", None, "Unknown operator ts_ir"),
        coverage.Used(["ts_ir"], "rejected", None, "Syntax error"),
        coverage.Used(["ts_ir"], "rejected", None, "timeout"),
        coverage.Used(["max"], "waiting"),
    ]
    rows = {r.name: r for r in coverage.build(BASELINE, used, probe_operators=False)}
    ts = rows["ts_ir"]
    assert (ts.included, ts.submitted, ts.accepted, ts.rejected, ts.successful) == (4, 4, 1, 3, 1)
    assert ts.failure_reasons == {"operator": 1, "syntax": 1, "timeout": 1}
    assert rows["max"].included == 2 and rows["max"].submitted == 1
    assert "ts_ir" not in coverage.avoided(list(rows.values()))  # it was accepted once
    assert coverage.avoided(coverage.build(BASELINE, used[1:4], probe_operators=False))
    summary = coverage.summary(list(rows.values()))
    assert summary["registered"] == 161 and summary["eligible"] + summary["blocked"] == 161


def test_rarely_used_operators_get_slots() -> None:
    cfg = Config(controls=Controls(seed=1, exploration=1.0))
    planner = Planner(BASELINE, cfg, [], generation=0)
    made = planner.plan(6, focus=["self_corr", "reduce_max"])
    assert any("self_corr" in p.operators for p in made)
    assert any(p.origin == "coverage-guided" for p in made)


# --- classic mode and requests -----------------------------------------------------------


def test_classic_mode_is_unchanged() -> None:
    found = super_alpha.candidates(
        region="USA",
        delay=1,
        universe="TOP3000",
        neutralization="SUBINDUSTRY",
        selection_limit=40,
        activations=["IS", "OS"],
    )
    assert len(found) == 80
    assert super_alpha.DECAYS == (4, 10) and super_alpha.TRUNCATION == 0.05
    assert all(c.request.settings.selection_handling == "POSITIVE" for c in found)


def test_super_requests_serialize_correctly() -> None:
    p = Planner(BASELINE, Config(controls=Controls(seed=1)), [], generation=0).plan(1)[0]
    run = SuperResearchParams(
        region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY", selection_limit=40
    )
    trial = engine.trial_of(1, 0, 0, run, p, baseline=False)
    wire = request_of(trial).to_wire()
    assert wire["type"] == SimulationType.SUPER == "SUPER"
    assert wire["settings"]["simulationMode"] == "FULL"
    assert wire["settings"]["selectionHandling"] == "POSITIVE"
    assert wire["settings"]["selectionLimit"] == 40
    assert wire["settings"]["region"] == "USA" and wire["settings"]["universe"] == "TOP3000"
    assert wire["selection"] == p.selection and wire["combo"] == p.combo


def test_no_task_defaults_to_glb_or_other_region() -> None:
    run = SuperResearchParams(region="EUR", delay=1, universe="TOP1200", neutralization="MARKET")
    p = Planner(BASELINE, Config(controls=Controls(seed=1)), [], generation=0).plan(1)[0]
    assert engine.settings_for(run, p)["region"] == "EUR"


# --- persistence, resume, concurrency (database-backed, no BRAIN) ------------------------


class FakeEngine:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def enqueue(self, requests: list[Any], task: str) -> dict[str, Any]:
        self.sent.extend(requests)
        return {
            "outcomes": [{"recordId": None, "alphaId": None, "status": "QUEUED"} for _ in requests]
        }

    async def drop_queued(self, task: str) -> None: ...
    async def set_quota(self, task: str, n: int) -> None: ...


class FakeMetadata:
    def __init__(self, operators: list[dict[str, Any]] | None) -> None:
        self.operators = operators

    async def cached_operators(self) -> list[dict[str, Any]] | None:
        return self.operators


class FakeOptimizer:
    def __init__(self, db: Any, operators: list[dict[str, Any]] | None = None) -> None:
        self.db = db
        self.engine = FakeEngine()
        self.metadata = FakeMetadata(operators)
        self.open_trials: dict[int, Any] = {}
        self.studies: dict[int, Any] = {}

    async def notify(self) -> None: ...


async def make_task(db: Any, **over: Any) -> Study:
    params = SuperResearchParams(
        region="USA",
        delay=1,
        universe="TOP3000",
        neutralization="SUBINDUSTRY",
        selection_limit=40,
        cores=SUPER_ALPHA_SLOTS,
        seed=21,
        **over,
    )
    row = Study(
        name="t",
        template_name="t",
        template_source="1",
        sampler=SUPER_RESEARCH,
        sampler_params=params.dump(),
        objectives=["sharpe"],
        directions=["maximize"],
        batch_size=30,
        max_trials=60,
        task="super-research-test",
        status=StudyStatus.RUNNING,
    )
    async with db.session() as session:
        session.add(row)
    return row


async def trials(db: Any, study_id: int) -> list[Trial]:
    async with db.session() as session:
        return list(
            (
                await session.scalars(
                    select(Trial).where(Trial.study_id == study_id).order_by(Trial.number)
                )
            ).all()
        )


async def test_refill_persists_valid_deduplicated_candidates(db: Any) -> None:
    row = await make_task(db)
    opt = FakeOptimizer(db)
    sent = await engine.refill(opt, row, 30, False)
    written = await trials(db, row.id)
    assert sent == len(written) > 0 and sent <= 30
    keys = [t.params["key"] for t in written]
    assert len(keys) == len(set(keys))
    assert len(opt.engine.sent) == sent
    for t in written:
        assert (
            check(t.params["selection"], SEL, BASELINE).ok
            and check(t.params["combo"], COMBO, BASELINE).ok
        )
        assert t.generation == 0 and t.settings["region"] == "USA"
        assert t.settings["simulationMode"] == "FULL"
    assert any(t.params["baseline"] for t in written)  # an equal-weight comparison exists
    assert Counter(t.params["origin"] for t in written)


async def test_refill_resumes_parked_work_after_interruption(db: Any) -> None:
    row = await make_task(db)
    opt = FakeOptimizer(db)
    await engine.refill(opt, row, 30, False)
    written = await trials(db, row.id)
    # Simulate a pause or crash between writing and sending: every trial back to parked.
    async with db.session() as session:
        for t in written:
            stored = await session.get(Trial, t.id)
            stored.state, stored.message = TrialState.PRUNED, PENDING_SEND
    fresh = FakeOptimizer(db)  # a restarted application has no memory of what was sent
    sent = await engine.refill(fresh, row, 10, False)
    assert sent == 10 and len(fresh.engine.sent) == 10
    assert len(await trials(db, row.id)) == len(written)  # nothing regenerated while work is parked


async def test_next_generation_learns_from_results_and_never_repeats(db: Any) -> None:
    row = await make_task(db)
    opt = FakeOptimizer(db)
    await engine.refill(opt, row, 30, False)
    first = await trials(db, row.id)
    # Results as BRAIN's stats would arrive; the values are inputs to the loop, not asserted on.
    async with db.session() as session:
        for i, t in enumerate(first):
            stored = await session.get(Trial, t.id)
            stored.state = TrialState.COMPLETE if i % 5 else TrialState.FAIL
            stored.message = None if i % 5 else "Unknown operator foo"
            stored.result = {
                "stats": {
                    "sharpe": 0.5 + (i % 7) * 0.3,
                    "fitness": 0.4,
                    "turnover": 0.15,
                    "drawdown": 0.06,
                },
                "checks": [],
                "failedChecks": [],
            }
            stored.values = [1.0]
    await engine.refill(opt, row, 30, False)
    both = await trials(db, row.id)
    second = [t for t in both if t.generation == 1]
    assert second, "a second generation was written"
    assert not {t.params["key"] for t in second} & {t.params["key"] for t in first}
    assert any("mutation" in t.params["origin"] for t in second)  # bred from what scored


async def test_dashboard_and_candidates_agree_with_the_trials(db: Any) -> None:
    row = await make_task(db)
    opt = FakeOptimizer(db)
    await engine.refill(opt, row, 30, False)
    rows = await engine.rows_of(opt, row.id)
    run = SuperResearchParams.model_validate(row.sampler_params)
    candidates = report.evaluate(rows, run)
    board = report.dashboard(candidates, row.max_trials)
    assert board["generated"] == len(candidates) == len(rows)
    assert board["budget"] == 60 and board["generated"] == board["locallyValid"]
    assert (
        board["simulated"] + board["failed"] + board["submitted"] + board["waiting"]
        == board["generated"]
    )
    assert all(c["submittable"] is False for c in candidates)  # nothing is submittable unproven


async def test_submittable_needs_the_platform_checks_to_pass(db: Any) -> None:
    row = await make_task(db)
    opt = FakeOptimizer(db)
    await engine.refill(opt, row, 10, False)
    async with db.session() as session:
        stored = (
            await session.scalars(
                select(Trial).where(Trial.study_id == row.id).order_by(Trial.number)
            )
        ).first()
        stored.state = TrialState.COMPLETE
        stored.result = {
            "stats": {"sharpe": 3.0, "fitness": 2.0, "turnover": 0.2},
            "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}],
            "failedChecks": ["LOW_SHARPE"],
        }
    run = SuperResearchParams.model_validate(row.sampler_params)
    out = report.evaluate(await engine.rows_of(opt, row.id), run)[0]
    assert out["sharpe"] == 3.0 and out["submittable"] is False
    async with db.session() as session:
        stored = (
            await session.scalars(
                select(Trial).where(Trial.study_id == row.id).order_by(Trial.number)
            )
        ).first()
        stored.result = {
            "stats": {"sharpe": 3.0, "fitness": 2.0, "turnover": 0.2},
            "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
            "failedChecks": [],
        }
    assert report.evaluate(await engine.rows_of(opt, row.id), run)[0]["submittable"] is True


async def test_early_stop_and_budget(db: Any) -> None:
    candidates = [
        {"generation": g, "score": s, "outcome": "accepted"}
        for g, s in [(0, 2.0), (1, 1.0), (2, 1.5), (3, 1.2), (4, 1.1), (5, 0.9)]
    ]
    assert report.stalled(candidates, 4) is True
    assert report.stalled(candidates[:4], 4) is False
    candidates.append({"generation": 6, "score": 3.0, "outcome": "accepted"})
    assert report.stalled(candidates, 4) is False


async def test_concurrency_is_the_documented_three(db: Any) -> None:
    from alpha_harness.api.super_research import ResearchTask

    assert SUPER_ALPHA_SLOTS == 3
    with pytest.raises(ValueError):
        ResearchTask(region="USA", delay=1, universe="TOP3000", neutralization="MARKET", cores=4)
    task = ResearchTask(region="USA", delay=1, universe="TOP3000", neutralization="MARKET")
    assert task.cores == 3 and task.checked() is None


async def test_user_proposals_run_the_same_checker(db: Any) -> None:
    row = await make_task(
        db,
        proposals=[
            {
                "selection": "(operator_count <= 200) * (1 - self_correlation)",
                "combo": STATS + "max(ts_ir(stats.returns, 60), 0)",
            },
            {"selection": "turnover", "combo": "reduce_max(1)"},  # invalid: refused, never written
        ],
    )
    opt = FakeOptimizer(db)
    await engine.refill(opt, row, 10, False)
    written = await trials(db, row.id)
    assert any(t.params["family"] == "proposed" for t in written)
    assert all("reduce_max(1)" not in t.params["combo"] for t in written)


async def test_other_labs_are_unaffected(db: Any) -> None:
    from alpha_harness.labs.params import BY_SAMPLER, SUPER_LAB, SuperParams

    assert BY_SAMPLER[SUPER_LAB] is SuperParams and SUPER_RESEARCH != SUPER_LAB
    assert len(BY_SAMPLER) == 10
