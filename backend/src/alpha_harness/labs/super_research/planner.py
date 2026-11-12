"""Writes the next generation of SuperAlpha candidates from what is known so far.

Pure with respect to the database: it is handed the history (what was tried, how it scored,
what BRAIN refused) and returns proposals. Given the same history, config and seed it returns
the same proposals, which is what makes a research task reproducible.
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..fastexpr import Node, parse, render
from . import templates as tpl
from .canon import canonical, key_of
from .checker import check
from .grammar import Controls, Grammar, accepted
from .mutate import mutate
from .registry import Ctx, Registry, Status


@dataclass(slots=True)
class Config:
    controls: Controls = field(default_factory=Controls)
    #: "templates", "grammar", "evolve" or "mixed".
    strategy: str = "mixed"
    mutation_probability: float = 0.6
    #: Elite candidates kept as parents.
    elite: int = 8
    #: At most this many elites per strategy family, so one idea cannot fill the pool.
    per_family: int = 2
    decays: tuple[int, ...] = (4, 10)
    truncations: tuple[float, ...] = (0.05,)
    activations: tuple[str, ...] = ("IS", "OS")
    selection_limit: int = 30
    operator_cap: int = 266
    #: Generations with no new best before the task may stop early.
    patience: int = 4


@dataclass(frozen=True, slots=True)
class Done:
    """One earlier candidate, reduced to what planning needs."""

    key: str
    selection: str
    combo: str
    family: str
    score: float | None
    settings: tuple[int, float, str]
    state: str = "COMPLETE"


@dataclass(slots=True)
class Proposal:
    selection: str
    combo: str
    decay: int
    truncation: float
    activation: str
    family: str
    origin: str
    explanation: str
    operators: list[str]
    key: str
    parents: list[str] = field(default_factory=list)
    local: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family} · {self.origin}"


def _canon_text(text: str) -> str:
    return canonical(parse(text))


def proposal_key(selection: str, combo: str, decay: int, truncation: float, activation: str) -> str:
    return key_of(
        _canon_text(selection), _canon_text(combo), str(decay), f"{truncation:g}", activation
    )


class Planner:
    def __init__(
        self,
        registry: Registry,
        config: Config,
        history: list[Done],
        *,
        generation: int,
        avoid: frozenset[str] = frozenset(),
    ) -> None:
        self.registry = registry
        self.config = config
        self.history = history
        self.generation = generation
        controls = config.controls
        self.controls = Controls(
            max_depth=controls.max_depth,
            max_operators=controls.max_operators,
            max_variables=controls.max_variables,
            seed=controls.seed,
            exploration=controls.exploration,
            disabled_operators=controls.disabled_operators | avoid,
            disabled_families=controls.disabled_families,
            preferred_families=controls.preferred_families,
        )
        # Seeded by the task's seed and the generation, never by the clock.
        self.rng = random.Random(f"{controls.seed}:{generation}")
        self.grammar = Grammar(registry, self.controls, self.rng)
        self.seen = {h.key for h in history}
        self.used: Counter[str] = Counter()

    # -- pools -------------------------------------------------------------------------------

    def elites(self) -> list[Done]:
        scored = sorted(
            (h for h in self.history if h.score is not None and h.state == "COMPLETE"),
            key=lambda h: -(h.score or 0),
        )
        out: list[Done] = []
        per: Counter[str] = Counter()
        for h in scored:
            if per[h.family] >= self.config.per_family:
                continue
            per[h.family] += 1
            out.append(h)
            if len(out) >= self.config.elite:
                break
        return out

    def usable(self, template: tpl.Template, ctx: Ctx) -> bool:
        off = self.controls.disabled_operators
        return (
            all(
                (spec := self.registry.get(n)) is not None
                and spec.status == Status.VERIFIED
                and n not in off
                and self.registry.eligibility(
                    n, ctx, disabled=off, families_off=self.controls.disabled_families
                ).eligible
                for n in template.needs
            )
            and template.family not in self.controls.disabled_families
        )

    def available_templates(self, ctx: Ctx) -> list[tpl.Template]:
        pool = tpl.COMBO_TEMPLATES if ctx == Ctx.COMBO else tpl.SELECTION_TEMPLATES
        return [t for t in pool if self.usable(t, ctx)]

    # -- pieces ------------------------------------------------------------------------------

    def new_combo(self) -> tuple[Node, str, str, str] | None:
        """``(tree, family, origin, explanation)`` for a combo not taken from the elites."""
        strategy = self.config.strategy
        use_template = strategy == "templates" or (
            strategy in ("mixed", "evolve") and self.rng.random() < 0.5
        )
        if use_template:
            options = self.available_templates(Ctx.COMBO)
            if options:
                weights = [
                    1.0 / (1 + sum(self.used[o] for o in t.needs) / max(len(t.needs), 1))
                    for t in options
                ]
                t = self.rng.choices(options, weights=weights)[0]
                params = t.sample(self.rng)
                return t.build(params), t.family, f"template {t.key} {params}", t.explanation
            if strategy == "templates":
                return None
        for _ in range(12):
            tree = self.grammar.combo()
            checked = accepted(tree, Ctx.COMBO, self.registry, self.controls)
            if checked is not None and tree is not None:
                return (
                    tree,
                    "grammar",
                    "grammar",
                    "Grammar-built weighting over the verified operators.",
                )
        return None

    def new_selection(self) -> tuple[Node, str, str, str] | None:
        strategy = self.config.strategy
        cap = self.config.operator_cap
        use_template = strategy == "templates" or (
            strategy in ("mixed", "evolve") and self.rng.random() < 0.6
        )
        if use_template:
            options = self.available_templates(Ctx.SELECTION)
            if options:
                t = self.rng.choice(options)
                params = t.sample(self.rng)
                return (
                    tpl.gate(t.build(params), cap),
                    t.family,
                    f"template {t.key} {params}",
                    t.explanation,
                )
        for _ in range(12):
            tree = self.grammar.selection(cap)
            if (
                tree is not None
                and accepted(tree, Ctx.SELECTION, self.registry, self.controls) is not None
            ):
                return tree, "grammar", "grammar", "Grammar-built ranking of component Alphas."
        return None

    def evolve(self, ctx: Ctx, parents: list[Done]) -> tuple[Node, str, str, str, list[str]] | None:
        pick = self.rng.choice(parents)
        other = self.rng.choice(parents)
        text = pick.combo if ctx == Ctx.COMBO else pick.selection
        partner = other.combo if ctx == Ctx.COMBO else other.selection
        result = mutate(self.rng, parse(text), ctx, self.grammar, partner=parse(partner))
        if result is None:
            return None
        tree, kind = result
        if ctx == Ctx.SELECTION and not _gated(tree):
            tree = tpl.gate(tree, self.config.operator_cap)
        return (
            tree,
            pick.family,
            f"{kind} mutation",
            f"A {kind} change to a strong {pick.family} candidate.",
            [pick.key],
        )

    # -- one generation ----------------------------------------------------------------------

    def plan(self, count: int, *, focus: list[str] | None = None) -> list[Proposal]:
        out: list[Proposal] = []
        parents = self.elites()
        attempts = 0
        focus = list(focus or [])
        while len(out) < count and attempts < count * 40:
            attempts += 1
            made = self.one(parents, focus)
            if made is None or made.key in self.seen:
                continue
            self.seen.add(made.key)
            for op in made.operators:
                self.used[op] += 1
                self.grammar.uses[op] += 1
            out.append(made)
        return out

    def one(self, parents: list[Done], focus: list[str]) -> Proposal | None:
        cfg = self.config
        rng = self.rng
        combo = selection = None
        origin, family, why, lineage = "", "", "", []
        exploit = (
            bool(parents)
            and cfg.strategy in ("evolve", "mixed")
            and rng.random() > self.controls.exploration * 0.6
        )
        if focus:
            # A slot spent on an operator no candidate has used yet.
            tree = self.focused(focus.pop(0))
            if tree is not None:
                combo, family, origin, why = (
                    tree,
                    "coverage",
                    "coverage-guided",
                    "Built around an operator the research has not yet explored.",
                )
        if combo is None and exploit and rng.random() < cfg.mutation_probability:
            made = self.evolve(Ctx.COMBO, parents)
            if made is not None:
                combo, family, origin, why, lineage = made
        if combo is None:
            fresh = self.new_combo()
            if fresh is None:
                return None
            combo, family, origin, why = fresh
        # The selection: an elite's own, a mutated one, or a new one.
        if exploit and rng.random() < 0.5:
            sel_text = rng.choice(parents).selection
            selection = parse(sel_text)
            if rng.random() < cfg.mutation_probability * 0.5:
                made = self.evolve(Ctx.SELECTION, parents)
                if made is not None:
                    selection = made[0]
        else:
            fresh_sel = self.new_selection()
            if fresh_sel is None:
                return None
            selection = fresh_sel[0]
        combo_checked = check(combo, Ctx.COMBO, self.registry, self.controls.limits())
        sel_checked = check(selection, Ctx.SELECTION, self.registry, self.controls.limits())
        if not (combo_checked.ok and sel_checked.ok):
            return None
        decay = rng.choice(cfg.decays)
        truncation = rng.choice(cfg.truncations)
        activation = rng.choice(cfg.activations)
        combo_text, sel_text = combo_checked.text, sel_checked.text
        return Proposal(
            selection=sel_text,
            combo=combo_text,
            decay=decay,
            truncation=truncation,
            activation=activation,
            family=family,
            origin=origin,
            explanation=why,
            operators=sorted(set(combo_checked.operators) | set(sel_checked.operators)),
            key=proposal_key(sel_text, combo_text, decay, truncation, activation),
            parents=lineage,
            local={
                "depth": combo_checked.depth,
                "operators": combo_checked.operator_count,
                "selectionOperators": sel_checked.operator_count,
            },
        )

    def focused(self, name: str) -> Node | None:
        """A combo containing ``name``, found by weighting the grammar heavily toward it."""
        spec = self.registry.get(name)
        if spec is None:
            return None
        for _ in range(60):
            tree = self.grammar.combo_with(name)
            checked = accepted(tree, Ctx.COMBO, self.registry, self.controls)
            if checked is not None and name in checked.operators:
                return tree
        return None


def _gated(tree: Node) -> bool:
    return "operator_count" in render(tree)
