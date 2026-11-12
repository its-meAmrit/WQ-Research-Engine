"""Grammar-guided generation: typed trees built from whatever the registry can vouch for.

Nothing here knows a shortlist of operators. The generator asks the registry which verified
operators return the type it needs, builds each argument at the type that parameter takes, and
leans toward operators the research has used least. A tree the checker refuses is discarded.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..fastexpr import Node, operator_count
from .build import binop, call, num, program, ref
from .checker import Checked, Limits, check
from .registry import (
    ALPHA,
    BOOL,
    ENUM,
    FLOAT,
    INT,
    JOIN,
    MAT,
    NUM,
    STR,
    UNIVERSE,
    VEC,
    Ctx,
    OperatorSpec,
    Param,
    Registry,
    Ty,
)

if TYPE_CHECKING:
    import random

LADDER = (3, 5, 10, 20, 40, 60, 120, 250, 500)


@dataclass(slots=True)
class Controls:
    """Everything the user can steer, all of it deterministic given ``seed``."""

    max_depth: int = 5
    max_operators: int = 14
    max_variables: int = 3
    seed: int = 0
    #: 0 follows the operators already used most; 1 spends every draw on the least explored.
    exploration: float = 0.5
    disabled_operators: frozenset[str] = frozenset()
    disabled_families: frozenset[str] = frozenset()
    #: Families that get extra weight, e.g. {"correlation_aware"}.
    preferred_families: frozenset[str] = frozenset()

    def limits(self) -> Limits:
        return Limits(
            max_depth=self.max_depth + 3,
            max_operators=self.max_operators,
            max_variables=self.max_variables,
        )


@dataclass(slots=True)
class Grammar:
    registry: Registry
    controls: Controls
    rng: random.Random
    #: How often each operator has already been put in a candidate.
    uses: Counter[str] = field(default_factory=Counter)
    #: While set, this operator is overwhelmingly likely wherever it fits.
    focus: str | None = None

    # -- choosing --------------------------------------------------------------------------

    def eligible(self, ctx: Ctx) -> list[OperatorSpec]:
        c = self.controls
        return [
            s
            for s in self.registry.specs.values()
            if self.registry.eligibility(
                s.name, ctx, disabled=c.disabled_operators, families_off=c.disabled_families
            ).eligible
        ]

    def weight(self, spec: OperatorSpec) -> float:
        rare = 1.0 / (1.0 + self.uses[spec.name])
        base = (1 - self.controls.exploration) + self.controls.exploration * rare * 4
        if spec.name == self.focus:
            base *= 60
        if spec.family in self.controls.preferred_families:
            base *= 2.5
        return max(base, 0.05)

    def pick(self, options: list[OperatorSpec]) -> OperatorSpec | None:
        if not options:
            return None
        return self.rng.choices(options, weights=[self.weight(o) for o in options])[0]

    # -- building --------------------------------------------------------------------------

    def leaf(self, ty: Ty, ctx: Ctx) -> Node | None:
        if ty == Ty.SCALAR:
            return num(self.rng.choice((0.01, 0.05, 0.1, 0.5, 1, 2)))
        if ty == Ty.VEC:
            sources = self.registry.vec_sources(ctx)
            if not sources:
                return None
            chosen = self.rng.choice(sources)
            return ref(chosen.name if ctx == Ctx.SELECTION else f"stats.{chosen.name}")
        return None

    def build(self, ty: Ty, ctx: Ctx, depth: int, pool: list[OperatorSpec]) -> Node | None:
        if depth <= 0:
            if ty == Ty.MATRIX:
                inner = self.leaf(Ty.VEC, ctx)
                spec = next((s for s in pool if s.name == "self_corr"), None)
                return (
                    None
                    if inner is None or spec is None
                    else self.apply(spec, ctx, 0, pool, [inner])
                )
            return self.leaf(ty, ctx)
        options = [s for s in pool if self.yields(s, ty)]
        # Leaves stay in play so shallow trees happen; deeper levels lean on operators.
        if ty != Ty.MATRIX and self.rng.random() < 0.18 + 0.1 * (
            self.controls.max_depth - depth
        ) / max(self.controls.max_depth, 1):
            return self.leaf(ty, ctx)
        spec = self.pick(options)
        if spec is None:
            return self.leaf(ty, ctx)
        return self.apply(spec, ctx, depth, pool, want=ty)

    @staticmethod
    def yields(spec: OperatorSpec, ty: Ty) -> bool:
        if spec.returns == JOIN:
            data = [p for p in spec.params or () if p.kind in (NUM, VEC, MAT)]
            if ty == Ty.MATRIX:
                # only an elementwise operator passes a matrix through: one that wants a
                # vector input cannot be handed one
                return bool(data) and all(p.kind == NUM for p in data)
            return ty == Ty.VEC and bool(data)
        return {"v": Ty.VEC, "m": Ty.MATRIX, "c": Ty.SCALAR}.get(spec.returns) == ty

    def apply(
        self,
        spec: OperatorSpec,
        ctx: Ctx,
        depth: int,
        pool: list[OperatorSpec],
        forced: list[Node] | None = None,
        want: Ty = Ty.VEC,
    ) -> Node | None:
        args: list[Node] = []
        kwargs: list[tuple[str, Node]] = []
        forced = list(forced or [])
        first_data = True
        for param in spec.params or ():
            if param.variadic:
                continue
            if not param.required and self.rng.random() < 0.6:
                continue
            made = forced.pop(0) if forced and param.kind in (NUM, VEC, MAT) else None
            node = made or self.argument(param, ctx, depth, pool, want, first_data)
            if node is None:
                return None
            if param.kind in (NUM, VEC, MAT):
                first_data = False
            if param.required:
                args.append(node)
            else:
                # An optional input after an omitted one has to be named.
                kwargs.append((param.name, node))
        return Node("call", spec.name, tuple(args), tuple(kwargs))

    def argument(
        self,
        param: Param,
        ctx: Ctx,
        depth: int,
        pool: list[OperatorSpec],
        want: Ty,
        first_data: bool,
    ) -> Node | None:
        kind = param.kind
        if kind == INT:
            low, high = (
                int(param.low if param.low is not None else 1),
                int(param.high if param.high is not None else 512),
            )
            rungs = [w for w in LADDER if low <= w <= high] or [low]
            return num(self.rng.choice(rungs))
        if kind == FLOAT:
            low = param.low if param.low is not None else 0.0
            high = param.high if param.high is not None else 1.0
            if param.default is not None and self.rng.random() < 0.5:
                return num(float(param.default))
            top = min(high, low + 1.0) if high - low > 20 else high
            return num(round(self.rng.uniform(low, top), 2))
        if kind == BOOL:
            return ref(self.rng.choice(("true", "false")))
        if kind == ENUM:
            return ref(self.rng.choice(param.choices))
        if kind == ALPHA:
            return ref("alpha")
        if kind == UNIVERSE:
            return ref("universe")
        if kind == STR:
            return None
        if kind == MAT:
            return self.build(Ty.MATRIX, ctx, depth - 1, pool)
        if kind == VEC:
            return self.build(Ty.VEC, ctx, depth - 1, pool)
        # NUM: any shape. Follow the shape being built for the first input; later ones may be
        # a constant, which broadcasts.
        shape = want if first_data else (Ty.SCALAR if self.rng.random() < 0.4 else want)
        return self.build(shape, ctx, depth - 1, pool)

    # -- whole expressions --------------------------------------------------------------

    def combo(self) -> Node | None:
        pool = self.eligible(Ctx.COMBO)
        body = self.build(Ty.VEC, Ctx.COMBO, self.controls.max_depth, pool)
        if body is None:
            return None
        # Weights must be positive: a zero, negative or NaN weight drops an Alpha, so the
        # grammar never leaves that to chance.
        positive = {"rank", "sigmoid", "abs", "exp", "ts_rank"}
        if not (body.kind == "call" and body.value in positive):
            body = call("max", body, num(self.rng.choice((0.0, 0.01, 0.05))))
        return program(body)

    def combo_with(self, name: str) -> Node | None:
        """A combo built while ``name`` is strongly favoured."""
        self.focus = name
        try:
            return self.combo()
        finally:
            self.focus = None

    def selection(self, cap: int) -> Node | None:
        pool = self.eligible(Ctx.SELECTION)
        body = self.build(Ty.VEC, Ctx.SELECTION, self.controls.max_depth, pool)
        if body is None:
            return None
        return binop("*", binop("<=", ref("operator_count"), num(cap)), body)


IDEMPOTENT = frozenset({"max", "min", "abs", "rank", "sign", "zscore"})
#: What makes a combo depend on how the Alphas have been doing, not just on today's return.
HISTORY_PREFIXES = ("ts_", "self_corr", "reduce_", "hump", "jump_", "kth_", "last_diff")


def nontrivial(tree: Node, ctx: Ctx, used: list[str]) -> bool:
    """Valid is not the same as worth a simulation. Refuse the shapes that cannot say anything:
    a weight that is today's raw return, an operator wrapped in itself, a selection that reads
    a single property."""
    from ..fastexpr import walk_with_parents

    nodes = [n for _, n, _ in walk_with_parents(tree)]
    conditions = {
        "greater",
        "less",
        "equal",
        "is_nan",
        "is_not_nan",
        "is_finite",
        "and",
        "or",
        "not",
    }
    for node in nodes:
        if node.kind != "call":
            continue
        windowed = node.value.startswith("ts_")
        # An operator fed to itself adds nothing, except a window over a window.
        if not windowed and any(a.kind == "call" and a.value == node.value for a in node.args):
            return False
        # The same input twice says nothing: max(x, x), if_else(c, x, x).
        if len(node.args) > 1 and len(
            {a for a in node.args if not windowed or a.kind != "num"}
        ) < len([a for a in node.args if not windowed or a.kind != "num"]):
            return False
        if node.value == "if_else" and node.args:
            test = node.args[0]
            comparison = test.kind == "binary" and test.value in (
                "<",
                ">",
                "<=",
                ">=",
                "==",
                "!=",
                "&&",
                "||",
            )
            if not (comparison or (test.kind == "call" and test.value in conditions)):
                return False
    if ctx == Ctx.COMBO:
        return any(name.startswith(HISTORY_PREFIXES) for name in used)
    # operator_count is in every selection already, as the 8,000-operator gate.
    fields = {
        n.value for n in nodes if n.kind == "name" and n.value not in ("universe", "operator_count")
    }
    return len(fields) >= 2


def accepted(tree: Node | None, ctx: Ctx, registry: Registry, controls: Controls) -> Checked | None:
    """The checked form of a generated tree, or ``None`` if it fails any local rule or is
    trivial."""
    if tree is None:
        return None
    result = check(tree, ctx, registry, controls.limits())
    if not result.ok or operator_count(tree) > controls.max_operators + 2:
        return None
    if not nontrivial(tree, ctx, result.operators):
        return None
    return result
