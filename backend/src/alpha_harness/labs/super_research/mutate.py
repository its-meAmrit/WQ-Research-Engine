"""Evolution on typed trees. Every operator here returns a candidate tree or ``None``, and the
caller still runs the checker: a mutation never gets to skip validation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..fastexpr import Node, replace_at, walk_with_parents
from .build import num, program
from .canon import inline
from .checker import Limits, Walker, check, constant
from .registry import FLOAT, INT, Ctx, OperatorSpec, Param, Registry, Ty

if TYPE_CHECKING:
    import random

    from .grammar import Grammar

Path = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Site:
    path: Path
    node: Node
    ty: Ty


def typed_sites(tree: Node, ctx: Ctx, registry: Registry) -> list[Site]:
    """Every call in an inlined tree with the type it produces there."""
    walker = Walker(registry, ctx, Limits())
    walker.vars["stats"] = Ty.STATS
    sites: list[Site] = []
    for path, node, _ in walk_with_parents(tree):
        if node.kind != "call":
            continue
        before = list(walker.problems)
        ty = walker.infer(node)
        walker.problems[:] = before
        if ty is not None and ty != Ty.STATS:
            sites.append(Site(tuple(path), node, ty))
    return sites


def _shape(spec: OperatorSpec) -> tuple[tuple[tuple[str, bool, bool], ...], str]:
    return (tuple((p.kind, p.required, p.variadic) for p in spec.params or ()), spec.returns)


def interchangeable(spec: OperatorSpec, ctx: Ctx, grammar: Grammar) -> list[OperatorSpec]:
    """Operators that can stand where ``spec`` stands: same parameter kinds, same result."""
    return [
        other
        for other in grammar.eligible(ctx)
        if other.name != spec.name and _shape(other) == _shape(spec)
    ]


def mutate_parameter(rng: random.Random, tree: Node, registry: Registry) -> Node | None:
    """Nudge one numeric argument inside its documented range."""
    slots: list[tuple[Path, Param, float]] = []
    for path, node, up in walk_with_parents(tree):
        if up is None or up.kind != "call" or constant(node) is None or node.kind == "unary":
            continue
        spec = registry.get(up.value)
        if spec is None or not spec.params or path[-1] >= len(spec.params):
            continue
        param = spec.params[path[-1]]
        if param.kind in (INT, FLOAT):
            slots.append((tuple(path), param, float(node.value)))
    if not slots:
        return None
    path, param, value = rng.choice(slots)
    low = param.low if param.low is not None else 1
    high = param.high if param.high is not None else 512
    if param.kind == INT:
        scaled = max(round(value * rng.choice((0.5, 0.75, 1.5, 2))), 1)
        new = float(min(max(scaled, int(low)), int(high)))
    else:
        span = min(high - low, 2.0)
        new = round(min(max(value + rng.uniform(-0.25, 0.25) * span, low), high), 2)
    return None if new == value else replace_at(tree, path, num(new))


def substitute_operator(rng: random.Random, tree: Node, ctx: Ctx, grammar: Grammar) -> Node | None:
    sites = typed_sites(tree, ctx, grammar.registry)
    rng.shuffle(sites)
    for site in sites:
        spec = grammar.registry.get(site.node.value)
        if spec is None or spec.params is None:
            continue
        options = interchangeable(spec, ctx, grammar)
        if not options:
            continue
        other = grammar.pick(options)
        if other is None:
            continue
        return replace_at(
            tree, site.path, Node("call", other.name, site.node.args, site.node.kwargs)
        )
    return None


def replace_subtree(rng: random.Random, tree: Node, ctx: Ctx, grammar: Grammar) -> Node | None:
    """Replace one subtree with a freshly grown one of the same type and no more depth."""
    sites = typed_sites(tree, ctx, grammar.registry)
    if not sites:
        return None
    site = rng.choice(sites)
    pool = grammar.eligible(ctx)
    grown = grammar.build(
        site.ty, ctx, rng.randint(1, max(grammar.controls.max_depth - 2, 1)), pool
    )
    if grown is None or grown == site.node:
        return None
    return replace_at(tree, site.path, grown)


def crossover(
    rng: random.Random, left: Node, right: Node, ctx: Ctx, registry: Registry
) -> Node | None:
    """Swap in a subtree of the same type from the other parent."""
    mine = typed_sites(left, ctx, registry)
    theirs = typed_sites(right, ctx, registry)
    rng.shuffle(mine)
    for site in mine:
        matching = [t for t in theirs if t.ty == site.ty and t.node != site.node]
        if matching:
            donor = rng.choice(matching)
            return replace_at(left, site.path, donor.node)
    return None


def mutate(
    rng: random.Random,
    tree: Node,
    ctx: Ctx,
    grammar: Grammar,
    *,
    partner: Node | None = None,
) -> tuple[Node, str] | None:
    """One mutation, chosen at random from those that apply. The tree comes back as a complete
    program, with ``stats`` and repeated calls named as the generator names them."""
    flat = inline(tree)
    order = ["parameter", "operator", "subtree"] + (["crossover"] if partner is not None else [])
    rng.shuffle(order)
    for kind in order:
        child: Node | None
        if kind == "parameter":
            child = mutate_parameter(rng, flat, grammar.registry)
        elif kind == "operator":
            child = substitute_operator(rng, flat, ctx, grammar)
        elif kind == "subtree":
            child = replace_subtree(rng, flat, ctx, grammar)
        elif partner is not None:
            child = crossover(rng, flat, inline(partner), ctx, grammar.registry)
        else:
            child = None
        if child is None or child == flat:
            continue
        built = _program(child, ctx)
        # Growth limits belong to the caller; a type error never leaves here.
        if check(built, ctx, grammar.registry, Limits(99, 999, 99, 99)).ok:
            return built, kind
    return None


def _program(tree: Node, ctx: Ctx) -> Node:
    return program(tree) if ctx == Ctx.COMBO else tree
