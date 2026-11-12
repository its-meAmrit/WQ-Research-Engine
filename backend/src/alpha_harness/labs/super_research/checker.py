"""The local gate: whether an expression may be simulated at all.

It reads a ``fastexpr.Node`` tree, infers what every part of it is (a number, one value per Alpha,
one per pair of Alphas, or an Alpha's statistics) and refuses whatever the registry cannot
vouch for. Local acceptance never means BRAIN accepts: BRAIN stays the authority, and a
rejection from it is recorded against the operators the expression used.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..fastexpr import Node, ParseError, children, operator_count, parse, render
from .registry import (
    ALPHA,
    BOOL,
    ENUM,
    FLOAT,
    INT,
    JOIN,
    LITERAL_NAMES,
    MAT,
    NUM,
    STR,
    UNIVERSE,
    VEC,
    Ctx,
    OperatorSpec,
    Param,
    Registry,
    Status,
    Ty,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: BRAIN's own words for a few common slips, so a rejection says what to change.
SHAPE_RANK = {Ty.SCALAR: 0, Ty.VEC: 1, Ty.MATRIX: 2}


@dataclass(frozen=True, slots=True)
class Limits:
    max_depth: int = 8
    max_operators: int = 24
    max_variables: int = 4
    max_statements: int = 6


@dataclass(slots=True)
class Checked:
    problems: list[str] = field(default_factory=list)
    ty: Ty | None = None
    #: Operator names called, in order of first use.
    operators: list[str] = field(default_factory=list)
    operator_count: int = 0
    depth: int = 0
    text: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems and self.ty is not None


def join(a: Ty, b: Ty) -> Ty | None:
    """The shape of an elementwise result, or ``None`` when two shapes cannot be combined.

    A scalar joins anything. A vector (one per Alpha) and a matrix (one per pair) have no
    agreed alignment, so they are refused rather than broadcast on a guess.
    """
    if Ty.STATS in (a, b):
        return None
    if a == b:
        return a
    if a == Ty.SCALAR:
        return b
    if b == Ty.SCALAR:
        return a
    return None


def depth_of(node: Node) -> int:
    kids = children(node)
    return 1 + (max(depth_of(k) for k in kids) if kids else 0)


class Walker:
    def __init__(self, registry: Registry, ctx: Ctx, limits: Limits) -> None:
        self.registry = registry
        self.ctx = ctx
        self.limits = limits
        self.problems: list[str] = []
        self.used: list[str] = []
        self.vars: dict[str, Ty] = {}

    def fail(self, message: str) -> None:
        if message not in self.problems:
            self.problems.append(message)

    # -- names -----------------------------------------------------------------------------

    def name(self, node: Node, *, as_arg: bool = False) -> Ty | None:
        text = node.value
        if text.lower() in LITERAL_NAMES:
            return Ty.SCALAR
        if text in self.vars:
            return self.vars[text]
        if "." in text:
            base, _, member = text.partition(".")
            if self.vars.get(base) != Ty.STATS:
                self.fail(f"{text}: {base} is not an Alpha's statistics here.")
                return None
            known = {m.name: m for m in self.registry.stats_members}
            found = known.get(member)
            if found is None or found.status != Status.VERIFIED:
                self.fail(f"{text}: {member} is not a verified statistic.")
                return None
            return found.ty
        if self.ctx == Ctx.SELECTION:
            field_ = {f.name: f for f in self.registry.selection_fields}.get(text)
            if field_ is not None:
                if field_.status != Status.VERIFIED:
                    self.fail(f"{text} is not a verified selection property.")
                    return None
                return field_.ty
            if text == "universe" and as_arg:
                return Ty.SCALAR
        elif text == "alpha" and as_arg:
            return Ty.SCALAR
        self.fail(f"{text} is not available in a {self.ctx.value} expression.")
        return None

    # -- expressions -----------------------------------------------------------------------

    def infer(self, node: Node, *, as_arg: bool = False) -> Ty | None:
        match node.kind:
            case "num":
                return Ty.SCALAR
            case "str":
                self.fail(f"The string {node.value} is not valid here.")
                return None
            case "name":
                return self.name(node, as_arg=as_arg)
            case "unary":
                inner = self.infer(node.args[0])
                if inner == Ty.STATS:
                    self.fail(
                        "Statistics cannot be used as numbers; take a member such as .returns."
                    )
                    return None
                return inner
            case "binary":
                return self.binary(node)
            case "ternary":
                condition, yes, no = (self.infer(a) for a in node.args)
                if None in (condition, yes, no):
                    return None
                if condition is None or yes is None or no is None:
                    return None
                picked = join(yes, no)
                if picked is None:
                    self.fail("The two branches of ? : have shapes that cannot be combined.")
                    return None
                whole = join(condition, picked)
                if whole is None:
                    self.fail("The condition's shape does not match its branches.")
                return whole
            case "call":
                return self.call(node)
            case "assign" | "seq":
                self.fail("Assignments belong at the top level of an expression.")
                return None
            case _:
                self.fail(f"Unsupported expression kind {node.kind}.")
                return None

    def binary(self, node: Node) -> Ty | None:
        left, right = (self.infer(a) for a in node.args)
        if left is None or right is None:
            return None
        if node.value == "/" and node.args[1].kind == "num" and float(node.args[1].value) == 0:
            self.fail("Division by the constant zero.")
        if node.value == "^" and node.args[0].kind == "num" and float(node.args[0].value) < 0:
            self.fail("A negative constant raised to a power is undefined.")
        shape = join(left, right)
        if shape is None:
            self.fail(
                f"Cannot apply {node.value} to {left.value} and {right.value}: "
                "shapes are not compatible."
            )
        return shape

    # -- calls -----------------------------------------------------------------------------

    def call(self, node: Node) -> Ty | None:
        name = node.value
        spec = self.registry.get(name)
        if spec is None:
            self.fail(f"{name} is not a documented SuperAlpha operator.")
            return None
        if self.ctx not in spec.contexts:
            kind = "Selection" if Ctx.SELECTION in spec.contexts else "Combo"
            self.fail(f"{name} is {kind}-only: it is not valid in a {self.ctx.value} expression.")
            return None
        if spec.params is None:
            self.fail(f"{name} has no verified signature: it cannot be used yet.")
            return None
        if spec.status != Status.VERIFIED:
            self.fail(f"{name} is {spec.status.value.replace('_', ' ')}: it cannot be used yet.")
            return None
        if name not in self.used:
            self.used.append(name)
        bound = self.bind(spec, node)
        if bound is None:
            return None
        shapes: list[Ty] = []
        ok = True
        for param, arg in bound:
            ty = self.argument(spec, param, arg)
            if ty is None:
                ok = False
            elif param.kind in (NUM, VEC, MAT):
                shapes.append(ty)
        if not ok:
            return None
        return self.result(spec, shapes)

    def bind(self, spec: OperatorSpec, node: Node) -> list[tuple[Param, Node]] | None:
        params = spec.params or ()
        positional = [p for p in params if not p.variadic]
        variadic = next((p for p in params if p.variadic), None)
        bound: list[tuple[Param, Node]] = []
        taken: set[str] = set()
        if len(node.args) > len(positional) and variadic is None:
            self.fail(
                f"{spec.name} takes at most {len(positional)} inputs; {len(node.args)} given."
            )
            return None
        for index, arg in enumerate(node.args):
            if index < len(positional):
                bound.append((positional[index], arg))
                taken.add(positional[index].name)
            elif variadic is not None:
                bound.append((variadic, arg))
        by_name = {p.name: p for p in positional}
        for key, value in node.kwargs:
            param = by_name.get(key)
            if param is None:
                self.fail(f"{spec.name} has no parameter named {key}.")
                return None
            if key in taken:
                self.fail(f"{spec.name}: {key} is given twice.")
                return None
            taken.add(key)
            bound.append((param, value))
        missing = [p.name for p in positional if p.required and p.name not in taken]
        if missing:
            self.fail(f"{spec.name} is missing {', '.join(missing)}.")
            return None
        return bound

    def argument(self, spec: OperatorSpec, param: Param, arg: Node) -> Ty | None:
        label = f"{spec.name}({param.name})"
        kind = param.kind
        if kind in (INT, FLOAT):
            value = constant(arg)
            if value is None:
                self.fail(f"{label} must be a numeric constant.")
                return None
            if kind == INT and value != int(value):
                self.fail(f"{label} must be a whole number; got {value:g}.")
                return None
            if (param.low is not None and value < param.low) or (
                param.high is not None and value > param.high
            ):
                self.fail(
                    f"{label} must be between {param.low:g} and {param.high:g}; got {value:g}."
                )
                return None
            if spec.name in ("log", "sqrt") and value < 0:
                self.fail(f"{spec.name} of a negative constant is undefined.")
            return Ty.SCALAR
        if kind == BOOL:
            if arg.kind == "name" and arg.value.lower() in ("true", "false"):
                return Ty.SCALAR
            self.fail(f"{label} must be true or false.")
            return None
        if kind == ENUM:
            if arg.kind in ("name", "str") and arg.value.strip("\"'") in param.choices:
                return Ty.SCALAR
            self.fail(f"{label} must be one of {', '.join(param.choices)}.")
            return None
        if kind == STR:
            if arg.kind == "str":
                return Ty.SCALAR
            self.fail(f"{label} must be a quoted string.")
            return None
        if kind == ALPHA:
            if self.ctx == Ctx.COMBO and arg.kind == "name" and arg.value == "alpha":
                return Ty.SCALAR
            self.fail(f"{label} takes the combo's alpha input.")
            return None
        if kind == UNIVERSE:
            if self.ctx == Ctx.SELECTION and arg.kind == "name" and arg.value == "universe":
                return Ty.SCALAR
            self.fail(f"{label} takes the selection's universe input.")
            return None
        ty = self.infer(arg)
        if ty is None:
            return None
        if ty == Ty.STATS:
            self.fail(f"{label} was given an Alpha's statistics; take a member such as .returns.")
            return None
        if kind == VEC and ty != Ty.VEC:
            self.fail(f"{label} needs one value per Alpha, not a {ty.value}.")
            return None
        if kind == MAT and ty != Ty.MATRIX:
            self.fail(f"{label} needs an Alpha-by-Alpha matrix, not a {ty.value}.")
            return None
        return ty

    def result(self, spec: OperatorSpec, shapes: Sequence[Ty]) -> Ty | None:
        match spec.returns:
            case "c":
                return Ty.SCALAR
            case "v":
                return Ty.VEC
            case "m":
                return Ty.MATRIX
            case "S":
                return Ty.STATS
            case v if v == JOIN:
                shape = Ty.SCALAR
                for each in shapes:
                    joined = join(shape, each)
                    if joined is None:
                        self.fail(f"{spec.name}: its inputs have shapes that cannot be combined.")
                        return None
                    shape = joined
                return shape
            case _:
                return None


def constant(node: Node) -> float | None:
    """A literal number, with a leading minus allowed."""
    if node.kind == "num":
        return float(node.value)
    if node.kind == "unary" and node.value == "-" and node.args[0].kind == "num":
        return -float(node.args[0].value)
    return None


def check(
    text_or_tree: str | Node,
    ctx: Ctx,
    registry: Registry,
    limits: Limits | None = None,
) -> Checked:
    """Every local rule, in one pass. ``Checked.ok`` is what gates simulation."""
    limits = limits or Limits()
    out = Checked()
    try:
        tree = parse(text_or_tree) if isinstance(text_or_tree, str) else text_or_tree
    except ParseError as exc:
        out.problems.append(f"Syntax: {exc}")
        return out
    out.text = render(tree)
    # The serialized form must read back as the same tree: what is checked is what is sent.
    try:
        if parse(out.text) != tree:
            out.problems.append("Serialization does not round-trip.")
            return out
    except ParseError as exc:
        out.problems.append(f"Serialization is not valid BRAIN syntax: {exc}")
        return out

    statements = list(tree.args) if tree.kind == "seq" else [tree]
    if len(statements) > limits.max_statements:
        out.problems.append(f"More than {limits.max_statements} statements.")
    out.depth = max(depth_of(s) for s in statements)
    out.operator_count = operator_count(tree)
    if out.depth > limits.max_depth:
        out.problems.append(f"Nested {out.depth} deep; the limit is {limits.max_depth}.")
    if out.operator_count > limits.max_operators:
        out.problems.append(f"{out.operator_count} operators; the limit is {limits.max_operators}.")

    walker = Walker(registry, ctx, limits)
    final = statements[-1]
    if final.kind == "assign":
        out.problems.append("The last statement must be the expression's value, not an assignment.")
    for statement in statements[:-1]:
        if statement.kind != "assign":
            out.problems.append("Only the last statement may be a bare expression.")
            continue
        if statement.value in walker.vars:
            out.problems.append(f"{statement.value} is assigned twice.")
            continue
        if statement.value.lower() in LITERAL_NAMES or statement.value in (
            "alpha",
            "universe",
        ):
            out.problems.append(f"{statement.value} is a reserved name.")
            continue
        ty = walker.infer(statement.args[0])
        if ty is not None:
            walker.vars[statement.value] = ty
    if len(walker.vars) > limits.max_variables:
        out.problems.append(
            f"{len(walker.vars)} intermediate variables; the limit is {limits.max_variables}."
        )
    ty = walker.infer(final) if final.kind != "assign" else None
    out.problems.extend(p for p in walker.problems if p not in out.problems)
    out.operators = walker.used
    if ty is None:
        return out
    wanted = (Ty.VEC,) if ctx == Ctx.SELECTION else (Ty.VEC, Ty.SCALAR)
    if ty not in wanted:
        word = "one score per Alpha" if ctx == Ctx.SELECTION else "one weight per Alpha"
        out.problems.append(f"A {ctx.value} must end in {word}; this ends in a {ty.value}.")
        return out
    out.ty = ty
    return out
