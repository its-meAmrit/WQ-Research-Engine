"""Small constructors for expression trees, and the one place intermediate variables are made."""

from ..fastexpr import Node, render, walk_with_parents

STATS_VAR = "stats"


def num(value: float) -> Node:
    """A constant, spelled the way the parser reads it back: a negative is a minus in front."""
    if value < 0:
        return Node("unary", "-", (num(-value),))
    text = f"{value:g}" if float(value) != int(value) else str(int(value))
    return Node("num", text)


def ref(name: str) -> Node:
    return Node("name", name)


def call(name: str, *args: Node, **kwargs: Node) -> Node:
    return Node("call", name, tuple(args), tuple(kwargs.items()))


def binop(op: str, left: Node, right: Node) -> Node:
    return Node("binary", op, (left, right))


def uses_stats(tree: Node) -> bool:
    return any(
        n.kind == "name" and n.value.startswith(f"{STATS_VAR}.")
        for _, n, _ in walk_with_parents(tree)
    )


def program(tree: Node, *, hoist: bool = True) -> Node:
    """A complete expression: ``stats = generate_stats(alpha)`` first if the tree reads it, and
    any call repeated in the tree named once."""
    statements: list[Node] = []
    own: list[Node] = []
    if tree.kind == "seq":  # a template's own statements stay, in order, ahead of its value
        *own, tree = tree.args
    if uses_stats(tree) or any(uses_stats(s) for s in own):
        statements.append(Node("assign", STATS_VAR, (call("generate_stats", ref("alpha")),)))
    named: list[Node] = []
    if hoist:
        tree, named = _hoist(tree)
    statements.extend(own)
    statements.extend(named)
    if not statements:
        return tree
    return Node("seq", args=(*statements, tree))


def _count(tree: Node) -> dict[Node, int]:
    seen: dict[Node, int] = {}
    for _, node, _ in walk_with_parents(tree):
        if node.kind == "call" and node.value != "generate_stats":
            seen[node] = seen.get(node, 0) + 1
    return seen


def _hoist(tree: Node) -> tuple[Node, list[Node]]:
    named: list[Node] = []
    index = 0
    while True:
        repeated = [n for n, c in _count(tree).items() if c > 1]
        if not repeated:
            return tree, named
        # The largest repeat first, so what it contains is not named separately.
        target = max(repeated, key=lambda n: len(render(n)))
        index += 1
        label = f"v{index}"
        tree = _replace(tree, target, ref(label))
        named.append(Node("assign", label, (target,)))
        named = [
            Node("assign", s.value, (_replace(s.args[0], target, ref(label)),))
            if s.value != label
            else s
            for s in named
        ]


def _replace(node: Node, old: Node, new: Node) -> Node:
    if node == old:
        return new
    if not node.args and not node.kwargs:
        return node
    return Node(
        node.kind,
        node.value,
        tuple(_replace(a, old, new) for a in node.args),
        tuple((k, _replace(v, old, new)) for k, v in node.kwargs),
    )
