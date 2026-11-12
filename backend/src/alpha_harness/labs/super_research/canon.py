"""Two expressions that compute the same thing should count once.

``canonical`` inlines intermediate variables, orders the inputs of commutative operators, and
drops arithmetic that changes nothing, so a rewrite that only renames a variable or swaps
``a + b`` for ``b + a`` hashes to the same key and is never simulated twice.
"""

import hashlib

from ..fastexpr import Node, render

COMMUTATIVE_CALLS = frozenset({"add", "multiply", "max", "min", "and", "or", "equal"})
COMMUTATIVE_BINARY = frozenset({"+", "*", "&&", "||", "==", "!="})


def inline(tree: Node) -> Node:
    """The tree with every assigned variable replaced by what it was assigned."""
    if tree.kind != "seq":
        return tree
    env: dict[str, Node] = {}
    for statement in tree.args[:-1]:
        env[statement.value] = _substitute(statement.args[0], env)
    return _substitute(tree.args[-1], env)


def _substitute(node: Node, env: dict[str, Node]) -> Node:
    if node.kind == "name" and node.value in env:
        return env[node.value]
    if not node.args and not node.kwargs:
        return node
    return Node(
        node.kind,
        node.value,
        tuple(_substitute(a, env) for a in node.args),
        tuple((k, _substitute(v, env)) for k, v in node.kwargs),
    )


def _is(node: Node, value: float) -> bool:
    return node.kind == "num" and float(node.value) == value


def _simplify(node: Node) -> Node:
    if node.kind == "binary":
        left, right = node.args
        if node.value == "*" and (_is(left, 1) or _is(right, 1)):
            return right if _is(left, 1) else left
        if node.value == "+" and (_is(left, 0) or _is(right, 0)):
            return right if _is(left, 0) else left
        if node.value == "-" and _is(right, 0):
            return left
        if node.value == "/" and _is(right, 1):
            return left
    return node


def _canon(node: Node) -> Node:
    if not node.args and not node.kwargs:
        return node
    args = tuple(_canon(a) for a in node.args)
    kwargs = tuple(sorted((k, _canon(v)) for k, v in node.kwargs))
    rebuilt = Node(node.kind, node.value, args, kwargs)
    if (rebuilt.kind == "call" and rebuilt.value in COMMUTATIVE_CALLS) or (
        rebuilt.kind == "binary" and rebuilt.value in COMMUTATIVE_BINARY
    ):
        rebuilt = Node(rebuilt.kind, rebuilt.value, tuple(sorted(args, key=render)), kwargs)
    return _simplify(rebuilt)


def canonical(tree: Node) -> str:
    return render(_canon(inline(tree)))


def key_of(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]
