"""Static validation of one candidate: what this application can check without BRAIN.

Three outcomes, and a candidate is never marked better than the checks below can support:

``valid``         every static check passed: *valid for task preview*, nothing more.
``needs_review``  well formed, but something a person should read first (timing, a proxy, a
                  complexity threshold, a point-in-time description).
``blocked``       known not to run: unreadable, an unknown operator or field, a wrong number
                  of inputs, a vector field misused, a setting BRAIN does not offer here, or a
                  scope the catalogue does not hold.

A ``valid`` candidate can still fail BRAIN's own checks and can perform badly; ``NOT_CHECKED``
says what this module cannot see, and every result carries it.
"""

from datetime import UTC, datetime
from typing import Any, Protocol

from ..brain.platform import GROUPING_FIELDS, LANGUAGE, MAX_DECAY
from ..brain.settings_schema import validate_settings
from ..catalog.queries import Tuple4
from ..labs import fastexpr
from ..research.feasibility import pit_status

VALID = "valid"
REVIEW = "needs_review"
BLOCKED = "blocked"

#: This application's own thresholds, not BRAIN's: a long expression is harder to review and
#: has more degrees of freedom, so it is flagged for reading rather than refused.
MAX_OPERATORS = 25
MAX_DEPTH = 10
MAX_FIELDS = 5

#: What static checking here cannot establish. Shown with every validation.
NOT_CHECKED = (
    "Whether each operator's inputs are the right kind of data (matrix, vector, group): only "
    "vector and group field misuse is checked, and only by field type.",
    "BRAIN's own simulation checks (turnover, sub-universe, weight concentration, correlation, "
    "Sharpe and fitness thresholds): only BRAIN can evaluate them.",
    "Whether the signal predicts returns. A well-formed expression has no known predictive value.",
    "Look-ahead beyond negative look-backs and field descriptions that say restated or latest.",
)

REQUIRED_SETTINGS = ("instrumentType", "region", "delay", "universe", "language")
CONFIRMED_SETTINGS = ("neutralization", "decay", "truncation")


class Queries(Protocol):
    async def fields_by_id(
        self, scope: Tuple4, field_ids: list[str], *, whole_datasets: bool = False
    ) -> list[dict[str, Any]]: ...

    async def field_availability(self, field_id: str) -> list[dict[str, Any]]: ...

    async def synced_tuples(self) -> list[dict[str, Any]]: ...


def _check(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _depth(node: fastexpr.Node) -> int:
    kids = fastexpr.children(node)
    return 1 + max((_depth(k) for k in kids), default=0)


def scope_of(scope: dict[str, Any]) -> Tuple4:
    return Tuple4(
        instrument_type=scope["instrumentType"],
        region=scope["region"],
        delay=scope["delay"],
        universe=scope["universe"],
    )


def check_settings(
    settings: dict[str, Any] | None, scope: dict[str, Any], schema: dict[str, Any] | None
) -> list[dict[str, str]]:
    """Settings against BRAIN's own list of choices and against the scope they must keep."""
    checks: list[dict[str, str]] = []
    if not settings:
        return [_check("settings_missing", "block", "No simulation settings were given.")]
    checks.extend(
        _check(
            "scope_mismatch",
            "block",
            f"{key} is {settings.get(key)!r} in the settings but {scope.get(key)!r} in "
            "the chosen scope. The scope is never changed to match; fix the settings.",
        )
        for key in ("instrumentType", "region", "delay", "universe")
        if settings.get(key) != scope.get(key)
    )
    checks.extend(
        _check("setting_missing", "block", f"{key} is not set.")
        for key in REQUIRED_SETTINGS
        if settings.get(key) in (None, "")
    )
    checks.extend(
        _check(
            "setting_unconfirmed",
            "block",
            f"{key} has not been chosen. It is never filled in for you.",
        )
        for key in CONFIRMED_SETTINGS
        if settings.get(key) is None
    )
    if settings.get("language") not in (None, LANGUAGE):
        checks.append(
            _check(
                "unsupported_language",
                "block",
                f"Language {settings.get('language')!r} is not supported here; "
                f"Alpha Builder writes {LANGUAGE} only.",
            )
        )
    decay = settings.get("decay")
    if decay is not None and (
        not isinstance(decay, int) or isinstance(decay, bool) or not 0 <= decay <= MAX_DECAY
    ):
        checks.append(
            _check("bad_decay", "block", f"Decay must be a whole number from 0 to {MAX_DECAY}.")
        )
    truncation = settings.get("truncation")
    if truncation is not None and (
        not isinstance(truncation, int | float)
        or isinstance(truncation, bool)
        or not 0 <= truncation <= 1
    ):
        checks.append(_check("bad_truncation", "block", "Truncation must be a number from 0 to 1."))
    if settings.get("maxTrade") == "ON" and settings.get("maxPosition") == "ON":
        checks.append(
            _check("pair_conflict", "block", "Max Trade and Max Position cannot both be On.")
        )
    if not schema:
        checks.append(
            _check(
                "no_settings_metadata",
                "block",
                "BRAIN's list of settings choices is not loaded, so the settings cannot be "
                "checked. Sign in again to load it.",
            )
        )
    else:
        checks.extend(
            _check("setting_not_offered", "block", problem)
            for problem in validate_settings(schema, settings)
        )
    return checks


async def validate(
    *,
    expression: str | None,
    settings: dict[str, Any] | None,
    scope: dict[str, Any],
    operators: list[dict[str, Any]] | None,
    schema: dict[str, Any] | None,
    queries: Queries,
    lookahead_concerns: list[Any] | None = None,
) -> dict[str, Any]:
    """Every static check, as a result the screens and the task preview both read."""
    checks = check_settings(settings, scope, schema)
    tuple4 = scope_of(scope)
    detail: dict[str, Any] = {
        "outcome": BLOCKED,
        "checks": checks,
        "fieldsUsed": [],
        "operatorsUsed": [],
        "groupingUsed": [],
        "complexity": None,
        "notChecked": list(NOT_CHECKED),
        "validatedAt": datetime.now(UTC).isoformat(),
        "scope": scope,
    }

    held = {
        (r["instrument_type"], r["region"], int(r["delay"]), r["universe"])
        for r in await queries.synced_tuples()
    }
    if (tuple4.instrument_type, tuple4.region, tuple4.delay, tuple4.universe) not in held:
        checks.append(
            _check(
                "scope_not_synced",
                "block",
                f"The field catalogue holds no data for {tuple4.label}. Sync that market in the "
                "Data Explorer; the scope is not changed to one that is synced.",
            )
        )
    if not operators:
        checks.append(
            _check(
                "operators_unread",
                "block",
                "Your BRAIN operators are not loaded, so operators cannot be checked. "
                "Sign in again, then reload.",
            )
        )

    text = (expression or "").strip()
    if not text:
        checks.append(_check("no_expression", "block", "There is no expression."))
        return _finish(detail)
    try:
        tree = fastexpr.parse(text)
    except fastexpr.ParseError as exc:
        checks.append(
            _check("syntax", "block", f"Not a Fast Expression this reader can parse: {exc}")
        )
        return _finish(detail)

    nodes = fastexpr.walk_with_parents(tree)
    for _, node, _ in nodes:
        if node.kind == "name" and (node.value.startswith("$") or node.value == "..."):
            checks.append(
                _check(
                    "placeholder",
                    "block",
                    f"{node.value!r} is a template placeholder, not BRAIN syntax.",
                )
            )

    data = fastexpr.data_fields(tree)
    grouping = [g for g in fastexpr.data_fields(tree, grouping=True) if g in GROUPING_FIELDS]
    detail["fieldsUsed"], detail["groupingUsed"] = data, grouping
    detail["operatorsUsed"] = list(dict.fromkeys(n.value for _, n, _ in nodes if n.kind == "call"))
    if not data:
        checks.append(_check("no_data_field", "block", "The expression reads no data field."))

    found = {
        str(r["field_id"]): r
        for r in (
            await queries.fields_by_id(tuple4, [*data, *grouping]) if (data or grouping) else []
        )
    }
    table = fastexpr.operator_table(operators or [], "REGULAR") if operators else {}
    if table:
        known = {*found, *grouping}
        checks.extend(_check("rejected", "block", p) for p in fastexpr.validate(tree, table, known))
    for name in data:
        if name not in found:
            elsewhere = await queries.field_availability(name)
            places = sorted(f"{p['region']}/D{p['delay']}/{p['universe']}" for p in elsewhere)
            where = (
                f" It exists in {', '.join(sorted(set(places))[:4])}"
                f" but not in {tuple4.label}; nothing was substituted."
                if elsewhere
                else ""
            )
            checks.append(
                _check(
                    "field_not_in_scope",
                    "block",
                    f"{name} is not a data field in {tuple4.label}.{where}",
                )
            )
    for name in grouping:
        if name not in found:
            checks.append(
                _check(
                    "grouping_not_in_catalogue",
                    "review",
                    f"{name} is a grouping field but the catalogue lists no such field in "
                    f"{tuple4.label}; some markets lack some groupings.",
                )
            )

    _types(nodes, found, table, checks)
    _lookahead(nodes, found, checks, lookahead_concerns or [])

    operators_count = fastexpr.operator_count(tree)
    depth = _depth(tree)
    detail["complexity"] = {"operators": operators_count, "depth": depth, "fields": len(data)}
    if operators_count > MAX_OPERATORS or depth > MAX_DEPTH or len(data) > MAX_FIELDS:
        checks.append(
            _check(
                "complexity",
                "review",
                f"{operators_count} operators, depth {depth}, {len(data)} fields. More than "
                f"{MAX_OPERATORS} operators, depth {MAX_DEPTH} or {MAX_FIELDS} fields is flagged "
                "by this application (not by BRAIN): more parts mean more room to overfit.",
            )
        )
    return _finish(detail)


def _types(
    nodes: list[tuple[fastexpr.Path, fastexpr.Node, fastexpr.Node | None]],
    found: dict[str, dict[str, Any]],
    table: dict[str, fastexpr.OperatorInfo],
    checks: list[dict[str, str]],
) -> None:
    """Vector and group fields used where only a matrix field belongs."""
    for path, node, parent in nodes:
        if node.kind != "name" or node.value not in found:
            continue
        kind = str(found[node.value].get("field_type") or "").upper()
        keyword = parent is not None and path[-1] >= len(parent.args)
        if keyword:
            continue
        reads_vector = (
            parent is not None
            and parent.kind == "call"
            and (info := table.get(parent.value)) is not None
            and info.category.lower() == "vector"
        )
        if kind == "VECTOR" and not reads_vector:
            checks.append(
                _check(
                    "vector_misuse",
                    "block",
                    f"{node.value} is a VECTOR field. A vector operator (such as vec_avg or "
                    "vec_sum) has to reduce it to a matrix before anything else reads it.",
                )
            )
        elif kind == "MATRIX" and reads_vector:
            checks.append(
                _check(
                    "matrix_in_vector_operator",
                    "review",
                    f"{node.value} is a MATRIX field passed to the vector operator "
                    f"{parent.value if parent else ''}; BRAIN may reject it.",
                )
            )
        elif kind == "GROUP" and not fastexpr.guarded(parent, path):
            checks.append(
                _check(
                    "group_misuse",
                    "block",
                    f"{node.value} is a GROUP field and belongs in the group argument of a "
                    "group operator, not as data.",
                )
            )


def _lookahead(
    nodes: list[tuple[fastexpr.Path, fastexpr.Node, fastexpr.Node | None]],
    found: dict[str, dict[str, Any]],
    checks: list[dict[str, str]],
    concerns: list[Any],
) -> None:
    for _, node, _ in nodes:
        if node.kind != "call" or not node.value.startswith(("ts_", "delay")):
            continue
        checks.extend(
            _check(
                "negative_lookback",
                "block",
                f"{node.value} is given a negative window; that reads the future.",
            )
            for arg in node.args[1:]
            if arg.kind == "unary" and arg.value == "-" and arg.args[0].kind == "num"
        )
    for name, row in found.items():
        if pit_status(str(row.get("description") or "")) == "revised":
            checks.append(
                _check(
                    "revised_data",
                    "review",
                    f"{name} is described as restated or latest values, which can look ahead.",
                )
            )
    if concerns:
        checks.append(
            _check(
                "idea_lookahead",
                "review",
                "The source idea lists timing concerns: "
                + "; ".join(str(c)[:140] for c in concerns[:3])
                + ". Check them against this expression.",
            )
        )


def _finish(detail: dict[str, Any]) -> dict[str, Any]:
    severities = {c["severity"] for c in detail["checks"]}
    detail["outcome"] = (
        BLOCKED if "block" in severities else REVIEW if "review" in severities else VALID
    )
    return detail
