"""Data Field Matcher: conceptual variables to real BRAIN fields, in one exact scope.

Rules this module holds to:

* A field id is only ever read from the local catalogue, in the exact instrument / region /
  delay / universe the person chose. The model's names for data are search terms, never ids.
* A field found only in another scope is reported as such, with where, and never used.
* Nothing is called an *exact* match on meaning: ``exact`` here means the catalogue's own
  wording covers every word of the variable's name. Everything else is a ``proxy``, and a
  proxy is not usable until the person accepts it.
* A person's substitution must name a field that exists in the scope; its type is shown, and a
  VECTOR or GROUP field is never presented as an ordinary data field.
"""

import re
from typing import Any, Protocol

from ..brain.platform import GROUPING_FIELDS
from ..catalog.queries import FieldFilter, Tuple4
from ..research.feasibility import UNSUPPORTED_KINDS, pit_status

CANDIDATES = 6
MAX_TERMS = 4
_STOP = frozenset({"of", "the", "and", "for", "per", "to", "in", "on", "a", "an", "from", "by"})

EXACT = "exact"
PROXY = "proxy"
MANUAL = "manual"
MISSING = "missing"
OTHER_SCOPE = "other_scope"
UNSUPPORTED = "unsupported"
#: Statuses a candidate may be written from, once any needed confirmation is given.
USABLE = {EXACT, PROXY, MANUAL}


class Queries(Protocol):
    async def fields(
        self, scope: Tuple4, filters: FieldFilter, *, counted: bool = True
    ) -> dict[str, Any]: ...

    async def fields_by_id(
        self, scope: Tuple4, field_ids: list[str], *, whole_datasets: bool = False
    ) -> list[dict[str, Any]]: ...

    async def fields_anywhere(self, search: str, limit: int) -> list[dict[str, Any]]: ...

    async def field_availability(self, field_id: str) -> list[dict[str, Any]]: ...


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def view(row: dict[str, Any]) -> dict[str, Any]:
    description = str(row.get("description") or "")
    return {
        "fieldId": str(row["field_id"]),
        "datasetId": row.get("dataset_id"),
        "datasetName": row.get("dataset_name"),
        "description": description[:300],
        "fieldType": str(row.get("field_type") or "").upper() or None,
        "coverage": row.get("coverage"),
        "category": row.get("category_name"),
        "pointInTime": pit_status(description),
    }


def covers(variable_name: str, row: dict[str, Any]) -> bool:
    """Whether the field's id and description hold every word of the variable's name."""
    wanted = _words(variable_name)
    haystack = f"{row.get('field_id', '')} {row.get('description', '')}".lower().replace("_", " ")
    return bool(wanted) and all(w in haystack for w in wanted)


def _fits_type(expected: str, field_type: str | None) -> bool:
    return expected == "ANY" or field_type is None or field_type == expected


async def match(
    queries: Queries, scope: Tuple4, variables: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One entry per variable: what was found in ``scope``, how well, and why."""
    return [await _one(queries, scope, v) for v in variables]


async def _one(queries: Queries, scope: Tuple4, variable: dict[str, Any]) -> dict[str, Any]:
    name = str(variable["name"])
    kind = str(variable.get("kind", "other"))
    expected = str(variable.get("expected_type", "ANY"))
    entry: dict[str, Any] = {
        "name": name,
        "meaning": variable.get("meaning", ""),
        "kind": kind,
        "essential": bool(variable.get("essential", True)),
        "expectedType": expected,
        "sourceBacked": bool(variable.get("source_backed", True)),
        "searchTerms": [str(t) for t in variable.get("search_terms", [])][:MAX_TERMS],
        "status": MISSING,
        "selected": None,
        "candidates": [],
        "availableElsewhere": [],
        "rationale": "",
        "approximation": None,
        "needsConfirmation": False,
        "typeNote": None,
    }
    if kind in UNSUPPORTED_KINDS:
        entry["status"] = UNSUPPORTED
        entry["rationale"] = (
            f"'{name}' needs {UNSUPPORTED_KINDS[kind]}, which BRAIN's equity field catalogue "
            "does not carry. Nothing was substituted."
        )
        return entry

    found: dict[str, dict[str, Any]] = {}
    for term in entry["searchTerms"]:
        page = await queries.fields(
            scope, FieldFilter(search=term, limit=CANDIDATES, sort_by="alpha_count"), counted=False
        )
        for row in page["results"]:
            found.setdefault(str(row["field_id"]), row)
    rows = list(found.values())
    # Whole-name matches first, then type-compatible ones, then the catalogue's own order.
    rows.sort(
        key=lambda r: (
            not covers(name, r),
            not _fits_type(expected, str(r.get("field_type") or "").upper() or None),
        )
    )
    entry["candidates"] = [view(r) for r in rows[:CANDIDATES]]

    if not rows:
        elsewhere: list[dict[str, Any]] = []
        for term in entry["searchTerms"]:
            for row in await queries.fields_anywhere(term, 3):
                elsewhere.extend(
                    {
                        "fieldId": str(row["field_id"]),
                        **{k: p[k] for k in ("region", "delay", "universe")},
                    }
                    for p in await queries.field_availability(str(row["field_id"]))
                )
            if elsewhere:
                break
        entry["availableElsewhere"] = elsewhere[:12]
        if elsewhere:
            entry["status"] = OTHER_SCOPE
            entry["rationale"] = (
                f"Fields for '{name}' exist in other scopes but none in {scope.label}. "
                "Nothing was substituted; choose another scope or a field yourself."
            )
        else:
            entry["rationale"] = (
                f"No field in the catalogue matches '{name}' "
                f"(searched: {', '.join(entry['searchTerms']) or 'nothing'})."
            )
        return entry

    best = next(
        (r for r in rows if _fits_type(expected, str(r.get("field_type") or "").upper() or None)),
        rows[0],
    )
    best_type = str(best.get("field_type") or "").upper() or None
    entry["selected"] = view(best)
    if not _fits_type(expected, best_type):
        entry["typeNote"] = (
            f"The design expects a {expected} field; {best['field_id']} is {best_type}."
        )
    if covers(name, best) and entry["typeNote"] is None:
        entry["status"] = EXACT
        entry["rationale"] = (
            f"{best['field_id']}'s catalogue wording covers every word of '{name}'. "
            "That is a wording match: confirm it means what the idea needs."
        )
    else:
        entry["status"] = PROXY
        entry["needsConfirmation"] = True
        entry["approximation"] = (
            f"{best['field_id']} is the closest field the catalogue offers for '{name}', "
            "not a field described as it."
        )
        entry["rationale"] = (
            "No field's wording covers the whole concept; this one is a proxy and is not used "
            "until you accept it."
        )
    return entry


async def verify_field(queries: Queries, scope: Tuple4, field_id: str) -> dict[str, Any] | None:
    """The field as this scope's catalogue holds it, or ``None`` when it is not here."""
    rows = await queries.fields_by_id(scope, [field_id])
    return view(rows[0]) if rows else None


async def resolve(
    queries: Queries,
    scope: Tuple4,
    mapping: list[dict[str, Any]],
    substitutions: dict[str, str],
    accepted: list[str],
) -> list[dict[str, Any]]:
    """The mapping as it stands once the person's substitutions and accepted proxies apply.

    Each entry gains ``final`` (the field that would be used, or ``None``), ``usable`` and
    ``blockedReason``. A substitution that is not in this scope's catalogue is refused, never
    kept.
    """
    out: list[dict[str, Any]] = []
    for item in mapping:
        entry = dict(item)
        name = entry["name"]
        entry["final"], entry["usable"], entry["blockedReason"] = None, False, None
        if name in substitutions:
            field = await verify_field(queries, scope, substitutions[name])
            if field is None:
                entry["blockedReason"] = (
                    f"{substitutions[name]} is not a field in {scope.label}; "
                    "the substitution cannot be used."
                )
            else:
                entry["status"] = MANUAL
                entry["final"], entry["usable"] = field, True
                entry["approximation"] = (
                    f"You chose {field['fieldId']} for '{name}'. It is your substitution, not "
                    "the matcher's finding."
                )
                entry["needsConfirmation"] = False
                if not _fits_type(entry["expectedType"], field["fieldType"]):
                    entry["typeNote"] = (
                        f"The design expects {entry['expectedType']}; this field is "
                        f"{field['fieldType']}."
                    )
        elif entry["status"] == EXACT and entry["selected"]:
            entry["final"], entry["usable"] = entry["selected"], True
        elif entry["status"] == PROXY and entry["selected"]:
            if name in accepted:
                entry["final"], entry["usable"] = entry["selected"], True
            else:
                entry["blockedReason"] = (
                    f"{entry['selected']['fieldId']} is a proxy for '{name}'. "
                    "Accept the proxy or choose a field."
                )
        else:
            entry["blockedReason"] = entry["rationale"]
        out.append(entry)
    return out


def usable_fields(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the model may use: real ids with their types, descriptions and what they stand for."""
    return [
        {
            "forVariable": e["name"],
            "fieldId": e["final"]["fieldId"],
            "fieldType": e["final"]["fieldType"],
            "description": e["final"]["description"],
            "dataset": e["final"].get("datasetName") or e["final"].get("datasetId"),
            "isProxy": e["status"] in {PROXY, MANUAL},
        }
        for e in resolved
        if e["usable"] and e["final"]
    ]


def grouping_field(name: str) -> bool:
    return name in GROUPING_FIELDS
