"""Whether an idea can be drafted as an alpha in the scope the person chose, and why.

Three answers, and the reasons behind each, always shown:

``ready``       every essential input has candidate fields in this exact scope, every named
                operator exists, and nothing is known to look ahead.
``adaptation``  a reasonable route exists but a person must decide something: a proxy for a
                missing input, a different operator, a scope the paper did not study, a
                data series whose revision history is unknown.
``blocked``     a core input cannot be had here at all, or the method needs information that
                was not available at the time (look-ahead), or the scope holds no data.

Rules this module holds to:

* Field ids are only ever **read from the local catalogue**. The model's names for data are
  search terms, never ids. A candidate is a suggestion to choose from, not a decision.
* The scope is the one passed in. A field found only in another scope is reported as
  *available elsewhere* with where, and never swapped in.
* A candidate's point-in-time status is read from the catalogue's own description. Where the
  description does not say, it is reported as unknown, not assumed safe.
* An operator is checked against the account's cached operator list, by exact name. With no
  list loaded, operators are reported unverified and the idea cannot be ``ready``.
* An idea is kept whatever the answer. ``blocked`` marks what cannot be done now.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..catalog.queries import FieldFilter, Tuple4
from ..labs.fastexpr import operator_table

READY = "ready"
ADAPTATION = "adaptation"
BLOCKED = "blocked"

CANDIDATES = 5
MAX_SEARCH_TERMS = 3

#: What a variable's kind needs from the platform. These data BRAIN's equity catalogue
#: does not carry as fields at all, so an essential one blocks the idea.
UNSUPPORTED_KINDS = {
    "intraday": "intraday or tick data",
    "options": "options-market data",
    "order_book": "order-book data",
}

#: How a region is named in papers, to the BRAIN region codes. Used only to *warn* about a
#: mismatch with the selected scope. A person decides; nothing is changed.
REGION_WORDS: dict[str, str] = {
    "united states": "USA",
    "u.s.": "USA",
    "us stocks": "USA",
    "nyse": "USA",
    "nasdaq": "USA",
    "amex": "USA",
    "crsp": "USA",
    "s&p 500": "USA",
    "russell": "USA",
    "europe": "EUR",
    "european": "EUR",
    "uk ": "EUR",
    "united kingdom": "EUR",
    "germany": "EUR",
    "france": "EUR",
    "japan": "JPN",
    "china": "CHN",
    "chinese": "CHN",
    "korea": "KOR",
    "taiwan": "TWN",
    "india": "IND",
    "hong kong": "HKG",
    "asia": "ASI",
    "global": "GLB",
    "international": "GLB",
    "emerging": "GLB",
    "developed markets": "GLB",
    "48 countries": "GLB",
}

_REVISED = re.compile(
    r"\b(restat\w*|revis\w*|latest|most recent|as of today|back-?fill\w*)\b", re.IGNORECASE
)
_POINT_IN_TIME = re.compile(
    r"\b(point[- ]in[- ]time|as[- ]reported|as originally reported|pit)\b", re.IGNORECASE
)


class Queries(Protocol):
    """What the assessor reads. :class:`~alpha_harness.catalog.queries.CatalogQueries` fits."""

    async def fields(
        self, scope: Tuple4, filters: FieldFilter, *, counted: bool = True
    ) -> dict[str, Any]: ...

    async def fields_anywhere(self, search: str, limit: int) -> list[dict[str, Any]]: ...

    async def field_availability(self, field_id: str) -> list[dict[str, Any]]: ...

    async def synced_tuples(self) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class Reason:
    code: str
    severity: str  # info | adapt | block
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


def pit_status(description: str) -> str:
    """``point_in_time``, ``revised`` or ``unknown``, from a field's own description."""
    if _POINT_IN_TIME.search(description or ""):
        return "point_in_time"
    if _REVISED.search(description or ""):
        return "revised"
    return "unknown"


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    description = str(row.get("description") or "")
    return {
        "fieldId": str(row["field_id"]),
        "description": description[:240],
        "fieldType": row.get("field_type"),
        "datasetId": row.get("dataset_id"),
        "coverage": row.get("coverage"),
        "alphaCount": row.get("alpha_count"),
        "pointInTime": pit_status(description),
    }


def _operators(
    needed: list[str], operators: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[Reason]]:
    """Each operator the idea names, found or not, by exact name."""
    if not needed:
        return [], []
    if not operators:
        return (
            [{"name": n, "status": "unverified"} for n in needed],
            [
                Reason(
                    "operators_unverified",
                    "adapt",
                    "The account's operator list is not loaded, so the operators this idea "
                    "uses could not be checked. Sign in to BRAIN to load them.",
                )
            ],
        )
    table = operator_table(operators)
    results: list[dict[str, Any]] = []
    reasons: list[Reason] = []
    for name in dict.fromkeys(needed):
        info = table.get(name.strip())
        if info is None:
            results.append({"name": name, "status": "unsupported"})
            reasons.append(
                Reason(
                    "unsupported_operator",
                    "adapt",
                    f"'{name}' is not a BRAIN operator on this account. The operation has to "
                    "be built from operators that exist, or replaced.",
                )
            )
        else:
            results.append({"name": name, "status": "supported", "category": info.category})
    return results, reasons


def _markets(markets: list[str], universe: str) -> list[str]:
    text = " ".join([*markets, universe]).lower() + " "
    return sorted({code for word, code in REGION_WORDS.items() if word in text})


async def _variable(
    queries: Queries, scope: Tuple4, variable: dict[str, Any]
) -> tuple[dict[str, Any], list[Reason]]:
    """Look one variable up: here first, then (only to report) anywhere else."""
    name = str(variable["name"])
    kind = str(variable.get("kind", "other"))
    essential = bool(variable.get("essential", True))
    severity = "block" if essential else "adapt"
    reasons: list[Reason] = []
    terms = [str(t) for t in variable.get("search_terms", [])][:MAX_SEARCH_TERMS]
    found: dict[str, dict[str, Any]] = {}
    for term in terms:
        page = await queries.fields(
            scope, FieldFilter(search=term, limit=CANDIDATES, sort_by="alpha_count"), counted=False
        )
        for row in page["results"]:
            found.setdefault(str(row["field_id"]), row)
        if len(found) >= CANDIDATES:
            break
    candidates = [_candidate(r) for r in list(found.values())[:CANDIDATES]]

    entry: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "essential": essential,
        "searchTerms": terms,
        "candidates": candidates,
        "availableElsewhere": [],
        "pointInTimeRisk": variable.get("point_in_time_risk", "none"),
    }

    if kind in UNSUPPORTED_KINDS:
        entry["status"] = "unsupported"
        reasons.append(
            Reason(
                "unsupported_data",
                severity,
                f"'{name}' needs {UNSUPPORTED_KINDS[kind]}, which BRAIN's field catalogue does not "
                + ("provide. The idea cannot be built without it." if essential else "provide."),
            )
        )
        return entry, reasons

    if not candidates:
        elsewhere: list[dict[str, Any]] = []
        for term in terms:
            elsewhere = await queries.fields_anywhere(term, 3)
            if elsewhere:
                break
        places: list[dict[str, Any]] = []
        for row in elsewhere[:2]:
            places.extend(
                {
                    "fieldId": str(row["field_id"]),
                    "region": place["region"],
                    "delay": place["delay"],
                    "universe": place["universe"],
                }
                for place in await queries.field_availability(str(row["field_id"]))
            )
        entry["availableElsewhere"] = places[:12]
        if places:
            entry["status"] = "other_scope"
            reasons.append(
                Reason(
                    "scope_mismatch",
                    "adapt",
                    f"'{name}' matches fields that exist in other scopes "
                    f"({', '.join(sorted({f'{p["region"]}/D{p["delay"]}' for p in places})[:6])}) "
                    "but none in "
                    f"{scope.region}/D{scope.delay}/{scope.universe}. Nothing was substituted; "
                    "pick another scope or a proxy yourself.",
                )
            )
        else:
            entry["status"] = "missing"
            reasons.append(
                Reason(
                    "missing_field",
                    severity,
                    f"No field in the catalogue matches '{name}' "
                    f"(searched: {', '.join(terms) or 'nothing'}). ",
                )
            )
        return entry, reasons

    entry["status"] = "candidates"
    pit_values = {c["pointInTime"] for c in candidates}
    risk = entry["pointInTimeRisk"]
    if risk == "uses_future_information":
        reasons.append(
            Reason(
                "lookahead",
                "block" if essential else "adapt",
                f"'{name}' is used in the paper in a way that was not knowable when the "
                "portfolio was formed. Any data BRAIN holds would be point-in-time, so the "
                "paper's result cannot be reproduced as published.",
            )
        )
    elif risk == "revised_values" and "point_in_time" not in pit_values:
        reasons.append(
            Reason(
                "unsupported_point_in_time",
                "adapt",
                f"The paper used revised or restated values of '{name}'. None of the candidate "
                "fields is described as point-in-time, so results may differ and may look ahead.",
            )
        )
    elif kind in {"fundamental", "analyst", "macro"} and pit_values <= {"unknown"}:
        reasons.append(
            Reason(
                "point_in_time_unknown",
                "adapt",
                f"The catalogue does not say whether the candidate fields for '{name}' are "
                "point-in-time or restated. Check the dataset description before relying on them.",
            )
        )
    elif "revised" in pit_values and "point_in_time" not in pit_values:
        reasons.append(
            Reason(
                "revised_data",
                "adapt",
                f"Candidate fields for '{name}' are described as restated or latest values, "
                "which can look ahead.",
            )
        )
    return entry, reasons


def _verdict(reasons: list[Reason]) -> str:
    if any(r.severity == "block" for r in reasons):
        return BLOCKED
    if any(r.severity == "adapt" for r in reasons):
        return ADAPTATION
    return READY


async def assess(
    queries: Queries,
    operators: list[dict[str, Any]] | None,
    scope: Tuple4,
    idea: dict[str, Any],
    *,
    now: str,
) -> dict[str, Any]:
    """The assessment of one idea row (as :func:`..extraction.idea_row` produces) in ``scope``."""
    reasons: list[Reason] = []
    claims = idea.get("claims") or {}

    synced = await queries.synced_tuples()
    in_scope = any(
        r["instrument_type"] == scope.instrument_type
        and r["region"] == scope.region
        and int(r["delay"]) == scope.delay
        and r["universe"] == scope.universe
        for r in synced
    )
    variables: list[dict[str, Any]] = []
    if not in_scope:
        reasons.append(
            Reason(
                "scope_not_downloaded",
                "block",
                f"No data is downloaded for {scope.region}/D{scope.delay}/{scope.universe}, so "
                "nothing can be checked there. Download that scope in Data Explorer, or "
                "choose another. No other scope was used in its place.",
            )
        )
    else:
        for variable in idea.get("required_variables") or []:
            entry, found = await _variable(queries, scope, variable)
            variables.append(entry)
            reasons.extend(found)

    ops, op_reasons = _operators([str(o) for o in claims.get("operatorsNeeded") or []], operators)
    reasons.extend(op_reasons)

    studied = _markets(list(claims.get("marketsStudied") or []), str(idea.get("universe") or ""))
    if studied and scope.region not in studied and "GLB" not in studied:
        reasons.append(
            Reason(
                "market_differs",
                "adapt",
                f"The paper studies {', '.join(studied)}; you selected {scope.region}. The "
                "result may not carry over. Your selection was kept.",
            )
        )

    reasons.extend(
        Reason("lookahead_concern", "adapt", f"Noted by the analysis: {concern}")
        for concern in idea.get("lookahead_concerns") or []
    )

    if idea.get("formula_needs_review"):
        reasons.append(
            Reason(
                "formula_review",
                "adapt",
                "The paper's formula could not be extracted reliably; read it in the PDF before "
                "writing the expression.",
            )
        )
    if idea.get("evidence_scope") == "abstract":
        reasons.append(
            Reason(
                "abstract_only",
                "adapt",
                "This idea was read from the abstract only; the method, data and formula "
                "are unknown.",
            )
        )
    if idea.get("support") == "speculative":
        reasons.append(
            Reason(
                "speculative",
                "adapt",
                "No cited passage could be verified in the paper, so this idea is a model "
                "suggestion, not something the paper is shown to say.",
            )
        )

    verdict = _verdict(reasons)
    if verdict == READY and not variables:
        # Nothing to check against is not the same as nothing missing.
        verdict = ADAPTATION
        reasons.append(
            Reason(
                "no_variables", "adapt", "The idea named no input data, so none could be checked."
            )
        )
    return {
        "readiness": verdict,
        "scope": {
            "instrumentType": scope.instrument_type,
            "region": scope.region,
            "delay": scope.delay,
            "universe": scope.universe,
        },
        "scopeHasData": in_scope,
        "variables": variables,
        "operators": ops,
        "reasons": [r.to_dict() for r in reasons],
        "marketsStudied": studied,
        "assessedAt": now,
        "note": (
            "Candidate fields are suggestions found in your catalogue, not chosen inputs. "
            "No simulation was run and nothing was submitted."
        ),
    }
