"""Five separate indicators of an idea, each with the reasons for it. Never one score.

They answer five different questions and a total would only hide which one is weak: *where it
came from* (source quality), *how measurable it is* (specificity), *whether BRAIN can express
it* (feasibility), *whether it is new to this library* (novelty) and *how well the page backs
it* (evidence quality). Nothing here estimates a Sharpe ratio or a return, and a quoted figure
is never an input. When a question cannot be answered from what is held, the answer is
``unknown``, not a guess.

Novelty compares an idea with the other ideas in this library by their wording. It says
nothing about correlation with any alpha you hold.
"""

from typing import Any

from . import dedupe

LEVELS = ("high", "medium", "low", "unknown")
_NOT_STATED = "not stated"


def _stated(text: Any) -> bool:
    value = str(text or "").strip().lower()
    return bool(value) and not value.startswith(_NOT_STATED) and value != "n/a"


def _indicator(level: str, reasons: list[str], basis: str) -> dict[str, Any]:
    return {"level": level, "reasons": reasons, "basis": basis}


def _level(points: int, high: int, medium: int) -> str:
    return "high" if points >= high else "medium" if points >= medium else "low"


def source_quality(row: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    basis = "Who wrote it, when, where, and whether the method is written down."
    if source is None:
        return _indicator("unknown", ["The source is not held."], basis)
    reasons: list[str] = []
    points = 0
    kind = source.get("sourceClass") or "research_paper"
    if source.get("authors"):
        points += 1
        reasons.append("Named author(s).")
    else:
        reasons.append("No author is given.")
    if source.get("published"):
        points += 1
        reasons.append("Publication date given.")
    else:
        reasons.append("No publication date.")
    if source.get("doi") or source.get("venue"):
        points += 1
        reasons.append("A scholarly record (DOI or venue).")
    elif source.get("registryTier") == "practitioner":
        reasons.append("Practitioner publication: not peer reviewed.")
    elif source.get("registryTier") == "preprint_server":
        reasons.append("Preprint: not peer reviewed.")
    if (row.get("claims") or {}).get("methodologyDocumented") or (
        kind == "research_paper" and _stated(row.get("methodology"))
    ):
        points += 1
        reasons.append("The method is described.")
    else:
        reasons.append("The method is not described in enough detail.")
    level = _level(points, 3, 2)
    if kind in {"unverified_claim", "not_research"}:
        level = "low"
        reasons.append("The document is classed as a claim, not research.")
    return _indicator(level, reasons, basis)


def specificity(row: dict[str, Any]) -> dict[str, Any]:
    basis = "Whether the hypothesis is stated as inputs a test could measure."
    reasons: list[str] = []
    points = 0
    variables = row.get("required_variables") or []
    essential = [v for v in variables if v.get("essential")]
    if essential and all(v.get("search_terms") for v in essential):
        points += 1
        reasons.append(f"{len(essential)} essential input(s), each searchable by name.")
    else:
        reasons.append("No clearly measurable input.")
    for label, field in (
        ("The signal is defined", "signal_definition"),
        ("The lookback is given", "lookback"),
        ("The holding or rebalancing is given", "holding_period"),
        ("The universe is given", "universe"),
    ):
        if _stated(row.get(field)):
            points += 1
            reasons.append(f"{label}.")
        else:
            reasons.append(f"{label.replace(' is ', ' is not ')}.")
    if row.get("signal_direction") in {"long_high", "long_low", "long_short_by_sign"}:
        points += 1
        reasons.append("The direction of the relationship is stated.")
    else:
        reasons.append("The direction is unclear.")
    return _indicator(_level(points, 5, 3), reasons, basis)


def feasibility(assessment: dict[str, Any] | None) -> dict[str, Any]:
    basis = "Whether the data and operators exist in the scope you assessed it against."
    if not assessment:
        return _indicator(
            "unknown",
            ["Not assessed yet. Choose a scope and assess the idea."],
            basis,
        )
    readiness = assessment.get("readiness")
    reasons = [str(assessment.get("summary") or f"Readiness: {readiness}.")]
    level = {"ready": "high", "needs_adaptation": "medium", "blocked": "low"}.get(
        str(readiness), "unknown"
    )
    return _indicator(level, reasons, basis)


def novelty(row: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any]:
    basis = (
        "Wording overlap with the other ideas in this library. Not a measure of correlation "
        "with any alpha you hold."
    )
    if not others:
        return _indicator("unknown", ["There are no other ideas to compare with yet."], basis)
    best, nearest = 0.0, None
    for other in others:
        score = dedupe.similarity(row, other)
        if score > best:
            best, nearest = score, other
    if best >= dedupe.POSSIBLE:
        level = "low"
    elif best >= dedupe.RELATED:
        level = "medium"
    else:
        level = "high"
    reasons = [f"Compared with {len(others)} idea(s) in the library."]
    if nearest is not None and best >= dedupe.RELATED:
        reasons.append(f"Closest: “{nearest.get('title')}” (overlap {best:.2f}).")
    else:
        reasons.append("No idea in the library says much the same thing.")
    return _indicator(level, reasons, basis)


def evidence_quality(row: dict[str, Any]) -> dict[str, Any]:
    basis = (
        "Whether the quoted passages are in the text, and the construction and limits are given."
    )
    evidence = row.get("evidence") or []
    verified = [e for e in evidence if e.get("verified")]
    reasons: list[str] = []
    points = 0
    if evidence and len(verified) == len(evidence):
        points += 2
        reasons.append(f"All {len(verified)} cited passage(s) were found in the text.")
    elif verified:
        points += 1
        reasons.append(f"{len(verified)} of {len(evidence)} cited passages were found in the text.")
    else:
        reasons.append("No cited passage could be found in the text.")
    if _stated(row.get("signal_definition")):
        points += 1
        reasons.append("The construction of the signal is described.")
    if row.get("limitations"):
        points += 1
        reasons.append("Limitations are stated.")
    else:
        reasons.append("No limitations are stated.")
    if (row.get("claims") or {}).get("reportedPerformance"):
        reasons.append("Quoted performance is the source's claim and is not counted as evidence.")
    if not verified:
        # Nothing checkable in the text: a well-described construction does not make it evidence.
        points = min(points, 1)
    if row.get("evidence_scope") == "abstract":
        reasons.append("Read from an abstract only.")
        points = min(points, 2)
    return _indicator(_level(points, 4, 2), reasons, basis)


def rank(
    row: dict[str, Any],
    *,
    source: dict[str, Any] | None,
    assessment: dict[str, Any] | None,
    others: list[dict[str, Any]],
) -> dict[str, Any]:
    """All five indicators for an idea, side by side."""
    return {
        "sourceQuality": source_quality(row, source),
        "specificity": specificity(row),
        "brainFeasibility": feasibility(assessment),
        "novelty": novelty(row, others),
        "evidenceQuality": evidence_quality(row),
    }
