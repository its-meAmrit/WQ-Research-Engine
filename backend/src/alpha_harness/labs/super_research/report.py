"""Reads a research task's trials into candidates, scores, pools and a dashboard.

Used by both the refill loop (to learn) and the API (to show), so the screen and the planner
always agree about what happened.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...db.models import TrialState
from ..fastexpr import parse
from ..objectives import STRUCTURAL_CHECKS
from ..study import submittable as study_submittable
from . import scoring
from .canon import canonical
from .coverage import Used
from .planner import Done

if TYPE_CHECKING:
    from ..params import SuperResearchParams

PARKED = "Waiting for cores."


@dataclass(slots=True)
class Row:
    """The columns of a trial this module reads."""

    id: int
    number: int
    state: str
    params: dict[str, Any]
    settings: dict[str, Any]
    result: dict[str, Any] | None
    message: str | None
    generation: int | None
    alpha_id: str | None = None


def _selection_key(params: dict[str, Any]) -> str:
    try:
        return canonical(parse(str(params.get("selection", ""))))
    except Exception:  # noqa: BLE001 - a malformed stored expression just groups alone
        return str(params.get("selection", ""))


def outcome_of(row: Row) -> str:
    if row.state == TrialState.COMPLETE:
        return "accepted"
    if row.state == TrialState.FAIL:
        return "rejected"
    if row.state in (TrialState.QUEUED, TrialState.RUNNING):
        return "submitted"
    return "waiting"


def failed_structural(result: dict[str, Any] | None) -> int:
    failed = (result or {}).get("failedChecks") or []
    return len(STRUCTURAL_CHECKS.intersection(str(c) for c in failed))


def evaluate(rows: list[Row], run: SuperResearchParams) -> list[dict[str, Any]]:
    """Every candidate, in the order written, with its score and what it was scored against."""
    baseline: dict[str, float] = {}
    for row in rows:
        if row.params.get("baseline") and row.state == TrialState.COMPLETE and row.result:
            sharpe = (row.result.get("stats") or {}).get("sharpe")
            if isinstance(sharpe, int | float):
                key = _selection_key(row.params)
                baseline[key] = max(baseline.get(key, float("-inf")), float(sharpe))
    out: list[dict[str, Any]] = []
    for row in rows:
        p = row.params
        result = row.result or {}
        stats = result.get("stats") or {}
        graded = row.state == TrialState.COMPLETE and bool(result)
        value, parts = (
            scoring.score(
                result,
                objective=run.objective,
                baseline_sharpe=None if p.get("baseline") else baseline.get(_selection_key(p)),
                failed_checks=failed_structural(result),
            )
            if graded
            else (None, {})
        )
        out.append(
            {
                "trialId": row.id,
                "number": row.number,
                "generation": row.generation,
                "state": str(row.state),
                "outcome": outcome_of(row),
                "name": p.get("recipe"),
                "family": p.get("family"),
                "origin": p.get("origin"),
                "explanation": p.get("explanation"),
                "selection": p.get("selection"),
                "combo": p.get("combo"),
                "operators": p.get("operators") or [],
                "parents": p.get("parents") or [],
                "baseline": bool(p.get("baseline")),
                "key": p.get("key"),
                "settings": row.settings,
                "local": p.get("local") or {},
                "alphaId": row.alpha_id,
                "score": value,
                "scoreParts": parts,
                "sharpe": stats.get("sharpe"),
                "fitness": stats.get("fitness"),
                "turnover": stats.get("turnover"),
                "drawdown": stats.get("drawdown"),
                "returns": stats.get("returns"),
                "margin": stats.get("margin"),
                "turnoverInRange": scoring.turnover_in_range(result) if graded else None,
                "failedChecks": result.get("failedChecks") or [],
                "vsEqualWeight": parts.get("vs_equal_weight"),
                # Only BRAIN's own checks may call it submittable, and only a finished Full run.
                "submittable": bool(graded and study_submittable(result)),
                "rejection": row.message if row.state == TrialState.FAIL else None,
                "rejectionKind": scoring.categorize_failure(row.message)
                if row.state == TrialState.FAIL
                else None,
            }
        )
    return out


def history_of(candidates: list[dict[str, Any]]) -> list[Done]:
    return [
        Done(
            key=str(c["key"]),
            selection=str(c["selection"]),
            combo=str(c["combo"]),
            family=str(c["family"] or "unknown"),
            score=c["score"],
            settings=(
                int(c["settings"].get("decay", 0)),
                float(c["settings"].get("truncation", 0)),
                str(c["settings"].get("componentActivation", "")),
            ),
            state="COMPLETE" if c["score"] is not None else str(c["state"]),
        )
        for c in candidates
        if c.get("key")
    ]


def used_of(candidates: list[dict[str, Any]]) -> list[Used]:
    return [
        Used(
            operators=list(c["operators"]),
            outcome=c["outcome"],
            score=c["score"],
            failure=c["rejection"],
        )
        for c in candidates
        if c["outcome"] != "waiting" or c["operators"]
    ]


def best_by_generation(candidates: list[dict[str, Any]]) -> dict[int, float]:
    best: dict[int, float] = {}
    for c in candidates:
        if c["score"] is None or c["generation"] is None:
            continue
        g = int(c["generation"])
        best[g] = max(best.get(g, float("-inf")), float(c["score"]))
    return best


def stalled(candidates: list[dict[str, Any]], patience: int) -> bool:
    """No generation of the last ``patience`` beat everything before it."""
    best = best_by_generation(candidates)
    if len(best) <= patience:
        return False
    ordered = sorted(best)
    recent, earlier = ordered[-patience:], ordered[:-patience]
    return max(best[g] for g in recent) <= max(best[g] for g in earlier)


def diversity(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    done = [c for c in candidates if c["outcome"] != "waiting"]
    families = Counter(str(c["family"]) for c in done)
    combos = {canonical_safe(str(c["combo"])) for c in done}
    selections = {canonical_safe(str(c["selection"])) for c in done}
    total = max(len(done), 1)
    return {
        "families": dict(families),
        "distinctCombos": len(combos),
        "distinctSelections": len(selections),
        # 1.0 when every candidate is its own combo; falls as combos repeat.
        "comboUniqueness": round(len(combos) / total, 3),
    }


def canonical_safe(text: str) -> str:
    try:
        return canonical(parse(text))
    except Exception:  # noqa: BLE001
        return text


def dashboard(candidates: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    counts = Counter(c["outcome"] for c in candidates)
    scored = sorted(
        (c for c in candidates if c["score"] is not None), key=lambda c: -float(c["score"])
    )
    by_gen: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"candidates": 0, "scored": 0, "best": None}
    )
    for c in candidates:
        if c["generation"] is None:
            continue
        g = by_gen[int(c["generation"])]
        g["candidates"] += 1
        if c["score"] is not None:
            g["scored"] += 1
            g["best"] = c["score"] if g["best"] is None else max(g["best"], c["score"])
    spent = counts["accepted"] + counts["rejected"] + counts["submitted"]
    failures = Counter(c["rejectionKind"] for c in candidates if c["rejectionKind"])
    return {
        "generated": len(candidates),
        "locallyValid": len(candidates),  # only candidates that passed the checker are written
        "simulated": counts["accepted"],
        "submitted": counts["submitted"],
        "failed": counts["rejected"],
        "waiting": counts["waiting"],
        "budget": budget,
        "budgetUsed": spent,
        "submittable": sum(1 for c in candidates if c["submittable"]),
        "best": scored[:10],
        "history": [{"generation": g, **v} for g, v in sorted(by_gen.items())],
        "diversity": diversity(candidates),
        "failureKinds": dict(failures),
    }
