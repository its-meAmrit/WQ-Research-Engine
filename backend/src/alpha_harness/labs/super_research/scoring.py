"""A transparent score for a finished SuperAlpha, and whether it may be called submittable.

The score reads only the statistics BRAIN returns for the candidate's own simulation. It never
looks at out-of-sample figures or at anything chosen after the fact, and it never decides
submittability: that is BRAIN's checks, read by ``study.submittable``.
"""

from dataclasses import dataclass
from typing import Any

#: The documented turnover range for a SuperAlpha, as fractions.
TURNOVER_LOW = 0.02
TURNOVER_HIGH = 0.40
FAILURE_SCORE = -10.0


@dataclass(frozen=True, slots=True)
class Weights:
    """Each term is in "Sharpe points", so the weights read as trade-offs."""

    sharpe: float = 1.0
    fitness: float = 0.5
    #: Per unit of drawdown fraction (0.10 = 10%).
    drawdown: float = 4.0
    #: Per unit of turnover outside the documented range.
    turnover: float = 10.0
    #: Per failed structural check.
    check: float = 1.0
    #: Per point of Sharpe above the equal-weight baseline on the same selection.
    improvement: float = 0.5


OBJECTIVES: dict[str, Weights] = {
    "balanced": Weights(),
    "sharpe": Weights(sharpe=1.0, fitness=0.0, drawdown=0.0, improvement=0.0),
    "fitness": Weights(sharpe=0.0, fitness=1.0, drawdown=0.0, improvement=0.0),
    "robust": Weights(sharpe=1.0, fitness=0.5, drawdown=8.0, turnover=15.0, improvement=1.0),
}


def _num(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    return float(value) if isinstance(value, int | float) else None


def score(
    result: dict[str, Any],
    *,
    objective: str = "balanced",
    baseline_sharpe: float | None = None,
    failed_checks: int = 0,
) -> tuple[float, dict[str, float]]:
    """``(score, parts)``: the parts add up to the score, so a table can say why."""
    w = OBJECTIVES.get(objective, OBJECTIVES["balanced"])
    stats = result.get("stats") or {}
    sharpe, fitness = _num(stats, "sharpe"), _num(stats, "fitness")
    if sharpe is None:
        return FAILURE_SCORE, {"missing": FAILURE_SCORE}
    parts: dict[str, float] = {"sharpe": w.sharpe * sharpe}
    if fitness is not None:
        parts["fitness"] = w.fitness * fitness
    drawdown = _num(stats, "drawdown")
    if drawdown is not None:
        parts["drawdown"] = -w.drawdown * abs(drawdown)
    turnover = _num(stats, "turnover")
    if turnover is not None:
        outside = max(TURNOVER_LOW - turnover, 0.0) + max(turnover - TURNOVER_HIGH, 0.0)
        if outside:
            parts["turnover"] = -w.turnover * outside
    if failed_checks:
        parts["checks"] = -w.check * failed_checks
    if baseline_sharpe is not None:
        parts["vs_equal_weight"] = w.improvement * (sharpe - baseline_sharpe)
    return sum(parts.values()), parts


def turnover_in_range(result: dict[str, Any]) -> bool | None:
    turnover = _num(result.get("stats") or {}, "turnover")
    return None if turnover is None else TURNOVER_LOW <= turnover <= TURNOVER_HIGH


def categorize_failure(message: str | None) -> str:
    """A coarse reason a simulation was refused, for the coverage table and for deciding what
    not to try again."""
    text = (message or "").lower()
    if not text:
        return "unknown"
    rules = (
        ("operator", ("unknown operator", "operator", "undefined function", "not supported")),
        ("syntax", ("syntax", "parse", "unexpected", "invalid expression")),
        ("variable", ("unknown variable", "undefined variable", "not defined", "field")),
        ("scope", ("scope", "not allowed", "only available", "permission")),
        ("limit", ("limit", "too many", "exceed", "quota")),
        ("timeout", ("timeout", "timed out")),
    )
    for name, needles in rules:
        if any(n in text for n in needles):
            return name
    return "other"
