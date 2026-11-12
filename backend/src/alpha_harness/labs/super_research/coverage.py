"""Which operators the research has reached, and why the rest have not.

An operator's row is built from the registry (what is known about it) and the task's trials
(what has been done with it), so it survives a restart and can never disagree with the trials
the screen lists.
"""

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from .checker import check
from .grammar import Controls, Grammar
from .registry import Ctx, Registry, Status
from .scoring import categorize_failure

#: A candidate this good counts as a success for the operators it used.
SUCCESS_SCORE = 1.0


@dataclass(slots=True)
class OperatorCoverage:
    name: str
    contexts: list[str]
    family: str
    description: str
    signature: str | None
    example: str | None
    status: str
    verified_by: str | None
    registered: bool = True
    signature_verified: bool = False
    locally_validated: bool = False
    eligible: bool = False
    reason: str = ""
    included: int = 0
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    successful: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    last_failure: str | None = None
    brain_definition: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Used:
    """The operators of one candidate and what became of it."""

    operators: list[str]
    #: ``"submitted"`` once sent, then ``"accepted"`` (BRAIN simulated it), ``"rejected"``.
    outcome: str
    score: float | None = None
    failure: str | None = None


def probe(registry: Registry, name: str, ctx: Ctx, *, seed: int = 0, tries: int = 80) -> bool:
    """Whether a valid expression containing ``name`` can be built in ``ctx`` right now: the
    local test of "can the generator actually use this", as opposed to "is it listed"."""
    controls = Controls(seed=seed, max_depth=5, max_operators=24, max_variables=4)
    grammar = Grammar(registry, controls, random.Random(f"{seed}:{name}:{ctx.value}"))
    for _ in range(tries):
        grammar.focus = name
        tree = grammar.combo() if ctx == Ctx.COMBO else grammar.selection(266)
        grammar.focus = None
        if tree is None:
            continue
        result = check(tree, ctx, registry, controls.limits())
        if result.ok and name in result.operators:
            return True
    return False


def example_of(registry: Registry, name: str, ctx: Ctx, *, seed: int = 0) -> str | None:
    controls = Controls(seed=seed, max_depth=3, max_operators=10)
    grammar = Grammar(registry, controls, random.Random(f"ex:{name}"))
    spec = registry.get(name)
    if spec is None or spec.params is None:
        return None
    for _ in range(40):
        grammar.focus = name
        tree = grammar.combo() if ctx == Ctx.COMBO else grammar.selection(266)
        grammar.focus = None
        if tree is None:
            continue
        result = check(tree, ctx, registry, controls.limits())
        if result.ok and name in result.operators:
            return result.text
    return None


def build(
    registry: Registry,
    used: list[Used],
    *,
    disabled: frozenset[str] = frozenset(),
    families_off: frozenset[str] = frozenset(),
    probe_operators: bool = True,
    examples: bool = False,
) -> list[OperatorCoverage]:
    tally: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    last: dict[str, str] = {}
    for item in used:
        for op in set(item.operators):
            tally[op]["included"] += 1
            if item.outcome in ("submitted", "accepted", "rejected"):
                tally[op]["submitted"] += 1
            if item.outcome == "accepted":
                tally[op]["accepted"] += 1
                if item.score is not None and item.score >= SUCCESS_SCORE:
                    tally[op]["successful"] += 1
            if item.outcome == "rejected":
                tally[op]["rejected"] += 1
                reason = categorize_failure(item.failure)
                reasons[op][reason] += 1
                last[op] = item.failure or reason
    rows: list[OperatorCoverage] = []
    for name, spec in registry.specs.items():
        ctxs = sorted(spec.contexts, key=lambda c: c.value)
        eligibility = [
            registry.eligibility(name, c, disabled=disabled, families_off=families_off)
            for c in ctxs
        ]
        eligible = any(e.eligible for e in eligibility)
        reason = "" if eligible else next((e.reason for e in eligibility if e.reason), "")
        validated = False
        example = None
        if eligible and probe_operators:
            usable = [c for c, e in zip(ctxs, eligibility, strict=True) if e.eligible]
            validated = any(probe(registry, name, c) for c in usable)
            if not validated:
                eligible = False
                reason = "Eligible on paper, but no valid expression containing it could be built."
        if validated and examples:
            example = next((x for c in ctxs if (x := example_of(registry, name, c))), None)
        counts = tally[name]
        rows.append(
            OperatorCoverage(
                name=name,
                contexts=[c.value for c in ctxs],
                family=spec.family,
                description=spec.description,
                signature=spec.signature,
                example=example,
                status=spec.status.value,
                verified_by=spec.verified_by,
                signature_verified=spec.status == Status.VERIFIED,
                locally_validated=validated,
                eligible=eligible,
                reason=reason,
                included=counts["included"],
                submitted=counts["submitted"],
                accepted=counts["accepted"],
                rejected=counts["rejected"],
                successful=counts["successful"],
                failure_reasons=dict(reasons[name]),
                last_failure=last.get(name),
                brain_definition=spec.brain_definition,
                note=spec.note,
            )
        )
    return rows


def summary(rows: list[OperatorCoverage]) -> dict[str, int]:
    """Counts by validation and generation status: what the screen and the report quote."""
    return {
        "registered": len(rows),
        "signatureVerified": sum(r.signature_verified for r in rows),
        "pendingVerification": sum(r.status == Status.PENDING for r in rows),
        "incompatible": sum(r.status == Status.INCOMPATIBLE for r in rows),
        "locallyValidated": sum(r.locally_validated for r in rows),
        "eligible": sum(r.eligible for r in rows),
        "blocked": sum(not r.eligible for r in rows),
        "included": sum(r.included > 0 for r in rows),
        "submitted": sum(r.submitted > 0 for r in rows),
        "accepted": sum(r.accepted > 0 for r in rows),
        "rejected": sum(r.rejected > 0 for r in rows),
        "successful": sum(r.successful > 0 for r in rows),
    }


def rows_as_dicts(rows: list[OperatorCoverage]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]


def avoided(rows: list[OperatorCoverage], *, after: int = 3) -> dict[str, str]:
    """Operators BRAIN has refused ``after`` times and never accepted: left out of new
    candidates, each with the reason."""
    return {
        r.name: f"BRAIN refused {r.rejected} candidates using it and accepted none"
        f" ({r.last_failure or 'no reason given'})."
        for r in rows
        if r.rejected >= after and r.accepted == 0
    }
