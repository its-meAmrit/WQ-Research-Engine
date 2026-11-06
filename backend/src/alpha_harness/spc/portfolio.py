"""Portfolio figures that follow from saved output, and nothing that does not.

Every number here is arithmetic on positions a run produced. A confidence score is the model's
conviction and direction, not a normalised portfolio weight, so "exposure" below is the sum of
confidence scores and is labelled that way. Country comes from the ISIN's prefix (where the
security was issued, not necessarily where it trades or earns), and sector is not reported at
all: nothing here carries verified sector data, and a guess would look like a fact.

Local performance is deliberately absent. Without licensed, dated price data there is nothing
honest to compute, and a chart that resembled returns would be a fabrication.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

#: The scoring rules as WorldQuant's scoring article describes them, per the project brief. Shown
#: for orientation; this application does not compute the official score.
OFFICIAL_SCORING = (
    "Each prompt is evaluated on its update frequency.",
    "Active prompt portfolios are combined using their prompt weights.",
    "Daily portfolio PnL is calculated, and participants are ranked on cumulative monthly PnL.",
    "A participant needs PnL on at least 50% of trading days to be eligible for the monthly "
    "leaderboard.",
)

PERFORMANCE_UNAVAILABLE = (
    "No licensed, dated price history is configured, so no local performance is shown. Nothing "
    "on this screen is an estimate of a leaderboard score."
)

#: Confidence moved by at least this much counts as a change between two runs.
CHANGE_THRESHOLD = 0.2


@dataclass(frozen=True, slots=True)
class Pos:
    """A position as analytics sees it: only what it computes from."""

    isin: str
    mic: str
    ticker: str
    name: str
    confidence: float
    horizon: float
    prediction: float


def as_pos(row: dict[str, Any]) -> Pos:
    return Pos(
        isin=row["isin"],
        mic=row["mic"],
        ticker=row["ticker"],
        name=row["companyName"] if "companyName" in row else row["company_name"],
        confidence=float(
            row["confidenceScore"] if "confidenceScore" in row else row["confidence_score"]
        ),
        horizon=float(
            row["investmentHorizon"] if "investmentHorizon" in row else row["investment_horizon"]
        ),
        prediction=float(
            row["returnPrediction"] if "returnPrediction" in row else row["return_prediction"]
        ),
    )


def _share_table(counter: Counter[str], gross_by: dict[str, float]) -> list[dict[str, Any]]:
    total = sum(gross_by.values()) or 1.0
    return [
        {"key": key, "positions": counter[key], "share": round(gross_by[key] / total, 4)}
        for key in sorted(counter, key=lambda k: (-gross_by[k], k))
    ]


def summarise(positions: list[Pos]) -> dict[str, Any]:
    """The figures for one portfolio."""
    n = len(positions)
    if n == 0:
        return {"positions": 0, "empty": True}
    longs = [p for p in positions if p.confidence > 0]
    shorts = [p for p in positions if p.confidence < 0]
    gross = sum(abs(p.confidence) for p in positions)
    net = sum(p.confidence for p in positions)
    shares = [abs(p.confidence) / gross for p in positions] if gross else [0.0] * n
    hist = [0] * 10
    for p in positions:
        hist[min(9, int((p.confidence + 1.0) / 0.2))] += 1
    by_country: Counter[str] = Counter()
    by_mic: Counter[str] = Counter()
    gross_country: defaultdict[str, float] = defaultdict(float)
    gross_mic: defaultdict[str, float] = defaultdict(float)
    for p in positions:
        by_country[p.isin[:2]] += 1
        by_mic[p.mic] += 1
        gross_country[p.isin[:2]] += abs(p.confidence)
        gross_mic[p.mic] += abs(p.confidence)
    horizons = [p.horizon for p in positions]
    return {
        "positions": n,
        "empty": False,
        "longs": len(longs),
        "shorts": len(shorts),
        # Sums of confidence scores, not portfolio weights.
        "grossConfidence": round(gross, 4),
        "netConfidence": round(net, 4),
        "netShare": round(net / gross, 4) if gross else 0.0,
        "meanAbsConfidence": round(gross / n, 4),
        "largestShare": round(max(shares), 4),
        "top5Share": round(sum(sorted(shares, reverse=True)[:5]), 4),
        # 1/N for an even spread; 1 for a single position.
        "herfindahl": round(sum(s * s for s in shares), 4),
        "effectivePositions": round(1 / sum(s * s for s in shares), 1) if gross else 0.0,
        "meanHorizonDays": round(sum(horizons) / n, 1),
        "minHorizonDays": min(horizons),
        "maxHorizonDays": max(horizons),
        "meanReturnPrediction": round(sum(p.prediction for p in positions) / n, 6),
        "confidenceHistogram": [
            {"from": round(-1 + i * 0.2, 1), "to": round(-0.8 + i * 0.2, 1), "count": c}
            for i, c in enumerate(hist)
        ],
        "byIsinCountry": _share_table(by_country, gross_country),
        "byMic": _share_table(by_mic, gross_mic),
        "sector": None,
        "notes": [
            "Confidence scores are the model's conviction, not normalised weights: gross and "
            "net exposure are sums of scores.",
            "Country is the ISIN's prefix, the place of issue; it is not checked against where "
            "the company trades or earns.",
            "Sector is not shown: no verified sector data is loaded.",
        ],
    }


def compare(before: list[Pos], after: list[Pos]) -> dict[str, Any]:
    """How a later run's portfolio differs from an earlier one."""
    a = {p.isin: p for p in before}
    b = {p.isin: p for p in after}
    added = [b[k] for k in b if k not in a]
    removed = [a[k] for k in a if k not in b]
    flipped = [(a[k], b[k]) for k in a if k in b and (a[k].confidence > 0) != (b[k].confidence > 0)]
    changed = [
        (a[k], b[k])
        for k in a
        if k in b
        and (a[k].confidence > 0) == (b[k].confidence > 0)
        and abs(a[k].confidence - b[k].confidence) >= CHANGE_THRESHOLD
    ]

    def row(p: Pos) -> dict[str, Any]:
        return {"isin": p.isin, "ticker": p.ticker, "name": p.name, "confidence": p.confidence}

    def pair(x: Pos, y: Pos) -> dict[str, Any]:
        return {
            "isin": x.isin,
            "ticker": x.ticker,
            "name": x.name,
            "before": x.confidence,
            "after": y.confidence,
        }

    common = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return {
        "added": [row(p) for p in added],
        "removed": [row(p) for p in removed],
        "flipped": [pair(x, y) for x, y in flipped],
        "changed": [pair(x, y) for x, y in changed],
        "common": common,
        "jaccard": round(common / union, 4) if union else 0.0,
        "threshold": CHANGE_THRESHOLD,
    }


def overlap(portfolios: dict[int, list[Pos]]) -> dict[str, Any]:
    """Shared and opposed securities across several runs, keyed by run id."""
    ids = sorted(portfolios)
    held: dict[str, dict[int, Pos]] = {}
    for run_id in ids:
        for p in portfolios[run_id]:
            held.setdefault(p.isin, {})[run_id] = p
    pairs = []
    for x, y in combinations(ids, 2):
        a, b = {p.isin for p in portfolios[x]}, {p.isin for p in portfolios[y]}
        union = a | b
        conflicts = sum(
            1 for i in a & b if (held[i][x].confidence > 0) != (held[i][y].confidence > 0)
        )
        pairs.append(
            {
                "runA": x,
                "runB": y,
                "common": len(a & b),
                "jaccard": round(len(a & b) / len(union), 4) if union else 0.0,
                "conflicts": conflicts,
            }
        )
    conflicting: list[dict[str, Any]] = []
    for isin, by_run in held.items():
        signs = {c.confidence > 0 for c in by_run.values()}
        if len(by_run) > 1 and len(signs) > 1:
            first = next(iter(by_run.values()))
            conflicting.append(
                {
                    "isin": isin,
                    "ticker": first.ticker,
                    "name": first.name,
                    "byRun": {str(r): by_run[r].confidence for r in sorted(by_run)},
                }
            )
    shared = sum(1 for by_run in held.values() if len(by_run) > 1)
    return {
        "runs": ids,
        "pairs": pairs,
        "sharedSecurities": shared,
        "conflicting": sorted(conflicting, key=lambda c: str(c["isin"])),
        "uniqueSecurities": len(held),
    }
