"""A broad topic turned into focused research queries.

"Profitable alpha strategies" finds sales pages. A useful query asks for what a researcher
needs: how the signal is built, why it should work, which variables measure it and what the
evidence is. Each topic the person is likely to type is recognised and given the terms the
literature uses for it; any other topic gets the same four angles around its own words.

The result is only a proposal: the person edits it before anything is searched.
"""

import re
from dataclasses import dataclass

MAX_QUERIES = 5
MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 300

#: Shown as suggestions on the Discover screen.
SUGGESTED_TOPICS = [
    "Earnings surprise",
    "Price momentum",
    "Mean reversion",
    "Analyst estimate revisions",
    "Quality and profitability",
    "Volatility forecasting",
    "Liquidity anomalies",
    "Alternative data",
    "Cross-sectional factor investing",
    "Seasonal effects",
]

#: Terms the literature uses for a topic, found by a keyword in what the person typed. They
#: are the names of well-known measures, added so a query can match how papers are written.
_VOCABULARY: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"earnings (surprise|announcement)|\bsue\b|\bpead\b", re.IGNORECASE),
        "standardized unexpected earnings post-earnings-announcement drift",
    ),
    (
        re.compile(r"momentum", re.IGNORECASE),
        "past return formation period skip month winners losers",
    ),
    (
        re.compile(r"mean[- ]revers|reversal", re.IGNORECASE),
        "short-term reversal contrarian overreaction",
    ),
    (
        re.compile(r"analyst|estimate revision|forecast revision", re.IGNORECASE),
        "analyst forecast revisions dispersion consensus",
    ),
    (
        re.compile(r"quality|profitab", re.IGNORECASE),
        "gross profitability accruals earnings quality",
    ),
    (
        re.compile(r"volatil", re.IGNORECASE),
        "idiosyncratic volatility realized variance forecasting",
    ),
    (
        re.compile(r"liquid|illiquid", re.IGNORECASE),
        "illiquidity bid-ask spread turnover price impact",
    ),
    (
        re.compile(r"alternative data|alt data", re.IGNORECASE),
        "alternative data web traffic satellite credit card text signals",
    ),
    (
        re.compile(r"cross[- ]section|factor invest", re.IGNORECASE),
        "cross-sectional return predictability firm characteristics",
    ),
    (
        re.compile(r"season|calendar|turn[- ]of[- ]the[- ]month", re.IGNORECASE),
        "calendar anomalies seasonality",
    ),
]

_ANGLES = [
    ("methodology", "{topic} {vocab} signal construction methodology equity portfolio sorts"),
    ("rationale", "{topic} economic rationale risk-based behavioral explanation equity anomaly"),
    ("variables", "{topic} {vocab} variable definition measurement data required"),
    ("evidence", "{topic} empirical evidence out-of-sample robustness transaction costs"),
    ("implementation", "{topic} factor implementation open-source replication code"),
]


@dataclass(slots=True)
class ProposedQuery:
    #: Why this query: which part of the research it is after.
    angle: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"angle": self.angle, "text": self.text}


class QueryError(ValueError):
    """Queries that cannot be searched."""


def _clean(text: str) -> str:
    return " ".join(text.split())


def propose(topic: str, *, market: str = "", count: int = MAX_QUERIES) -> list[ProposedQuery]:
    """Up to ``count`` queries for ``topic``, optionally within a market ("US equities")."""
    base = _clean(topic)
    if len(base) < MIN_QUERY_CHARS:
        raise QueryError("Enter a topic of at least three characters.")
    vocabulary = next((terms for pattern, terms in _VOCABULARY if pattern.search(base)), "")
    context = _clean(market)
    queries: list[ProposedQuery] = []
    for angle, template in _ANGLES[: max(1, min(count, MAX_QUERIES))]:
        text = _clean(template.format(topic=base, vocab=vocabulary))
        if context:
            text = f"{text} {context}"
        queries.append(ProposedQuery(angle, text[:MAX_QUERY_CHARS].rstrip()))
    return queries


def validate(queries: list[str]) -> list[str]:
    """The queries the person finally chose, cleaned, de-duplicated and bounded."""
    cleaned = list(dict.fromkeys(q for q in (_clean(q) for q in queries) if q))
    if not cleaned:
        raise QueryError("Enter at least one search query.")
    if len(cleaned) > MAX_QUERIES:
        raise QueryError(
            f"At most {MAX_QUERIES} queries can run in one search; "
            f"remove {len(cleaned) - MAX_QUERIES}."
        )
    for query in cleaned:
        if len(query) < MIN_QUERY_CHARS:
            raise QueryError(f"“{query}” is too short to search for.")
        if len(query) > MAX_QUERY_CHARS:
            raise QueryError(f"A query is limited to {MAX_QUERY_CHARS} characters.")
    return cleaned
