"""Recognising an idea already held, and the ones near it.

Three signals, from strict to loose. The *fingerprint* is a hash of the idea's content words
(title, signal, variables) in sorted order: the same words in any order are the same idea.
*Similarity* is the overlap of those words and of word pairs. Two thresholds follow:

* at or above :data:`SAME` the new statement is treated as the held idea found again and
  filed as a further source of it;
* between :data:`RELATED` and :data:`SAME` the ideas are shown to each other as related, or as
  a possible duplicate above :data:`POSSIBLE`, and both are kept.

Nothing is dropped for being merely similar, and no source reference is ever merged into
another: a repeat adds a link.
"""

import hashlib
import re
from itertools import pairwise
from typing import Any

SAME = 0.82
POSSIBLE = 0.55
RELATED = 0.30

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "within",
        "using",
        "use",
        "used",
        "based",
        "than",
        "then",
        "these",
        "those",
        "such",
        "stock",
        "stocks",
        "equity",
        "equities",
        "return",
        "returns",
        "signal",
        "signals",
        "factor",
        "strategy",
        "model",
        "predict",
        "predicts",
        "predictive",
        "high",
        "low",
        "future",
        "across",
        "over",
    ]
)


def tokens(text: str) -> list[str]:
    """Content words, lower case, a trailing plural dropped, stop words removed."""
    out: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word in _STOP or len(word) < 2:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def core_text(row: dict[str, Any]) -> str:
    """What an idea is, for comparing: title, signal and the variables it measures."""
    variables = " ".join(str(v.get("name", "")) for v in row.get("required_variables") or [])
    return " ".join([str(row.get("title", "")), str(row.get("signal_definition", "")), variables])


def view_row(idea: Any) -> dict[str, Any]:
    """A library idea (an ORM row) as the plain mapping the functions here compare."""
    return {
        "title": idea.title,
        "signal_definition": idea.signal_definition,
        "required_variables": idea.required_variables,
    }


def fingerprint(row: dict[str, Any]) -> str:
    words = sorted(set(tokens(core_text(row))))
    return hashlib.sha256(" ".join(words).encode()).hexdigest()


def _shingles(words: list[str]) -> set[str]:
    return set(words) | {f"{a} {b}" for a, b in pairwise(words)}


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """0-1: how much two ideas say the same thing, from their content words and word pairs."""
    first, second = _shingles(tokens(core_text(a))), _shingles(tokens(core_text(b)))
    if not first or not second:
        return 0.0
    overlap = len(first & second)
    jaccard = overlap / len(first | second)
    contained = overlap / min(len(first), len(second))
    return round(max(jaccard, 0.85 * contained), 3)


def classify(score: float) -> str:
    return (
        "same"
        if score >= SAME
        else "possible_duplicate"
        if score >= POSSIBLE
        else "related"
        if score >= RELATED
        else "unrelated"
    )
