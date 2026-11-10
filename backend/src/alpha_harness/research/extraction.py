"""Turning a paper's text into research ideas a person can check.

The model is asked for one JSON object that must satisfy :class:`Extraction`. Anything that
does not is rejected whole, with the validation problems kept, and nothing is filled in on its
behalf: a missing field is an error, and an honest "not stated" is a value the model may give.

What the model returns is then **checked against the text**, not trusted:

* each evidence item quotes the paper and names a page; the quote is searched for in that page
  (whitespace and hyphenation tolerated), then in the rest of the document, and the item is
  marked ``verified`` only when it is found. A quote the paper does not contain stays in the
  idea, marked unverified, so the person sees what the model claimed and that it did not hold.
* an idea is ``source_backed`` only when every author claim is tied to a verified quote;
  ``partially_supported`` when some are; ``speculative`` when none are, or when the idea cites
  nothing at all. The model cannot raise its own rating.
* what the authors claimed, what the model inferred and what the model proposes for BRAIN are
  three separate lists, and only the first may be described as the paper's.

An abstract-only analysis is read from the abstract alone, and every idea from it says so.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from .pdf import BACK_MATTER, PAGE_JOINER

PROMPT_VERSION = "paper-ideas-v1"

#: Total characters of paper text sent per analysis, over all chunks.
DEFAULT_CHAR_BUDGET = 60_000
#: Characters in one request: sections are packed up to this and never split mid-section
#: unless a single section is larger.
CHUNK_CHARS = 24_000
MAX_CHUNKS = 4
MAX_IDEAS = 8
#: A rough conversion for the estimate shown before the work starts.
CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_PER_CHUNK = 3_500
RAW_OUTPUT_KEPT = 6_000

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
NotStated = "Not stated in the text."

Topic = Literal[
    "value",
    "momentum",
    "quality",
    "growth",
    "size",
    "volatility",
    "liquidity",
    "sentiment",
    "analyst",
    "event",
    "macro",
    "other",
]
Direction = Literal["long_high", "long_low", "long_short_by_sign", "unclear"]
VariableKind = Literal[
    "price",
    "volume",
    "fundamental",
    "analyst",
    "sentiment",
    "macro",
    "intraday",
    "options",
    "order_book",
    "alternative",
    "other",
]
PitRisk = Literal["none", "revised_values", "announcement_lag_unknown", "uses_future_information"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Which section the quote is from, as the labels in the prompt name them (``s3``).
    section_id: str | None = None
    page: int = Field(ge=1, le=10_000)
    #: Copied from the text, not paraphrased: it is searched for.
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=12, max_length=600)]
    #: What the quote supports, in a few words.
    supports: ShortText


class Variable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The concept in the paper's words: "book-to-market ratio", "12-month return".
    name: ShortText
    description: Text
    kind: VariableKind
    #: False for a variable the idea can be tested without.
    essential: bool
    #: Plain words to look the data up by in a field catalogue. Not field ids.
    search_terms: Annotated[list[ShortText], Field(min_length=1, max_length=5)]
    point_in_time_risk: PitRisk


class Idea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=4, max_length=140)]
    topic: Topic
    main_finding: Text
    economic_explanation: Text
    signal_definition: Text
    signal_direction: Direction
    variables: Annotated[list[Variable], Field(min_length=1, max_length=12)]
    #: BRAIN operator names the idea would use, as the model best knows them; checked later.
    operators_needed: Annotated[list[ShortText], Field(max_length=15)]
    reported_formula: str | None = Field(default=None, max_length=1_000)
    formula_page: int | None = Field(default=None, ge=1, le=10_000)
    lookback_period: Text
    holding_period: Text
    universe: Text
    #: Countries or regions the paper's sample covers, in its words.
    markets_studied: Annotated[list[ShortText], Field(max_length=10)]
    point_in_time_requirements: Text
    methodology: Text
    statistical_limitations: Annotated[list[Text], Field(max_length=10)]
    implementation_variants: Annotated[list[Text], Field(max_length=8)]
    lookahead_concerns: Annotated[list[Text], Field(max_length=8)]
    missing_information: Annotated[list[Text], Field(max_length=10)]
    #: Statements the authors make. Each needs evidence; the harness checks it does.
    author_claims: Annotated[list[Text], Field(max_length=8)]
    #: The model's own reading of the paper. Never attributed to the authors.
    model_interpretations: Annotated[list[Text], Field(max_length=8)]
    #: New ideas the model proposes for BRAIN that the paper does not contain.
    proposed_hypotheses: Annotated[list[Text], Field(max_length=6)]
    evidence: Annotated[list[Evidence], Field(max_length=12)]

    @model_validator(mode="after")
    def _formula_needs_a_place(self) -> Idea:
        if self.reported_formula and self.formula_page is None:
            raise ValueError("reported_formula needs formula_page: say where it appears")
        return self


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_summary: Text
    methodology_summary: Text
    ideas: Annotated[list[Idea], Field(max_length=MAX_IDEAS)]
    #: Why there are no ideas, when there are none (a theory paper, a survey).
    no_ideas_reason: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def _empty_needs_a_reason(self) -> Extraction:
        if not self.ideas and not self.no_ideas_reason:
            raise ValueError("ideas is empty: no_ideas_reason must say why")
        return self


SCHEMA: dict[str, Any] = Extraction.model_json_schema()

SYSTEM = (
    "You are a careful quantitative research assistant. You read academic finance papers and "
    "extract testable equity-signal ideas. You work only from the text you are given and you "
    "never present your own thinking as the authors'. Reply with one JSON object and nothing else."
)


def build_prompt(
    *, title: str, scope: str, text: str, abstract_only: bool, part: tuple[int, int]
) -> str:
    """The user message for one chunk. ``part`` is ``(this, of)`` chunks."""
    basis = (
        "ABSTRACT ONLY. You have not been given the paper, only its abstract. Do not describe "
        "methodology, data, formulas or results the abstract does not state; use "
        f'"{NotStated}" and list the gap in missing_information. Set formula fields to null.'
        if abstract_only
        else "The text below is from the paper itself. Pages and sections are labelled in "
        "square brackets, like [p3 s4 Data and Methodology]."
    )
    return f"""Paper: {title or "(untitled)"}
Part {part[0]} of {part[1]}.
{basis}

Task: extract the independent, testable signal ideas this text supports (up to {MAX_IDEAS}). Two
ideas are independent when each could be tested alone. Separate ideas even if the paper
combines them.

Rules:
- author_claims: only things the authors state. Each must be backed by an evidence item.
- model_interpretations: your inferences. proposed_hypotheses: adaptations you propose for the
  WorldQuant BRAIN platform that the paper does not contain. Never put these in author_claims.
- evidence: copy the quote EXACTLY from the text (12-600 characters, no paraphrase, no
  ellipsis) and give the page number from the label. Quotes that are not in the text are
  discarded.
- variables: the financial data the signal needs, named in the paper's words, with plain
  search_terms (e.g. "book value", "market capitalization") — NOT BRAIN field ids. Never invent
  field ids. Mark essential=false for a variable the test could do without.
- point_in_time_risk per variable: revised_values if the paper used restated or final data,
  announcement_lag_unknown if the reporting lag is not stated, uses_future_information if the
  paper's signal could not have been known at formation time, else none.
- operators_needed: BRAIN operator names you believe apply (e.g. rank, ts_delta, ts_rank,
  group_neutralize), or an empty list. They will be checked.
- reported_formula: only if the paper prints one, copied as it appears, with formula_page. If
  the text of an equation looks garbled, set it to null and say so in missing_information.
- Where the text does not say, write "{NotStated}" or use an empty list. Do not guess.
- If the text supports no testable idea, return ideas: [] and explain in no_ideas_reason.

BRAIN scope the person will test in (for your information only): {scope}

JSON schema to satisfy:
{json.dumps(SCHEMA, separators=(",", ":"))}

Text:
{text}"""


# --- choosing and chunking text -------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    section_ids: list[str]
    chars: int


def usable_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sections an idea is read from: everything but references, acknowledgements and appendices.

    When no heading was recognised there is one section, the whole document, and it is kept.
    """
    kept = [s for s in sections if not BACK_MATTER.match(str(s["heading"]).strip())]
    return kept or sections


def chunk_document(
    full_text: str,
    sections: list[dict[str, Any]],
    page_ranges: list[tuple[int, int, int]],
    *,
    budget: int = DEFAULT_CHAR_BUDGET,
    chunk_chars: int = CHUNK_CHARS,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Section-aware chunks within ``budget`` characters, and what was left out.

    Whole sections are packed in reading order. A section too big for one chunk is cut on
    paragraph boundaries. Page labels are written into the text so the model can cite them.
    If the budget runs out, the rest is dropped and reported rather than silently ignored.
    """
    pieces: list[tuple[str, str]] = []
    for section in usable_sections(sections):
        body = full_text[int(section["start"]) : int(section["end"])]
        for page_text, page in _by_page(body, int(section["start"]), page_ranges):
            label = f"[p{page} {section['id']} {section['heading'][:60]}]"
            pieces.append((section["id"], f"{label}\n{page_text}"))
    chunks: list[Chunk] = []
    current: list[str] = []
    ids: list[str] = []
    size = used = 0
    dropped = 0
    for section_id, piece in pieces:
        for part in _split(piece, chunk_chars):
            if used + len(part) > budget or len(chunks) >= MAX_CHUNKS:
                dropped += len(part)
                continue
            if size + len(part) > chunk_chars and current:
                chunks.append(Chunk(len(chunks) + 1, "\n\n".join(current), ids, size))
                current, ids, size = [], [], 0
                if len(chunks) >= MAX_CHUNKS:
                    dropped += len(part)
                    continue
            current.append(part)
            ids.append(section_id) if section_id not in ids else None
            size += len(part) + 2
            used += len(part)
    if current and len(chunks) < MAX_CHUNKS:
        chunks.append(Chunk(len(chunks) + 1, "\n\n".join(current), ids, size))
    total = sum(len(p) for _, p in pieces)
    return chunks, {
        "chars_total": total,
        "chars_sent": sum(c.chars for c in chunks),
        "chars_left_out": dropped,
        "truncated": dropped > 0,
    }


def _by_page(
    body: str, offset: int, page_ranges: list[tuple[int, int, int]]
) -> list[tuple[str, int]]:
    """A section's text cut at page boundaries, each with its page number."""
    out: list[tuple[str, int]] = []
    for page, start, end in page_ranges:
        lo, hi = max(start, offset), min(end, offset + len(body))
        if lo < hi:
            out.append((body[lo - offset : hi - offset].strip(), page))
    return [(t, p) for t, p in out if t] or (
        [(body.strip(), page_ranges[0][0])] if body.strip() and page_ranges else []
    )


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    current = ""
    for paragraph in re.split(r"\n{2,}", text):
        while len(paragraph) > limit:
            if current:
                out.append(current)
                current = ""
            out.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        if len(current) + len(paragraph) + 2 > limit and current:
            out.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        out.append(current)
    return out


def estimate(chunks: list[Chunk], stats: dict[str, Any]) -> dict[str, Any]:
    """What the analysis will cost, shown before it starts."""
    sent = sum(c.chars for c in chunks)
    prompt_overhead = len(json.dumps(SCHEMA)) + 2_500
    return {
        "chunks": len(chunks),
        "requests": len(chunks),
        "charsSent": sent,
        "estimatedInputTokens": (sent + prompt_overhead * len(chunks)) // CHARS_PER_TOKEN,
        "estimatedOutputTokens": OUTPUT_TOKENS_PER_CHUNK * len(chunks),
        "truncated": bool(stats.get("truncated")),
        "charsLeftOut": int(stats.get("chars_left_out", 0)),
    }


def cache_key(
    *, source_ref: str, model_ref: str, budget: int, scope: str, abstract_only: bool
) -> str:
    """Same text, model, prompt, budget and scope: the same analysis, so it is not run again."""
    raw = "|".join([source_ref, model_ref, PROMPT_VERSION, str(budget), scope, str(abstract_only)])
    return hashlib.sha256(raw.encode()).hexdigest()


# --- reading the model's answer -------------------------------------------------------------


def json_object(raw: str) -> dict[str, Any]:
    """The JSON object in a model's answer, fenced or not."""
    return _json_object(raw)


class ModelOutputError(Exception):
    """An answer that is not valid. ``problems`` are shown to the person as they are."""

    def __init__(self, message: str, problems: list[dict[str, Any]], raw: str) -> None:
        super().__init__(message)
        self.problems = problems
        self.raw = raw[:RAW_OUTPUT_KEPT]


def _json_object(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse_answer(raw: str) -> Extraction:
    """The model's reply as an :class:`Extraction`, or :class:`ModelOutputError`."""
    if not raw.strip():
        raise ModelOutputError(
            "The model returned an empty answer.", [{"loc": [], "msg": "empty"}], raw
        )
    try:
        data = _json_object(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            "The model's answer was not valid JSON.",
            [{"loc": [], "msg": f"{exc.msg} (line {exc.lineno}, column {exc.colno})"}],
            raw,
        ) from exc
    try:
        return Extraction.model_validate(data)
    except ValidationError as exc:
        problems = [
            {"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors(include_url=False, include_input=False)
        ][:25]
        raise ModelOutputError(
            "The model's answer did not match the required structure "
            f"({len(exc.errors())} problems).",
            problems,
            raw,
        ) from exc


# --- checking it against the text -----------------------------------------------------------

_DASHES = str.maketrans("\u2010\u2011\u2012\u2013\u2014\u2212", "------")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_LIGATURES = str.maketrans(
    {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}
)


def squash(text: str) -> str:
    """Text reduced to what survives PDF extraction: case, spacing, line-end hyphens, ligatures
    and typographic punctuation are all ignored."""
    text = text.translate(_DASHES).translate(_QUOTES).translate(_LIGATURES)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"[^a-z0-9%.$]+", " ", text.lower()).strip()


@dataclass(slots=True)
class PageIndex:
    """The pages of one document, squashed once for many lookups."""

    pages: dict[int, str]
    whole: str

    @classmethod
    def build(cls, pages: list[dict[str, Any]]) -> PageIndex:
        squashed = {int(p["page"]): squash(str(p["text"])) for p in pages}
        return cls(squashed, " ".join(squashed[k] for k in sorted(squashed)))


def verify_evidence(item: dict[str, Any], index: PageIndex) -> dict[str, Any]:
    """``item`` with ``verified`` and a ``location`` note, from searching for its quote."""
    quote = squash(str(item["quote"]))
    page = int(item["page"])
    out = dict(item)
    if len(quote) < 10:
        return out | {"verified": False, "location": "quote too short to check"}
    if quote in index.pages.get(page, ""):
        return out | {"verified": True, "location": f"page {page}"}
    for number, text in sorted(index.pages.items()):
        if quote in text:
            return out | {
                "verified": True,
                "page": number,
                "location": f"page {number} (the model said page {page})",
            }
    # A quote that crosses a page break appears only in the joined text.
    if quote in index.whole:
        return out | {"verified": True, "location": "spans a page break"}
    return out | {"verified": False, "location": "not found in the document text"}


def classify_support(idea: Idea, checked: list[dict[str, Any]], *, abstract_only: bool) -> str:
    """``source_backed``, ``partially_supported`` or ``speculative``.

    Counted from the verified quotes only. With no author claims there is nothing the paper
    is said to support, so the idea can be at most partial, and with no verified quote it is
    speculative whatever the model said. An abstract-only idea is never ``source_backed``: an
    abstract does not carry a method.
    """
    verified = [e for e in checked if e["verified"]]
    if not verified:
        return "speculative"
    if abstract_only:
        return "partially_supported"
    if idea.author_claims and len(verified) >= len(idea.author_claims):
        return "source_backed"
    return "partially_supported"


def formula_check(
    idea: Idea, equations: list[dict[str, Any]], index: PageIndex
) -> tuple[str | None, bool]:
    """The formula to store and whether it needs manual review.

    A formula is kept as the model gave it only when it appears in the extracted text of the
    page it names. Otherwise it is dropped, since a formula the text does not hold is the
    model's, and never a repaired version of the paper's. If the extractor marked equations
    on that page as doubtful, the formula is kept but flagged.
    """
    if not idea.reported_formula or idea.formula_page is None:
        return None, False
    page_text = index.pages.get(idea.formula_page, "")
    if squash(idea.reported_formula) not in page_text:
        return None, True
    doubtful = any(e["page"] == idea.formula_page and e["needs_review"] for e in equations)
    return idea.reported_formula, doubtful


def idea_row(
    idea: Idea,
    *,
    index: PageIndex,
    equations: list[dict[str, Any]],
    abstract_only: bool,
) -> dict[str, Any]:
    """The columns of a :class:`~alpha_harness.db.models.ResearchIdea` for ``idea``."""
    checked = [verify_evidence(e.model_dump(), index) for e in idea.evidence]
    formula, review = formula_check(idea, equations, index)
    missing = list(idea.missing_information)
    if idea.reported_formula and formula is None:
        missing.append(
            "The model reported a formula that could not be found in the extracted text of "
            f"page {idea.formula_page}, so it was not kept. Check the equation in the PDF."
        )
    if abstract_only:
        missing.append(
            "Read from the abstract only; the paper's method and data were not available."
        )
    unverified = sum(1 for e in checked if not e["verified"])
    if unverified:
        missing.append(f"{unverified} cited passage(s) could not be found in the document.")
    return {
        "title": idea.title,
        "topic": idea.topic,
        "main_finding": idea.main_finding,
        "economic_hypothesis": idea.economic_explanation,
        "anomaly": idea.main_finding,
        "signal_definition": idea.signal_definition,
        "signal_direction": idea.signal_direction,
        "required_variables": [v.model_dump() for v in idea.variables],
        "reported_formula": formula,
        "formula_needs_review": review,
        "lookback": idea.lookback_period,
        "holding_period": idea.holding_period,
        "universe": idea.universe,
        "point_in_time": idea.point_in_time_requirements,
        "methodology": idea.methodology,
        "limitations": idea.statistical_limitations,
        "implementation_variants": idea.implementation_variants,
        "lookahead_concerns": idea.lookahead_concerns,
        "missing_information": missing,
        "claims": {
            "authorClaims": idea.author_claims,
            "modelInterpretations": idea.model_interpretations,
            "proposedHypotheses": idea.proposed_hypotheses,
            "operatorsNeeded": idea.operators_needed,
            "marketsStudied": idea.markets_studied,
        },
        "evidence": checked,
        "support": classify_support(idea, checked, abstract_only=abstract_only),
        "evidence_scope": "abstract" if abstract_only else "paper",
        "search_text": " ".join(
            [idea.title, idea.main_finding, idea.economic_explanation]
            + [v.name for v in idea.variables]
        ).lower(),
    }


def merge_ideas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ideas from several chunks of one paper, the same idea found twice kept once.

    Two ideas are the same when their titles are, after squashing, or one contains the other.
    The one with more verified evidence is kept.
    """
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = squash(row["title"])
        for i, other in enumerate(kept):
            existing = squash(other["title"])
            if key == existing or (len(key) > 12 and (key in existing or existing in key)):
                if _score(row) > _score(other):
                    kept[i] = row
                break
        else:
            kept.append(row)
    return kept


def _score(row: dict[str, Any]) -> int:
    return sum(1 for e in row["evidence"] if e["verified"])


__all__ = [
    "PAGE_JOINER",
    "Chunk",
    "Extraction",
    "Idea",
    "ModelOutputError",
    "PageIndex",
    "build_prompt",
    "cache_key",
    "chunk_document",
    "estimate",
    "idea_row",
    "merge_ideas",
    "parse_answer",
    "verify_evidence",
]
