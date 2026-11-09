"""Reading a PDF: text by page, section headings, tables, and equations flagged for review.

Blocking and CPU-bound, so the service runs :func:`parse_pdf` on a thread. Everything here is
bounded before it is trusted: the file's size, its page count, the characters kept and the
time spent are all capped, and a file over any of them is refused with a reason rather than
read halfway and presented as the whole.

A PDF is data. Nothing in it is run: PyMuPDF does not execute JavaScript, and this module
never opens an embedded file, a launch action or a link.

**What is never done**: an equation is never rebuilt from a picture or guessed from a broken
glyph. A line that looks like one is kept as the text the file holds, marked
``needs_review`` whenever the file used a maths font, a private-use character or a split
layout, because those are the cases where extraction goes wrong without saying so. A page
with no text layer is reported, and no OCR is attempted.
"""

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from .identifiers import normalise_arxiv, normalise_doi

pymupdf.TOOLS.mupdf_display_errors(False)
# An advert for a separate package, printed to stdout the first time a page is read.
pymupdf.no_recommend_layout()

PARSER = f"pymupdf {pymupdf.__version__}"

#: How pages are joined into the one string the section offsets index.
PAGE_JOINER = "\n\n"

MAX_BYTES = 30 * 1024 * 1024
MAX_PAGES = 120
MAX_CHARS = 1_500_000
DEADLINE_SECONDS = 45.0
MAX_TABLES = 25
MAX_TABLE_ROWS = 40
MAX_EQUATIONS = 400
#: Fewer visible characters than this and a page is treated as having no text layer.
BLANK_PAGE_CHARS = 25
#: Share of pages without text beyond which the file is called scanned and refused.
SCANNED_SHARE = 0.5


class PdfError(Exception):
    """A file this module will not read, with a stable ``code`` and a message for a person."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Limits:
    max_bytes: int = MAX_BYTES
    max_pages: int = MAX_PAGES
    max_chars: int = MAX_CHARS
    deadline_seconds: float = DEADLINE_SECONDS


@dataclass(slots=True)
class Section:
    id: str
    heading: str
    level: int
    start: int
    end: int
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "heading": self.heading,
            "level": self.level,
            "start": self.start,
            "end": self.end,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(slots=True)
class ParsedPdf:
    pages: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    equations: list[dict[str, Any]]
    #: Title, authors, abstract, DOI and arXiv id as the file itself shows them: guesses
    #: from layout, labelled as such, to be checked against a registry where one exists.
    metadata: dict[str, Any]
    page_count: int
    stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.stats.get("blankPages")) or bool(self.stats.get("truncated"))


# --- validation ---------------------------------------------------------------------------


def validate_bytes(data: bytes, limits: Limits) -> None:
    """Refuse what is plainly not a PDF, or too big to read, before opening it."""
    if not data:
        raise PdfError("empty_file", "The file is empty.")
    if len(data) > limits.max_bytes:
        raise PdfError(
            "too_large",
            f"The file is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{limits.max_bytes // 1_048_576} MB.",
        )
    # The specification lets a few hundred bytes precede the header; readers scan 1 KiB.
    if b"%PDF-" not in data[:1024]:
        raise PdfError("not_a_pdf", "That file is not a PDF: it does not start with a PDF header.")


# --- text ---------------------------------------------------------------------------------

_MATH_FONT = re.compile(
    r"(math|cmmi|cmsy|cmex|cmr\d|msbm|msam|symbol|stix|mt\s?extra|euclid|mathematica)",
    re.IGNORECASE,
)
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
_BROKEN = re.compile(r"\ufffd|\(cid:\d+\)")
_MATH_CHARS = set(
    "\u2211\u220f\u222b\u2202\u221a\u221e\u2264\u2265\u2260\u2248"
    "\u00b1\u00d7\u00f7\u00b7\u2208\u2209\u2282\u2286\u222a\u2229"
    "\u2192\u21d2\u2200\u2203\u2207\u0394\u03a3\u03a0\u03a9"
    "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03bb"
    "\u03bc\u03bd\u03c0\u03c1\u03c3\u03c4\u03c6\u03c7\u03c8\u03c9"
)
_EQ_NUMBER = re.compile(r"\(\s*\d{1,2}(?:\.\d{1,2})?[a-z]?\s*\)\s*$")

_NUMBERED = re.compile(
    r"^((?:\d{1,2}|[IVX]{1,5}|[A-H])(?:\.\d{1,2}){0,3})[.)]?\s+([A-Z][^\n]{2,100})$"
)
_KNOWN = re.compile(
    r"^(abstract|introduction|literature review|related work|background|data|data and methodology|"
    r"data and sample|sample|methodology|methods?|"
    r"empirical (?:methodology|results|analysis|strategy)|"
    r"model|results|findings|discussion|robustness(?: checks| tests)?|conclusions?|references|"
    r"appendix[\w .:-]*|acknowledg(?:e)?ments?|tables?|figures?)\.?$",
    re.IGNORECASE,
)
_CAPTION = re.compile(r"^(table|figure|fig\.|panel)\s+[\dIVX]+", re.IGNORECASE)
#: Headings that open material no idea is read from.
BACK_MATTER = re.compile(
    r"^(references|bibliography|acknowledg|appendix|online appendix)", re.IGNORECASE
)


@dataclass(slots=True)
class _Line:
    text: str
    size: float
    bold: bool
    math_font: bool
    page: int
    top: float


def _lines(page: Any, number: int) -> tuple[list[_Line], list[str]]:
    """The page's lines with the size and weight of their text, and its blocks as paragraphs."""
    raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    out: list[_Line] = []
    paragraphs: list[str] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        texts: list[str] = []
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text")]
            text = "".join(s["text"] for s in spans).rstrip()
            if not text.strip():
                continue
            texts.append(text)
            weights = [(len(s["text"]), s) for s in spans]
            biggest = max(weights, key=lambda w: w[0])[1] if weights else {"size": 0.0, "flags": 0}
            out.append(
                _Line(
                    text=text.strip(),
                    size=round(float(biggest.get("size", 0.0)), 1),
                    bold=bool(int(biggest.get("flags", 0)) & 16)
                    or "bold" in str(biggest.get("font", "")).lower(),
                    math_font=any(_MATH_FONT.search(str(s.get("font", ""))) for s in spans),
                    page=number,
                    top=float(line.get("bbox", (0, 0, 0, 0))[1]),
                )
            )
        if texts:
            paragraphs.append("\n".join(texts))
    return out, paragraphs


def _body_size(lines: list[_Line]) -> float:
    """The size most characters are set in: what a heading has to stand out from."""
    weight: Counter[float] = Counter()
    for line in lines:
        weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 0.0


def _is_heading(line: _Line, body: float) -> tuple[bool, int]:
    text = line.text
    if len(text) > 110 or len(text) < 3 or _CAPTION.match(text):
        return False, 0
    if text.endswith((",", ";")) or text.count(" ") > 14:
        return False, 0
    known = _KNOWN.match(text.strip(" .:"))
    numbered = _NUMBERED.match(text)
    larger = body > 0 and line.size >= body * 1.15
    # A bold or larger line only counts when it also reads like a title: a bold sentence in a
    # paragraph is emphasis, and a bold table cell is data.
    if numbered and (line.bold or larger or numbered.group(1).count(".") == 0):
        depth = numbered.group(1).count(".") + 1
        return True, min(depth, 3)
    if known and (line.bold or larger or text.isupper() or text[:1].isupper()):
        return True, 1
    if larger and line.bold and not text.endswith("."):
        return True, 1
    return False, 0


# --- equations ----------------------------------------------------------------------------


def _equation(line: _Line, neighbours: tuple[str, str]) -> dict[str, Any] | None:
    """A line that looks like an equation, kept verbatim and flagged when it cannot be trusted."""
    text = line.text
    if len(text) > 240 or "=" not in text:
        return None
    mathy = (
        line.math_font
        or bool(_EQ_NUMBER.search(text))
        or any(c in _MATH_CHARS for c in text)
        or bool(_PRIVATE_USE.search(text))
    )
    # A sentence containing "=" in prose ("where t = 1 is the first month") is not display math.
    if not mathy or (len(text.split()) > 22 and not line.math_font):
        return None
    reasons: list[str] = []
    if line.math_font:
        reasons.append("maths font: symbols may not have extracted correctly")
    if _PRIVATE_USE.search(text) or _BROKEN.search(text):
        reasons.append("unrecognised glyphs in the extracted text")
    # A fraction or a sum set over several lines extracts as separate short lines.
    if any(len(n.strip()) <= 4 and n.strip() for n in neighbours):
        reasons.append("layout is split over several lines")
    if text.count("(") != text.count(")"):
        reasons.append("unbalanced brackets")
    return {
        "page": line.page,
        "text": text,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }


# --- tables -------------------------------------------------------------------------------


def _tables(page: Any, number: int) -> list[dict[str, Any]]:
    """Tables PyMuPDF finds with ruling or clear alignment, kept only when they are clean."""
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 - a table finder failing must not fail the paper
        return []
    kept: list[dict[str, Any]] = []
    for table in found.tables:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001, S112 - one unreadable table is skipped
            continue
        cleaned = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
        cells = [c for row in cleaned for c in row]
        if len(cleaned) < 2 or len(cleaned[0]) < 2 or not cells:
            continue
        # A "table" that is mostly empty cells is a layout the finder misread.
        if sum(1 for c in cells if c) / len(cells) < 0.5:
            continue
        kept.append(
            {
                "page": number,
                "rows": cleaned[:MAX_TABLE_ROWS],
                "truncated": len(cleaned) > MAX_TABLE_ROWS,
            }
        )
    return kept


# --- the document -------------------------------------------------------------------------

_ABSTRACT_END = re.compile(
    r"\n\s*(keywords?|jel\b|1[.\s]+introduction|introduction\b)", re.IGNORECASE
)


def _abstract(text: str, sections: list[Section]) -> str | None:
    for section in sections:
        if section.heading.strip(" .:").lower() == "abstract":
            body = text[section.start : section.end].split("\n", 1)
            return " ".join(body[1].split())[:3000] if len(body) > 1 else None
    head = text[:6000]
    match = re.search(r"\babstract\b[\s.:—-]*", head, re.IGNORECASE)
    if match is None:
        return None
    rest = head[match.end() :]
    stop = _ABSTRACT_END.search(rest)
    return " ".join((rest[: stop.start()] if stop else rest[:2000]).split()) or None


def _title(first: list[_Line], info_title: str) -> str | None:
    """The largest text in the top half of page one; the file's own title only when that
    finds nothing, since a title field often holds a file name or a template's."""
    top = [ln for ln in first if ln.top < 360 and len(ln.text) > 3]
    if top:
        size = max(ln.size for ln in top)
        parts = [ln.text for ln in top if ln.size >= size - 0.5]
        guess = " ".join(parts)[:300].strip()
        if len(guess) > 8:
            return guess
    clean = info_title.strip()
    return (
        clean if len(clean) > 8 and not clean.lower().endswith((".pdf", ".docx", ".tex")) else None
    )


def _authors(info_author: str) -> list[str]:
    parts = re.split(r"\s*(?:;|,\s+and\s+|\band\b|\s&\s|,)\s*", info_author)
    return [p.strip() for p in parts if 3 < len(p.strip()) < 80][:30]


def parse_pdf(data: bytes, limits: Limits | None = None) -> ParsedPdf:
    """Read ``data`` as a PDF within ``limits``. Raises :class:`PdfError` for what it refuses."""
    limits = limits or Limits()
    validate_bytes(data, limits)
    started = time.monotonic()
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfError(
            "corrupted", "That PDF could not be opened. It may be damaged or truncated."
        ) from exc
    try:
        return _parse(doc, limits, started)
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(
            "corrupted", f"That PDF could not be read ({type(exc).__name__}). It may be damaged."
        ) from exc
    finally:
        doc.close()


def _parse(doc: Any, limits: Limits, started: float) -> ParsedPdf:
    if doc.needs_pass or doc.is_encrypted:
        raise PdfError(
            "encrypted", "That PDF is password-protected. Remove the password and retry."
        )
    count = int(doc.page_count)
    if count == 0:
        raise PdfError(
            "corrupted", "That PDF has no readable pages. It may be damaged or truncated."
        )
    if count > limits.max_pages:
        raise PdfError(
            "too_many_pages",
            f"That PDF has {count} pages; the limit is {limits.max_pages}. "
            "Upload the paper without its appendices or data supplement.",
        )

    pages: list[dict[str, Any]] = []
    all_lines: list[_Line] = []
    paragraphs_by_page: list[list[str]] = []
    blank: list[int] = []
    garbled: list[int] = []
    tables: list[dict[str, Any]] = []
    total = 0
    truncated = False
    warnings: list[str] = []

    for index in range(count):
        if time.monotonic() - started > limits.deadline_seconds:
            raise PdfError(
                "timeout",
                f"Reading that PDF took longer than {limits.deadline_seconds:.0f} seconds. "
                "It may be unusually complex.",
            )
        number = index + 1
        page = doc.load_page(index)
        lines, paragraphs = _lines(page, number)
        text = "\n\n".join(paragraphs)
        visible = len(re.sub(r"\s+", "", text))
        if visible < BLANK_PAGE_CHARS:
            blank.append(number)
        elif len(_BROKEN.findall(text)) + len(_PRIVATE_USE.findall(text)) > max(8, visible // 20):
            garbled.append(number)
        if total + len(text) > limits.max_chars:
            truncated = True
            warnings.append(
                f"Text stops at page {number - 1}: the document is longer than "
                f"{limits.max_chars:,} characters."
            )
            break
        total += len(text) + len(PAGE_JOINER)
        pages.append({"page": number, "text": text})
        all_lines.extend(lines)
        paragraphs_by_page.append(paragraphs)
        if (
            len(tables) < MAX_TABLES
            and re.search(r"\btable\b", text, re.IGNORECASE)
            and visible >= BLANK_PAGE_CHARS
        ):
            tables.extend(_tables(page, number)[: MAX_TABLES - len(tables)])

    read = len(pages)
    if len(blank) > read * SCANNED_SHARE:
        raise PdfError(
            "no_text_layer",
            "This PDF has almost no extractable text, so it is probably scanned images. "
            "OCR is not performed here, so nothing was read. Upload a text-based PDF, "
            "or import by DOI to keep the abstract only.",
        )
    if blank:
        warnings.append(
            f"{len(blank)} page(s) had no extractable text ({_ranges(blank)}) and were skipped."
        )
    if garbled:
        warnings.append(
            f"Page(s) {_ranges(garbled)} contain unreadable characters; text and any "
            "equations there are unreliable."
        )

    full = PAGE_JOINER.join(p["text"] for p in pages)
    offsets = _offsets(pages)
    first_page_lines = [ln for ln in all_lines if ln.page == 1]
    meta = doc.metadata or {}
    title = _title(first_page_lines, str(meta.get("title") or ""))
    sections = _sections(all_lines, pages, offsets, full, title)
    equations = _equations(all_lines, garbled)
    head = "\n".join(p["text"] for p in pages[:2])
    doi = normalise_doi(_find(r"10\.\d{4,9}/[^\s\"<>]+", head) or "")
    arxiv = normalise_arxiv(
        _find(r"arXiv:\s*([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", head, 1) or ""
    )
    abstract = _abstract(full, [Section(**s) for s in sections])
    metadata = {
        "title": title,
        "authors": _authors(str(meta.get("author") or "")),
        "abstract": abstract,
        "doi": doi,
        "arxiv_id": arxiv,
        "basis": "Read from the PDF's own layout and metadata; verify against the publisher.",
    }
    return ParsedPdf(
        pages=pages,
        sections=sections,
        tables=tables,
        equations=equations[:MAX_EQUATIONS],
        metadata=metadata,
        page_count=count,
        stats={
            "chars": len(full),
            "pagesRead": read,
            "blankPages": blank,
            "garbledPages": garbled,
            "truncated": truncated,
            "tables": len(tables),
            "equations": len(equations),
            "equationsNeedingReview": sum(1 for e in equations if e["needs_review"]),
        },
        warnings=warnings,
    )


def _find(pattern: str, text: str, group: int = 0) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group) if match else None


def _ranges(numbers: list[int]) -> str:
    """``[1, 2, 3, 7]`` as ``1-3, 7``."""
    spans: list[str] = []
    start = prev = numbers[0]
    for n in [*numbers[1:], None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        spans.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(spans)


def _offsets(pages: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """``(page, start, end)`` of each page in the joined text."""
    out: list[tuple[int, int, int]] = []
    cursor = 0
    for p in pages:
        end = cursor + len(p["text"])
        out.append((p["page"], cursor, end))
        cursor = end + len(PAGE_JOINER)
    return out


def page_at(offsets: list[tuple[int, int, int]], position: int) -> int:
    for page, start, end in offsets:
        if start <= position <= end + len(PAGE_JOINER) - 1:
            return page
    return offsets[-1][0] if offsets else 1


def _sections(
    lines: list[_Line],
    pages: list[dict[str, Any]],
    offsets: list[tuple[int, int, int]],
    full: str,
    title: str | None = None,
) -> list[dict[str, Any]]:
    """Section boundaries as offsets into the joined text, found from heading-looking lines.

    Falls back to one section for the whole document when no heading is found: the pipeline
    then chunks by size, and the person is told sections were not recognised.
    """
    body = _body_size(lines)
    found: list[tuple[int, str, int]] = []
    cursor_by_page = {page: start for page, start, _ in offsets}
    for line in lines:
        heading, level = _is_heading(line, body)
        # The paper's own title is the document's name, not a section of it.
        if not heading or (title and line.page == 1 and line.text in title):
            continue
        base = cursor_by_page.get(line.page)
        if base is None:
            continue
        position = full.find(line.text, base)
        if position < 0:
            continue
        # The same heading text recurs in running page headers; the first hit per page counts.
        if found and found[-1][0] >= position:
            continue
        found.append((position, line.text, level))
    if not found:
        return [
            Section(
                "s1", "Document", 1, 0, len(full), pages[0]["page"], pages[-1]["page"]
            ).to_dict()
        ]
    sections: list[Section] = []
    if found[0][0] > 200:
        found.insert(0, (0, "Front matter", 1))
    for n, (start, heading, level) in enumerate(found):
        end = found[n + 1][0] if n + 1 < len(found) else len(full)
        sections.append(
            Section(
                f"s{n + 1}",
                heading.strip(),
                level,
                start,
                end,
                page_at(offsets, start),
                page_at(offsets, max(start, end - 1)),
            )
        )
    return [s.to_dict() for s in sections]


def _equations(lines: list[_Line], garbled_pages: list[int]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        before = lines[i - 1].text if i else ""
        after = lines[i + 1].text if i + 1 < len(lines) else ""
        eq = _equation(line, (before, after))
        if eq is None:
            continue
        if line.page in garbled_pages and not eq["needs_review"]:
            eq["needs_review"] = True
            eq["review_reasons"].append("the page's text is unreliable")
        found.append(eq)
    return found
