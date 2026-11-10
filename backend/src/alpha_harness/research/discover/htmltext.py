"""A web page as text: its headings, its paragraphs and what it says about itself.

Parsing only: nothing in the page is run, followed or fetched. Scripts, styles, navigation,
forms and embedded frames are dropped; headings become section boundaries so the extraction
can cite where a passage is; the page's own ``citation_*``, Open Graph and JSON-LD tags give
the author and date when it states them. A page that states none gets none.
"""

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

MAX_TEXT_CHARS = 200_000
MAX_JSON_LD_CHARS = 100_000
_SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "aside", "form", "iframe",
         "template", "button", "select", "dialog", "canvas", "head"}  # fmt: skip
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
_BLOCKS = {"p", "li", "blockquote", "pre", "td", "th", "dd", "dt", "figcaption", "caption", "div",
           "section", "article", "br", "tr"}  # fmt: skip
_SPACE = re.compile(r"[ \t\r\f\v\u00a0]+")


@dataclass(slots=True)
class ParsedPage:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    published: str | None = None
    site_name: str | None = None
    description: str | None = None
    canonical: str | None = None
    text: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.title = ""
        self.canonical: str | None = None
        self.json_ld: list[str] = []
        self.blocks: list[tuple[int, str]] = []  # (heading level or 0, text)
        self._skip = 0
        self._in_title = False
        self._heading = 0
        self._ld = False
        self._buffer: list[str] = []
        self._ld_buffer: list[str] = []

    def finish(self) -> None:
        """Flush text left after the last tag."""
        self._flush()

    def _flush(self) -> None:
        text = _SPACE.sub(" ", "".join(self._buffer)).strip()
        self._buffer = []
        if text:
            self.blocks.append((self._heading, text))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fields = {k.lower(): v for k, v in attrs if v is not None}
        if tag == "meta":
            key = (fields.get("name") or fields.get("property") or "").lower()
            if key and fields.get("content"):
                self.meta.setdefault(key, []).append(fields["content"])
            return
        if tag == "link" and "canonical" in fields.get("rel", "").lower():
            self.canonical = fields.get("href")
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "script" and "ld+json" in fields.get("type", "").lower():
            self._ld = True
            return
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADINGS:
            self._flush()
            self._heading = _HEADINGS[tag]
        elif tag in _BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._ld:
            self._ld = False
            self.json_ld.append("".join(self._ld_buffer)[:MAX_JSON_LD_CHARS])
            self._ld_buffer = []
        elif tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif not self._skip and (tag in _HEADINGS or tag in _BLOCKS):
            self._flush()
            if tag in _HEADINGS:
                self._heading = 0

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._ld:
            self._ld_buffer.append(data)
        elif not self._skip:
            self._buffer.append(data)


def _first(meta: dict[str, list[str]], *keys: str) -> str | None:
    return next((meta[k][0].strip() for k in keys if meta.get(k) and meta[k][0].strip()), None)


_ARTICLE_TYPES = {"Article", "NewsArticle", "BlogPosting", "ScholarlyArticle"}


def _ld_values(blobs: list[str]) -> tuple[list[str], str | None]:
    """Authors and a publication date from JSON-LD ``Article`` records, if there are any."""
    authors: list[str] = []
    published: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kind = node.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(k in _ARTICLE_TYPES for k in kinds):
                if isinstance(node.get("datePublished"), str):
                    published.append(node["datePublished"])
                person = node.get("author")
                for p in person if isinstance(person, list) else [person]:
                    name = p.get("name") if isinstance(p, dict) else p
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())
            for value in node.values():
                if isinstance(value, list | dict):
                    walk(value)

    for blob in blobs:
        try:
            walk(json.loads(blob))
        except ValueError:
            continue
    return list(dict.fromkeys(authors))[:20], (published[0] if published else None)


def _assemble(blocks: list[tuple[int, str]]) -> tuple[str, list[dict[str, Any]], bool]:
    """Blocks of ``(heading level or 0, text)`` as one text and the sections it falls into.

    Sections are offsets into that text, the shape the paper pipeline's chunking reads. With
    no heading there is one section, the whole text.
    """
    parts: list[str] = []
    sections: list[dict[str, Any]] = []
    cursor = 0
    truncated = False
    current: dict[str, Any] | None = None
    for level, text in blocks:
        if cursor + len(text) > MAX_TEXT_CHARS:
            truncated = True
            break
        if level:
            if current is not None:
                current["end"] = cursor
            current = {
                "id": f"s{len(sections) + 1}",
                "heading": text[:120],
                "level": level,
                "start": cursor,
                "end": cursor,
                "page_start": 1,
                "page_end": 1,
            }
            sections.append(current)
        parts.append(text)
        cursor += len(text) + 2
    if current is not None:
        current["end"] = cursor
    text = "\n\n".join(parts)
    if text and not sections:
        sections = [_section("s1", "Document", 0, len(text))]
    elif sections and sections[0]["start"] > 200:
        # Text before the first heading (an introduction, a byline) is a section of its own.
        sections.insert(0, _section("s0", "Introduction", 0, sections[0]["start"]))
    return text, sections, truncated


def _section(ident: str, heading: str, start: int, end: int) -> dict[str, Any]:
    return {
        "id": ident,
        "heading": heading,
        "level": 1,
        "start": start,
        "end": end,
        "page_start": 1,
        "page_end": 1,
    }


def parse_html(html: str) -> ParsedPage:
    """The readable text and the self-described metadata of ``html``."""
    reader = _Reader()
    try:
        reader.feed(html[: 4 * MAX_TEXT_CHARS])
        reader.close()
        reader.finish()
    except Exception:  # noqa: BLE001 - markup this parser cannot read is a page it cannot read
        return ParsedPage()
    ld_authors, ld_date = _ld_values(reader.json_ld)
    meta = reader.meta
    meta_authors = [a.strip() for a in meta.get("citation_author", []) + meta.get("author", [])]
    text, sections, truncated = _assemble(reader.blocks)
    title = _first(meta, "citation_title", "og:title") or " ".join(reader.title.split())
    published = (
        _first(meta, "citation_publication_date", "article:published_time", "dc.date", "date")
        or ld_date
    )
    return ParsedPage(
        title=title[:300],
        authors=list(dict.fromkeys(a for a in meta_authors + ld_authors if a))[:20],
        published=published,
        site_name=_first(meta, "og:site_name"),
        description=_first(meta, "description", "og:description"),
        canonical=reader.canonical,
        text=text,
        sections=sections,
        truncated=truncated,
    )


def parse_markdown(markdown: str) -> ParsedPage:
    """Text an extraction service returned as Markdown: its ``#`` headings become sections.

    Such a service reports no author or date, and none is invented.
    """
    blocks: list[tuple[int, str]] = []
    title = ""
    for raw in re.split(r"\n{2,}", markdown):
        block = raw.strip()
        if not block:
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", block)
        if heading:
            level, label = len(heading.group(1)), heading.group(2).strip()
            title = title or (label if level == 1 else "")
            blocks.append((level, label))
        else:
            blocks.append((0, block))
    text, sections, truncated = _assemble(blocks)
    return ParsedPage(title=title[:300], text=text, sections=sections, truncated=truncated)


_RESTRICTED = re.compile(
    "|".join(
        [
            r"subscribe to (continue|read)",
            r"sign in to (read|continue|view)",
            r"log ?in to (read|view|continue)",
            r"create (a )?(free )?account to (read|continue)",
            r"members[- ]only",
            r"this content is (for|available to) (subscribers|members)",
            r"verify you are (a )?human",
            r"are you a robot",
            r"captcha",
            r"enable javascript",
            r"access denied",
            r"unusual traffic",
        ]
    ),
    re.IGNORECASE,
)
MIN_USEFUL_CHARS = 600


def restriction(text: str) -> str | None:
    """Why a page's text is not the article, or ``None`` when it looks like one.

    A paywall, a login or bot check, or a stub. Recognised only to *stop*: nothing here tries
    to get past it.
    """
    body = text.strip()
    if _RESTRICTED.search(body[:4000]) and len(body) < 6000:
        return "The page asks for a subscription, a login or a human check, so it was not read."
    if len(body) < MIN_USEFUL_CHARS:
        return "The page holds too little text to be an article."
    return None


_INJECTION = re.compile(
    "|".join(
        [
            r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
            r"disregard (all |the )?(previous|above|prior)",
            r"you are now\b",
            r"new instructions:",
            r"system prompt",
            r"reveal your (instructions|prompt)",
            r"as an ai (language )?model",
            r"do not (tell|inform) the user",
            r"\bapi[_ -]?key\b.{0,40}(send|post|output|print)",
        ]
    ),
    re.IGNORECASE,
)


def injection_markers(text: str) -> list[str]:
    """Phrases that look like instructions to an assistant. The text is read as data either
    way; finding them only lets the person know the page was trying."""
    return list(dict.fromkeys(m.group(0).lower() for m in _INJECTION.finditer(text)))[:5]
