"""Where a paper's metadata and open full text come from: Crossref, OpenAlex, arXiv and,
optionally, Semantic Scholar.

Each adapter answers with :class:`PaperMeta`: only what that registry said, nothing completed
from memory. ``pdf_url`` is set only when the registry names a copy it says is open, and
``access`` says how much of the paper that gives: ``abstract_only`` when there is an abstract
but no open copy, ``metadata_only`` when there is neither. An abstract is never promoted to
"the paper".

The registries are public, anonymous APIs. They are read with a fixed client name and short
timeouts, and a failure of one is a reason to try the next, not a failure of the import.
"""

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

import httpx2

from .identifiers import normalise_arxiv, normalise_doi

USER_AGENT = "AlphaHarness-ResearchImporter/1.0 (local research tool)"
TIMEOUT = httpx2.Timeout(15.0, connect=8.0)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

_TAG = re.compile(r"<[^>]+>")
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ScholarlyError(Exception):
    """A registry that could not answer: ``code`` is ``not_found``, ``unavailable`` or
    ``bad_response``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class PaperMeta:
    provider: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    published: str | None = None
    venue: str | None = None
    topics: list[str] = field(default_factory=list)
    license: str | None = None
    #: A copy the registry says is openly readable.
    pdf_url: str | None = None
    #: Where ``pdf_url`` came from, in words, for the access note.
    pdf_note: str = ""
    citations: int | None = None

    @property
    def access(self) -> str:
        if self.pdf_url:
            return "full_text"
        return "abstract_only" if self.abstract else "metadata_only"


def clean_text(value: str | None) -> str | None:
    """Abstract markup (Crossref sends JATS) turned into plain text."""
    if not value:
        return None
    text = html.unescape(_TAG.sub(" ", value))
    text = re.sub(r"^\s*abstract[:.\s]+", "", " ".join(text.split()), flags=re.IGNORECASE)
    return text or None


def _date(parts: Any) -> str | None:
    """``[[2019, 5, 17]]`` as ``2019-05-17``, as far as the registry went."""
    try:
        numbers = [int(n) for n in parts["date-parts"][0] if n is not None]
    except KeyError, IndexError, TypeError, ValueError:
        return None
    if not numbers:
        return None
    return "-".join(f"{n:04d}" if i == 0 else f"{n:02d}" for i, n in enumerate(numbers[:3]))


async def _get_json(
    client: httpx2.AsyncClient, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        response = await client.get(url, params=params)
    except httpx2.TimeoutException as exc:
        raise ScholarlyError("unavailable", "The registry did not answer in time.") from exc
    except httpx2.TransportError as exc:
        raise ScholarlyError("unavailable", "The registry could not be reached.") from exc
    if response.status_code == 404:
        raise ScholarlyError("not_found", "The registry has no record of that.")
    if response.status_code == 429:
        raise ScholarlyError(
            "unavailable", "The registry is rate-limiting requests; try again shortly."
        )
    if response.status_code >= 400:
        raise ScholarlyError(
            "unavailable", f"The registry answered with an error ({response.status_code})."
        )
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ScholarlyError("bad_response", "The registry's answer was unexpectedly large.")
    try:
        body = response.json()
    except ValueError as exc:
        raise ScholarlyError("bad_response", "The registry did not answer with JSON.") from exc
    if not isinstance(body, dict):
        raise ScholarlyError("bad_response", "The registry's answer had an unexpected shape.")
    return body


class Scholarly:
    """The four registries behind one object. ``transport`` is for tests."""

    def __init__(
        self,
        *,
        contact: str | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        agent = f"{USER_AGENT} mailto:{contact}" if contact else USER_AGENT
        self._http = httpx2.AsyncClient(
            timeout=TIMEOUT,
            headers={"User-Agent": agent, "Accept": "application/json"},
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- Crossref ---------------------------------------------------------------------

    @staticmethod
    def _crossref(item: dict[str, Any]) -> PaperMeta:
        authors = [
            " ".join(p for p in (a.get("given"), a.get("family")) if p) or str(a.get("name") or "")
            for a in item.get("author", [])
            if isinstance(a, dict)
        ]
        licenses = [
            str(lic.get("URL"))
            for lic in item.get("license", [])
            if isinstance(lic, dict) and lic.get("URL")
        ]
        titles = item.get("title") or []
        venues = item.get("container-title") or []
        meta = PaperMeta(
            provider="crossref",
            title=clean_text(titles[0]) or "" if titles else "",
            authors=[a for a in authors if a],
            abstract=clean_text(item.get("abstract")),
            doi=normalise_doi(str(item.get("DOI") or "")),
            url=item.get("URL"),
            published=_date(item.get("issued") or item.get("published")),
            venue=venues[0] if venues else None,
            topics=[str(s) for s in item.get("subject", [])][:8],
            license=licenses[0] if licenses else None,
            citations=item.get("is-referenced-by-count"),
        )
        # Crossref only names a text-mining link, not a promise it is open; it is not used.
        return meta  # noqa: RET504

    async def crossref(self, doi: str) -> PaperMeta:
        body = await _get_json(self._http, f"https://api.crossref.org/works/{quote(doi, safe='/')}")
        item = body.get("message")
        if not isinstance(item, dict):
            raise ScholarlyError("bad_response", "Crossref's answer had an unexpected shape.")
        return self._crossref(item)

    # --- OpenAlex ---------------------------------------------------------------------

    @staticmethod
    def _abstract(index: Any) -> str | None:
        """OpenAlex sends abstracts as ``{word: [positions]}``; put them back in order."""
        if not isinstance(index, dict) or not index:
            return None
        words: dict[int, str] = {}
        for word, positions in index.items():
            if isinstance(positions, list):
                for position in positions:
                    if isinstance(position, int):
                        words[position] = str(word)
        return " ".join(words[i] for i in sorted(words)) or None

    @classmethod
    def _openalex(cls, work: dict[str, Any]) -> PaperMeta:
        best = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        access = work.get("open_access") or {}
        pdf = best.get("pdf_url") if work.get("open_access", {}).get("is_oa") else None
        meta = PaperMeta(
            provider="openalex",
            title=str(work.get("title") or work.get("display_name") or ""),
            authors=[
                str((a.get("author") or {}).get("display_name") or "")
                for a in work.get("authorships", [])
                if isinstance(a, dict) and (a.get("author") or {}).get("display_name")
            ],
            abstract=cls._abstract(work.get("abstract_inverted_index")),
            doi=normalise_doi(str(work.get("doi") or "")),
            url=best.get("landing_page_url") or primary.get("landing_page_url"),
            published=work.get("publication_date"),
            venue=source.get("display_name"),
            topics=[
                str(t.get("display_name"))
                for t in (work.get("topics") or work.get("concepts") or [])[:6]
                if isinstance(t, dict) and t.get("display_name")
            ],
            license=best.get("license"),
            pdf_url=pdf,
            pdf_note=f"open-access copy listed by OpenAlex ({access.get('oa_status', 'open')})"
            if pdf
            else "",
            citations=work.get("cited_by_count"),
        )
        for location in work.get("locations") or []:
            if isinstance(location, dict) and "arxiv.org" in str(
                location.get("landing_page_url") or ""
            ):
                meta.arxiv_id = normalise_arxiv(str(location["landing_page_url"]))
        return meta

    async def openalex(self, doi: str) -> PaperMeta:
        body = await _get_json(
            self._http, f"https://api.openalex.org/works/doi:{quote(doi, safe='/')}"
        )
        return self._openalex(body)

    async def search_openalex(self, query: str, *, limit: int = 10) -> list[PaperMeta]:
        body = await _get_json(
            self._http,
            "https://api.openalex.org/works",
            {"search": query, "per-page": min(limit, 25), "filter": "type:article|preprint"},
        )
        return [self._openalex(w) for w in body.get("results", []) if isinstance(w, dict)]

    # --- arXiv ------------------------------------------------------------------------

    @staticmethod
    def parse_arxiv(xml: str) -> list[PaperMeta]:
        """The entries of an arXiv Atom answer. A document that declares entities or a doctype
        is refused: arXiv sends neither, and expansion tricks need one."""
        head = xml[:2048].upper()
        if "<!DOCTYPE" in head or "<!ENTITY" in xml.upper()[:20_000]:
            raise ScholarlyError("bad_response", "arXiv's answer contained unexpected markup.")
        try:
            root = ET.fromstring(xml)  # noqa: S314 - doctype and entities refused above
        except ET.ParseError as exc:
            raise ScholarlyError("bad_response", "arXiv's answer could not be read.") from exc
        out: list[PaperMeta] = []
        for entry in root.findall(f"{_ATOM}entry"):
            identity = entry.findtext(f"{_ATOM}id") or ""
            arxiv_id = normalise_arxiv(identity)
            title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            if arxiv_id is None or not title or title.lower() == "error":
                continue
            doi = entry.findtext(f"{_ARXIV_NS}doi")
            out.append(
                PaperMeta(
                    provider="arxiv",
                    title=title,
                    authors=[
                        " ".join((a.findtext(f"{_ATOM}name") or "").split())
                        for a in entry.findall(f"{_ATOM}author")
                    ],
                    abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()) or None,
                    doi=normalise_doi(doi) if doi else None,
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    published=(entry.findtext(f"{_ATOM}published") or "")[:10] or None,
                    venue=entry.findtext(f"{_ARXIV_NS}journal_ref"),
                    topics=[
                        str(c.get("term"))
                        for c in entry.findall(f"{_ATOM}category")
                        if c.get("term")
                    ][:6],
                    # arXiv hosts the author's own copy; the licence is the author's choice.
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    pdf_note="author's copy hosted by arXiv",
                )
            )
        return out

    async def _arxiv_query(self, **params: Any) -> list[PaperMeta]:
        try:
            response = await self._http.get(
                f"https://export.arxiv.org/api/query?{urlencode(params)}",
                headers={"Accept": "application/atom+xml"},
            )
        except httpx2.TimeoutException as exc:
            raise ScholarlyError("unavailable", "arXiv did not answer in time.") from exc
        except httpx2.TransportError as exc:
            raise ScholarlyError("unavailable", "arXiv could not be reached.") from exc
        if response.status_code >= 400:
            raise ScholarlyError(
                "unavailable", f"arXiv answered with an error ({response.status_code})."
            )
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise ScholarlyError("bad_response", "arXiv's answer was unexpectedly large.")
        return self.parse_arxiv(response.text)

    async def arxiv(self, arxiv_id: str) -> PaperMeta:
        found = await self._arxiv_query(id_list=arxiv_id, max_results=1)
        if not found:
            raise ScholarlyError("not_found", f"arXiv has no paper {arxiv_id}.")
        return found[0]

    async def search_arxiv(self, query: str, *, limit: int = 10) -> list[PaperMeta]:
        return await self._arxiv_query(
            search_query=f"all:{query}", max_results=min(limit, 25), sortBy="relevance"
        )

    # --- Semantic Scholar (optional) --------------------------------------------------

    @staticmethod
    def _semantic(paper: dict[str, Any]) -> PaperMeta:
        ids = paper.get("externalIds") or {}
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        return PaperMeta(
            provider="semantic_scholar",
            title=str(paper.get("title") or ""),
            authors=[str(a.get("name")) for a in paper.get("authors", []) if a.get("name")],
            abstract=paper.get("abstract"),
            doi=normalise_doi(str(ids.get("DOI") or "")),
            arxiv_id=normalise_arxiv(str(ids.get("ArXiv") or "")),
            url=paper.get("url"),
            published=paper.get("publicationDate")
            or (str(paper["year"]) if paper.get("year") else None),
            venue=paper.get("venue") or None,
            topics=[str(f) for f in paper.get("fieldsOfStudy") or []][:6],
            pdf_url=pdf,
            pdf_note="open-access copy listed by Semantic Scholar" if pdf else "",
            citations=paper.get("citationCount"),
        )

    _SEMANTIC_FIELDS = (
        "title,authors,abstract,year,venue,publicationDate,externalIds,openAccessPdf,"
        "fieldsOfStudy,citationCount,url"
    )

    async def semantic_scholar(self, doi: str) -> PaperMeta:
        body = await _get_json(
            self._http,
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='/')}",
            {"fields": self._SEMANTIC_FIELDS},
        )
        return self._semantic(body)

    async def search_semantic_scholar(self, query: str, *, limit: int = 10) -> list[PaperMeta]:
        body = await _get_json(
            self._http,
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": query, "limit": min(limit, 25), "fields": self._SEMANTIC_FIELDS},
        )
        return [self._semantic(p) for p in body.get("data", []) if isinstance(p, dict)]

    # --- combined ---------------------------------------------------------------------

    async def by_doi(self, doi: str) -> PaperMeta:
        """Crossref for the record, OpenAlex for the abstract and an open copy, each filling
        what the other lacks. A registry that fails is skipped; all failing is an error."""
        found: list[PaperMeta] = []
        errors: list[ScholarlyError] = []
        for lookup in (self.crossref, self.openalex):
            try:
                found.append(await lookup(doi))
            except ScholarlyError as exc:
                errors.append(exc)
        if not found:
            codes = {e.code for e in errors}
            if codes == {"not_found"}:
                raise ScholarlyError("not_found", f"No registry has a record of DOI {doi}.")
            raise ScholarlyError("unavailable", "The metadata registries could not be reached.")
        return merge(found, doi=doi)

    async def search(
        self, query: str, *, limit: int = 10, semantic: bool = False
    ) -> list[PaperMeta]:
        """Keyword search: OpenAlex, then arXiv, then Semantic Scholar if asked. Results that
        are the same paper are one result."""
        results: list[PaperMeta] = []
        errors = 0
        searches = [self.search_openalex, self.search_arxiv]
        if semantic:
            searches.append(self.search_semantic_scholar)
        for lookup in searches:
            try:
                results.extend(await lookup(query, limit=limit))
            except ScholarlyError:
                errors += 1
        if errors == len(searches):
            raise ScholarlyError(
                "unavailable", "The academic search services could not be reached."
            )
        return dedupe(results)[:limit]


def merge(found: list[PaperMeta], *, doi: str | None = None) -> PaperMeta:
    """One record from several: the first non-empty value of each field, in the order given."""
    out = PaperMeta(provider="+".join(m.provider for m in found), doi=doi)
    for meta in found:
        out.title = out.title or meta.title
        out.authors = out.authors or meta.authors
        out.abstract = out.abstract or meta.abstract
        out.arxiv_id = out.arxiv_id or meta.arxiv_id
        out.url = out.url or meta.url
        out.published = out.published or meta.published
        out.venue = out.venue or meta.venue
        out.topics = out.topics or meta.topics
        out.license = out.license or meta.license
        if meta.pdf_url and not out.pdf_url:
            out.pdf_url, out.pdf_note = meta.pdf_url, meta.pdf_note
        out.citations = out.citations if out.citations is not None else meta.citations
    return out


def _key(meta: PaperMeta) -> str:
    if meta.doi:
        return f"doi:{meta.doi}"
    if meta.arxiv_id:
        return f"arxiv:{meta.arxiv_id}"
    return "title:" + re.sub(r"[^a-z0-9]+", "", meta.title.lower())


def dedupe(results: list[PaperMeta]) -> list[PaperMeta]:
    """Results that share a DOI, an arXiv id or a title are merged into the first."""
    groups: dict[str, list[PaperMeta]] = {}
    for meta in results:
        if meta.title:
            groups.setdefault(_key(meta), []).append(meta)
    # A preprint and its journal version differ in id but share a title: fold those together.
    by_title: dict[str, str] = {}
    folded: dict[str, list[PaperMeta]] = {}
    for key, items in groups.items():
        title = "title:" + re.sub(r"[^a-z0-9]+", "", items[0].title.lower())
        target = by_title.setdefault(title, key)
        folded.setdefault(target, []).extend(items)
    return [
        merge(items, doi=next((i.doi for i in items if i.doi), None)) for items in folded.values()
    ]
