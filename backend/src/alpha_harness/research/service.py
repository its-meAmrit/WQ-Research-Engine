"""The paper pipeline: import, parse, analyse, assess, hand off.

Every step that takes time is a :class:`~alpha_harness.db.models.ResearchJob` row written
before the work starts and updated as it goes. So a screen follows a job by reading it, a
restart can say where work stopped, and a finished analysis is still there afterwards.

Nothing here starts a simulation, calls BRAIN, or submits anything. The only network
reads are the public scholarly registries, the open full-text copy a registry names, a link
the person typed (through :mod:`.netfetch`), and, when configured, Tavily for supplementary
reading. The only model calls are the analysis's own.
"""

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from ..db.models import (
    AlphaCandidate,
    ResearchDocument,
    ResearchIdea,
    ResearchJob,
    ResearchJobStatus,
    ResearchSource,
    utcnow,
)
from ..integrations.base import IntegrationError, safe_public_url
from ..llm.keys import LLMError
from ..tasks import spawn
from . import extraction, feasibility, netfetch, pdf
from .identifiers import IdentifierError, canonical_url, classify, normalise_arxiv, normalise_doi
from .scholarly import PaperMeta, Scholarly, ScholarlyError

if TYPE_CHECKING:
    from ..catalog.queries import Tuple4
    from ..db.sqlite import Database
    from ..tasks import Task, TaskRegistry

log = structlog.get_logger(__name__)

CONCURRENCY = 2
TEMPERATURE = 0.1
MAX_NOTE_CHARS = 5_000
SEARCH_LIMIT = 10
MAX_HTML_BYTES = 3 * 1024 * 1024

Operators = Callable[[], Awaitable[list[dict[str, Any]] | None]]
ScholarlyFactory = Callable[[], Scholarly]


class ResearchError(Exception):
    """A request this service refuses, with a stable ``code`` and a ``status`` for HTTP."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class _HeadParser(HTMLParser):
    """The ``<meta>`` tags of a page's head, which publishers fill with ``citation_*``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.title = ""
        self._in_title = False
        self._done = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            fields = {k.lower(): v for k, v in attrs if v is not None}
            key = (fields.get("name") or fields.get("property") or "").lower()
            if key and fields.get("content"):
                self.meta.setdefault(key, []).append(fields["content"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self._done = True

    def handle_data(self, data: str) -> None:
        if self._in_title and not self._done:
            self.title += data


def page_metadata(html: str, url: str) -> PaperMeta | None:
    """What a publisher's landing page says about its paper, from its own ``citation_*`` tags.

    Only what the tags say. A page without them yields nothing, and the body text is never
    read as the paper: a web page is not a paper unless it tells us which one it is.
    """
    parser = _HeadParser()
    try:
        parser.feed(html[:400_000])
    except Exception:  # noqa: BLE001 - malformed markup is a page we cannot read, not a crash
        return None
    meta = parser.meta

    def first(*keys: str) -> str | None:
        return next((meta[k][0].strip() for k in keys if meta.get(k)), None)

    title = first("citation_title", "dc.title", "og:title")
    if not title:
        return None
    pdf_url = first("citation_pdf_url")
    return PaperMeta(
        provider="page",
        title=" ".join(title.split()),
        authors=[
            a.strip()
            for a in meta.get("citation_author", []) + meta.get("dc.creator", [])
            if a.strip()
        ][:30],
        abstract=first("citation_abstract", "dc.description", "og:description", "description"),
        doi=normalise_doi(first("citation_doi", "dc.identifier") or ""),
        arxiv_id=normalise_arxiv(first("citation_arxiv_id") or ""),
        url=url,
        published=first("citation_publication_date", "citation_date", "dc.date"),
        venue=first("citation_journal_title", "citation_conference_title"),
        pdf_url=pdf_url,
        pdf_note="the page's own citation_pdf_url" if pdf_url else "",
    )


@dataclass(slots=True)
class Started:
    """The answer to an import or an analysis request."""

    source_id: str | None
    job_id: str | None
    #: ``True`` when the paper (or the analysis) was already there and nothing was started.
    existing: bool
    message: str


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ResearchService:
    def __init__(
        self,
        db: Database,
        llm: Any,
        *,
        tasks: TaskRegistry | None = None,
        queries: Any = None,
        operators: Operators | None = None,
        web: Any = None,
        scholarly: ScholarlyFactory | None = None,
        fetch: Callable[..., Awaitable[netfetch.Fetched]] = netfetch.fetch,
        limits: pdf.Limits | None = None,
        contact: str | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.tasks = tasks
        self.queries = queries
        self._operators = operators
        self._web = web
        self._scholarly = scholarly or (lambda: Scholarly(contact=contact))
        self._fetch = fetch
        self.limits = limits or pdf.Limits()
        self._parse_gate = asyncio.Semaphore(CONCURRENCY)
        self._llm_gate = asyncio.Semaphore(CONCURRENCY)
        self._jobs: dict[str, asyncio.Task[None]] = {}
        self._why: dict[str, str] = {}
        self._stopping = False

    # --- lifecycle ----------------------------------------------------------------------

    async def reconcile(self) -> int:
        """Settle jobs a previous process left in flight: they died with it, so each is marked
        ``interrupted`` rather than left claiming to run. Finished work is untouched."""
        async with self.db.session() as session:
            result = await session.execute(
                update(ResearchJob)
                .where(
                    ResearchJob.status.in_(
                        [ResearchJobStatus.QUEUED.value, ResearchJobStatus.RUNNING.value]
                    )
                )
                .values(
                    status=ResearchJobStatus.INTERRUPTED.value,
                    error_category="interrupted",
                    error_message="Alpha Harness stopped while this job was in progress. "
                    "Nothing from it was saved; start it again.",
                    message="Interrupted by a restart.",
                    finished_at=utcnow(),
                )
            )
            count = result.rowcount or 0  # pyright: ignore[reportAttributeAccessIssue]
            # A source whose import never finished says so, and is not left "pending".
            await session.execute(
                update(ResearchSource)
                .where(ResearchSource.extraction_status == "pending")
                .where(~ResearchSource.id.in_(select(ResearchDocument.source_id)))
                .values(extraction_status="failed")
            )
        if count:
            log.warning("research.jobs_interrupted", count=count)
        return count

    async def stop(self) -> None:
        self._stopping = True
        for job_id, task in list(self._jobs.items()):
            self._why[job_id] = "Stopped because Alpha Harness was closing."
            task.cancel()
        await asyncio.gather(*self._jobs.values(), return_exceptions=True)

    # --- jobs ---------------------------------------------------------------------------

    async def _new_job(self, **values: Any) -> str:
        async with self.db.session() as session:
            job = ResearchJob(**values)
            session.add(job)
            await session.flush()
            return job.id

    async def _set_job(self, job_id: str, **values: Any) -> None:
        async with self.db.session() as session:
            await session.execute(
                update(ResearchJob).where(ResearchJob.id == job_id).values(**values)
            )

    def _launch(self, job_id: str, label: str, work: Callable[[str], Awaitable[None]]) -> None:
        task = spawn(self._execute(job_id, label, work), name=f"research-{job_id}")
        self._jobs[job_id] = task  # pyright: ignore[reportArgumentType]
        task.add_done_callback(lambda _t: self._jobs.pop(job_id, None))

    async def _execute(
        self, job_id: str, label: str, work: Callable[[str], Awaitable[None]]
    ) -> None:
        """Run ``work`` for the job and settle the row whichever way it ends."""
        registry_task: Task | None = None
        if self.tasks is not None:
            registry_task = await self.tasks.start("research", label, job_id=job_id)
        state, error = "done", None
        await self._set_job(job_id, status="running", started_at=utcnow())
        try:
            await work(job_id)
        except asyncio.CancelledError:
            state = "cancelled"
            if self._stopping:
                values = {
                    "status": "interrupted",
                    "error_category": "interrupted",
                    "message": self._why.pop(job_id, "Stopped."),
                }
            else:
                values = {"status": "cancelled", "message": self._why.pop(job_id, "Cancelled.")}
            await asyncio.shield(self._set_job(job_id, finished_at=utcnow(), **values))
            raise
        except ResearchError as exc:
            state, error = "failed", str(exc)
            await self._fail(job_id, exc.code, str(exc))
        except Exception as exc:
            log.exception("research.job_crashed", job_id=job_id)
            state, error = "failed", f"{type(exc).__name__}"
            await self._fail(
                job_id,
                "internal_error",
                "Something went wrong inside Alpha Harness. Nothing was saved.",
            )
        finally:
            if registry_task is not None and self.tasks is not None:
                await self.tasks.finish(registry_task, state=state, error=error)

    async def _fail(
        self,
        job_id: str,
        code: str,
        message: str,
        *,
        detail: list[dict[str, Any]] | None = None,
        raw: str | None = None,
    ) -> None:
        await self._set_job(
            job_id,
            status=ResearchJobStatus.FAILED.value,
            error_category=code,
            error_message=message,
            error_detail=detail,
            raw_output=raw,
            message="Failed.",
            finished_at=utcnow(),
        )

    async def _progress(self, job_id: str, progress: float, message: str) -> None:
        await self._set_job(job_id, progress=progress, message=message)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = await self._job_row(job_id)
        if ResearchJobStatus(job.status).terminal:
            raise ResearchError("job_finished", f"That job already {job.status}.", 409)
        await self._set_job(job_id, cancel_requested=True)
        task = self._jobs.get(job_id)
        if task is not None and not task.done():
            self._why[job_id] = "Cancelled."
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            # Nothing here runs it: an earlier process did. Settle the row.
            await self._set_job(
                job_id, status="cancelled", message="Cancelled.", finished_at=utcnow()
            )
        return await self.get_job(job_id)

    async def _job_row(self, job_id: str) -> ResearchJob:
        async with self.db.session() as session:
            job = await session.get(ResearchJob, job_id)
        if job is None:
            raise ResearchError("job_not_found", "There is no such job.", 404)
        return job

    @staticmethod
    def job_view(job: ResearchJob, *, active: bool = False) -> dict[str, Any]:
        return {
            "id": job.id,
            "kind": job.kind,
            "sourceId": job.source_id,
            "inputRef": job.input_ref,
            "status": job.status,
            "active": active,
            "progress": job.progress,
            "message": job.message,
            "errorCategory": job.error_category,
            "errorMessage": job.error_message,
            "errorDetail": job.error_detail,
            "modelRef": job.model_ref,
            "provider": job.provider,
            "modelId": job.model_id,
            "params": job.params,
            "usage": job.usage,
            "result": job.result,
            "rawOutput": job.raw_output,
            "cancelRequested": job.cancel_requested,
            "createdAt": _iso(job.created_at),
            "startedAt": _iso(job.started_at),
            "finishedAt": _iso(job.finished_at),
        }

    async def get_job(self, job_id: str) -> dict[str, Any]:
        job = await self._job_row(job_id)
        return self.job_view(job, active=job_id in self._jobs)

    async def list_jobs(
        self, *, source_id: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        stmt = select(ResearchJob).order_by(ResearchJob.created_at.desc()).limit(limit)
        if source_id:
            stmt = stmt.where(ResearchJob.source_id == source_id)
        async with self.db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [self.job_view(j, active=j.id in self._jobs) for j in rows]

    # --- sources: reading ---------------------------------------------------------------

    @staticmethod
    def source_view(
        source: ResearchSource, *, ideas: int = 0, saved: int = 0, has_text: bool = False
    ) -> dict[str, Any]:
        return {
            "id": source.id,
            "sourceType": source.source_type,
            "importMethod": source.import_method,
            "title": source.title,
            "authors": source.authors,
            "abstract": source.abstract,
            "canonicalUrl": source.canonical_url,
            "doi": source.doi,
            "arxivId": source.arxiv_id,
            "publishedAt": source.published_at,
            "venue": source.venue,
            "topics": source.topics,
            "retrievedAt": _iso(source.retrieved_at),
            "accessStatus": source.access_status,
            "accessDetail": source.access_detail,
            "license": source.license,
            "contentHash": source.content_hash,
            "extractionStatus": source.extraction_status,
            "pageCount": source.page_count,
            "metadata": source.metadata_,
            "provenance": source.provenance,
            "ideaCount": ideas,
            "savedIdeaCount": saved,
            "hasText": has_text,
            "createdAt": _iso(source.created_at),
        }

    async def list_sources(
        self,
        *,
        search: str | None = None,
        access: str | None = None,
        topic: str | None = None,
        readiness: str | None = None,
        saved: bool | None = None,
        imported_from: str | None = None,
        published_from: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        stmt = select(ResearchSource).order_by(ResearchSource.created_at.desc()).limit(limit)
        if search:
            like = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(ResearchSource.title).like(like),
                    func.lower(func.coalesce(ResearchSource.doi, "")).like(like),
                    func.lower(func.cast(ResearchSource.authors, extraction_text_type())).like(
                        like
                    ),
                )
            )
        if access:
            stmt = stmt.where(ResearchSource.access_status == access)
        if imported_from:
            stmt = stmt.where(ResearchSource.import_method == imported_from)
        if published_from:
            stmt = stmt.where(func.coalesce(ResearchSource.published_at, "") >= published_from)
        idea_filters = [ResearchIdea.source_id == ResearchSource.id]
        if topic:
            idea_filters.append(ResearchIdea.topic == topic)
        if readiness:
            idea_filters.append(ResearchIdea.readiness == readiness)
        if saved is True:
            idea_filters.append(ResearchIdea.status == "saved")
        if len(idea_filters) > 1:
            stmt = stmt.where(select(ResearchIdea.id).where(*idea_filters).exists())
        async with self.db.session() as session:
            sources = (await session.execute(stmt)).scalars().all()
            counts = {
                row[0]: (row[1], row[2])
                for row in (
                    await session.execute(
                        select(
                            ResearchIdea.source_id,
                            func.count(ResearchIdea.id),
                            func.sum(func.iif(ResearchIdea.status == "saved", 1, 0)),
                        ).group_by(ResearchIdea.source_id)
                    )
                ).all()
            }
            with_text = set(
                (await session.execute(select(ResearchDocument.source_id))).scalars().all()
            )
        return [
            self.source_view(
                s,
                ideas=int(counts.get(s.id, (0, 0))[0] or 0),
                saved=int(counts.get(s.id, (0, 0))[1] or 0),
                has_text=s.id in with_text,
            )
            for s in sources
        ]

    async def get_source(self, source_id: str) -> dict[str, Any]:
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            if source is None:
                raise ResearchError("source_not_found", "There is no such paper.", 404)
            document = await session.get(ResearchDocument, source_id)
            ideas = (
                await session.execute(
                    select(
                        func.count(ResearchIdea.id),
                        func.sum(func.iif(ResearchIdea.status == "saved", 1, 0)),
                    ).where(ResearchIdea.source_id == source_id)
                )
            ).one()
        view = self.source_view(
            source,
            ideas=int(ideas[0] or 0),
            saved=int(ideas[1] or 0),
            has_text=document is not None,
        )
        view["document"] = (
            {
                "pages": len(document.pages),
                "sections": [
                    {k: s[k] for k in ("id", "heading", "level", "page_start", "page_end")}
                    for s in document.sections
                ],
                "tables": document.tables,
                "equations": document.equations,
                "stats": document.stats,
                "parser": document.parser,
            }
            if document
            else None
        )
        return view

    async def get_pages(self, source_id: str, page: int) -> dict[str, Any]:
        """One page of stored text, so evidence can be shown where it was found."""
        async with self.db.session() as session:
            document = await session.get(ResearchDocument, source_id)
        if document is None:
            raise ResearchError("no_text", "No text is stored for this paper.", 404)
        for item in document.pages:
            if int(item["page"]) == page:
                return {"page": page, "text": item["text"], "pages": len(document.pages)}
        raise ResearchError("page_not_found", "That page is not in the stored text.", 404)

    async def delete_source(self, source_id: str) -> None:
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            if source is None:
                raise ResearchError("source_not_found", "There is no such paper.", 404)
            running = [j for j, t in self._jobs.items() if not t.done()]
            busy = (
                await session.execute(
                    select(ResearchJob.id).where(
                        ResearchJob.source_id == source_id,
                        ResearchJob.id.in_(running or [""]),
                    )
                )
            ).first()
            if busy:
                raise ResearchError(
                    "job_running", "Cancel the running job on this paper first.", 409
                )
            await session.execute(delete(ResearchIdea).where(ResearchIdea.source_id == source_id))
            await session.execute(delete(ResearchJob).where(ResearchJob.source_id == source_id))
            await session.execute(
                delete(ResearchDocument).where(ResearchDocument.source_id == source_id)
            )
            await session.delete(source)

    # --- sources: finding duplicates ----------------------------------------------------

    async def _existing(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        url: str | None = None,
        content_hash: str | None = None,
    ) -> ResearchSource | None:
        clauses = []
        if doi:
            clauses.append(ResearchSource.doi == doi)
        if arxiv_id:
            clauses.append(ResearchSource.arxiv_id == arxiv_id)
        if url:
            clauses.append(ResearchSource.canonical_url == url)
        if content_hash:
            clauses.append(ResearchSource.content_hash == content_hash)
        if not clauses:
            return None
        async with self.db.session() as session:
            return (
                (await session.execute(select(ResearchSource).where(or_(*clauses))))
                .scalars()
                .first()
            )

    # --- import: a PDF ------------------------------------------------------------------

    async def import_pdf(self, filename: str, data: bytes) -> Started:
        """Store an uploaded PDF's text. The same file twice is one paper."""
        try:
            pdf.validate_bytes(data, self.limits)
        except pdf.PdfError as exc:
            raise ResearchError(
                exc.code, str(exc), 413 if exc.code == "too_large" else 422
            ) from exc
        digest = hashlib.sha256(data).hexdigest()
        if (found := await self._existing(content_hash=digest)) is not None:
            return Started(found.id, None, True, "You have already imported this file.")
        name = re.sub(r"[^\w .()-]+", "_", filename or "paper.pdf")[:200]
        try:
            async with self.db.session() as session:
                source = ResearchSource(
                    import_method="upload",
                    title=name,
                    content_hash=digest,
                    access_status="full_text",
                    access_detail="Uploaded by you.",
                    extraction_status="pending",
                    provenance={"kind": "upload", "filename": name, "bytes": len(data)},
                )
                session.add(source)
                await session.flush()
                source_id = source.id
        except IntegrityError:
            found = await self._existing(content_hash=digest)
            if found is not None:
                return Started(found.id, None, True, "You have already imported this file.")
            raise
        job_id = await self._new_job(
            kind="import", source_id=source_id, input_ref=name, message="Queued."
        )

        async def work(job_id: str) -> None:
            await self._import_pdf_work(job_id, source_id, data)

        self._launch(job_id, f"Reading {name}", work)
        return Started(source_id, job_id, False, "Reading the PDF.")

    async def _store_parsed(self, source_id: str, parsed: pdf.ParsedPdf) -> None:
        async with self.db.session() as session:
            await session.merge(
                ResearchDocument(
                    source_id=source_id,
                    pages=parsed.pages,
                    sections=parsed.sections,
                    tables=parsed.tables,
                    equations=parsed.equations,
                    stats=parsed.stats | {"warnings": parsed.warnings},
                    parser=pdf.PARSER,
                )
            )

    async def _parse(self, data: bytes) -> pdf.ParsedPdf:
        async with self._parse_gate:
            # Off the event loop: PyMuPDF holds the thread for as long as the file takes.
            return await asyncio.to_thread(pdf.parse_pdf, data, self.limits)

    async def _import_pdf_work(self, job_id: str, source_id: str, data: bytes) -> None:
        await self._progress(job_id, 0.1, "Reading the PDF.")
        try:
            parsed = await self._parse(data)
        except pdf.PdfError as exc:
            await self._mark_source(
                source_id,
                extraction_status="unreadable" if exc.code == "no_text_layer" else "failed",
            )
            await self._fail(job_id, exc.code, str(exc))
            return
        await self._progress(job_id, 0.6, "Checking the paper's details.")
        found = parsed.metadata
        registry: PaperMeta | None = None
        reference = found.get("doi") or found.get("arxiv_id")
        if reference:
            registry = await self._lookup(found.get("doi"), found.get("arxiv_id"))
        updates = self._metadata_values(found, registry, from_pdf=True)
        updates["page_count"] = parsed.page_count
        updates["extraction_status"] = "partial" if parsed.partial else "parsed"
        duplicate = await self._existing(doi=updates.get("doi"), arxiv_id=updates.get("arxiv_id"))
        if duplicate is not None and duplicate.id != source_id:
            await self._merge_upload(job_id, source_id, duplicate, parsed, updates)
            return
        await self._store_parsed(source_id, parsed)
        await self._mark_source(source_id, **updates)
        warnings = parsed.warnings
        await self._set_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Read." + (f" {len(warnings)} warning(s)." if warnings else ""),
            result={
                "sourceId": source_id,
                "pages": parsed.page_count,
                "sections": len(parsed.sections),
                "warnings": warnings,
                "metadataVerified": registry is not None,
            },
            finished_at=utcnow(),
        )

    async def _merge_upload(
        self,
        job_id: str,
        new_id: str,
        existing: ResearchSource,
        parsed: pdf.ParsedPdf,
        updates: dict[str, Any],
    ) -> None:
        """The uploaded file turned out to be a paper already held (same DOI or arXiv id)."""
        gains_text = existing.access_status != "full_text"
        # The job points at the placeholder source; move it first or deleting the source
        # (ON DELETE CASCADE) would delete the job with it.
        await self._set_job(job_id, source_id=existing.id)
        async with self.db.session() as session:
            row = await session.get(ResearchSource, new_id)
            digest = row.content_hash if row else None
            await session.execute(delete(ResearchSource).where(ResearchSource.id == new_id))
        if gains_text:
            await self._store_parsed(existing.id, parsed)
            await self._mark_source(
                existing.id,
                access_status="full_text",
                access_detail="Uploaded by you, matched to this paper by its DOI/arXiv id.",
                extraction_status=updates["extraction_status"],
                page_count=parsed.page_count,
                content_hash=digest,
            )
        message = (
            "That paper was already in your library; the uploaded PDF now supplies its full text."
            if gains_text
            else "That paper was already in your library with its full text; nothing was "
            "duplicated."
        )
        await self._set_job(
            job_id,
            source_id=existing.id,
            status="completed",
            progress=1.0,
            message=message,
            result={
                "sourceId": existing.id,
                "duplicateOf": existing.id,
                "gainedFullText": gains_text,
            },
            finished_at=utcnow(),
        )

    async def _mark_source(self, source_id: str, **values: Any) -> None:
        async with self.db.session() as session:
            await session.execute(
                update(ResearchSource).where(ResearchSource.id == source_id).values(**values)
            )

    @staticmethod
    def _metadata_values(
        found: dict[str, Any], registry: PaperMeta | None, *, from_pdf: bool
    ) -> dict[str, Any]:
        """Source columns from a PDF's own guesses and a registry's record. The registry wins
        wherever it speaks, and says so in the provenance."""
        values: dict[str, Any] = {
            "title": (registry.title if registry and registry.title else found.get("title")) or "",
            "authors": (registry.authors if registry and registry.authors else found.get("authors"))
            or [],
            "abstract": (
                registry.abstract if registry and registry.abstract else found.get("abstract")
            ),
            "doi": (registry.doi if registry and registry.doi else found.get("doi")),
            "arxiv_id": (
                registry.arxiv_id if registry and registry.arxiv_id else found.get("arxiv_id")
            ),
        }
        if registry:
            values |= {
                "published_at": registry.published,
                "venue": registry.venue,
                "topics": registry.topics,
                "license": registry.license,
                "canonical_url": None,
                "provenance": {
                    "metadata": f"{registry.provider}, matched by the identifier in the PDF",
                    "retrievedAt": _now(),
                },
            }
            values.pop("canonical_url")
        elif from_pdf:
            values["provenance"] = {
                "metadata": "read from the PDF's own text and layout; not verified",
                "retrievedAt": _now(),
            }
        if not values["title"]:
            values.pop("title")
        return values

    async def _lookup(self, doi: str | None, arxiv_id: str | None) -> PaperMeta | None:
        """A registry's record for an identifier, or ``None`` if none will say."""
        client = self._scholarly()
        try:
            if doi:
                return await client.by_doi(doi)
            if arxiv_id:
                return await client.arxiv(arxiv_id)
        except ScholarlyError as exc:
            log.info("research.lookup_failed", code=exc.code)
        finally:
            await client.aclose()
        return None

    # --- import: a DOI, an arXiv id or a URL --------------------------------------------

    async def import_reference(self, raw: str) -> Started:
        try:
            ref = classify(raw)
        except IdentifierError as exc:
            raise ResearchError("bad_identifier", str(exc), 422) from exc
        found = await self._existing(
            doi=ref.value if ref.kind == "doi" else None,
            arxiv_id=ref.value if ref.kind == "arxiv" else None,
            url=ref.value if ref.kind == "url" else None,
        )
        if found is not None:
            return Started(found.id, None, True, "That paper is already in your library.")
        job_id = await self._new_job(kind="import", input_ref=ref.value, message="Queued.")

        async def work(job_id: str) -> None:
            await self._import_reference_work(job_id, ref.kind, ref.value)

        self._launch(job_id, f"Importing {ref.value[:60]}", work)
        return Started(None, job_id, False, "Looking the paper up.")

    async def _import_reference_work(self, job_id: str, kind: str, value: str) -> None:
        await self._progress(job_id, 0.1, "Looking the paper up.")
        meta: PaperMeta | None = None
        client = self._scholarly()
        try:
            if kind == "doi":
                meta = await client.by_doi(value)
            elif kind == "arxiv":
                meta = await client.arxiv(value)
        except ScholarlyError as exc:
            code = "not_found" if exc.code == "not_found" else "registry_unavailable"
            await self._fail(job_id, code, str(exc))
            return
        finally:
            await client.aclose()

        if kind == "url":
            meta, page_pdf = await self._read_url(job_id, value)
            if meta is None and page_pdf is None:
                return
        else:
            page_pdf = None
        await self._finish_import(job_id, kind, value, meta, page_pdf)

    async def _read_url(
        self, job_id: str, url: str
    ) -> tuple[PaperMeta | None, pdf.ParsedPdf | tuple[bytes, str] | None]:
        """A typed link: a PDF itself, or a page that says which paper it is."""
        try:
            fetched = await self._fetch(url, max_bytes=self.limits.max_bytes)
        except netfetch.FetchError as exc:
            await self._fail(job_id, exc.code, str(exc))
            return None, None
        if fetched.data[:1024].find(b"%PDF-") >= 0:
            return PaperMeta(provider="url", url=fetched.url), (fetched.data, fetched.url)
        if (
            fetched.content_type.startswith(("text/html", "application/xhtml"))
            or not fetched.content_type
        ):
            if len(fetched.data) > MAX_HTML_BYTES:
                await self._fail(job_id, "too_large", "That page is too large to read.")
                return None, None
            meta = page_metadata(fetched.data.decode("utf-8", errors="replace"), fetched.url)
            if meta is None:
                await self._fail(
                    job_id,
                    "not_a_paper",
                    "That page does not identify itself as a research paper (it has no "
                    "citation details), so nothing was imported. Use its DOI or upload the PDF.",
                )
                return None, None
            return meta, None
        await self._fail(
            job_id, "unsupported_type", f"That link is {fetched.content_type}, not a paper."
        )
        return None, None

    async def _open_copy(self, meta: PaperMeta) -> tuple[bytes, str] | None:
        """The registry-named open copy of a paper, if there is one and it can be read."""
        if not meta.pdf_url or safe_public_url(meta.pdf_url) is None:
            return None
        try:
            fetched = await self._fetch(meta.pdf_url, max_bytes=self.limits.max_bytes)
        except netfetch.FetchError as exc:
            log.info("research.open_copy_failed", code=exc.code)
            return None
        return (fetched.data, fetched.url) if b"%PDF-" in fetched.data[:1024] else None

    async def _finish_import(
        self,
        job_id: str,
        kind: str,
        value: str,
        meta: PaperMeta | None,
        direct: Any,
    ) -> None:
        meta = meta or PaperMeta(provider="none")
        await self._progress(job_id, 0.35, "Fetching the open full text, if there is one.")
        copy = direct if isinstance(direct, tuple) else await self._open_copy(meta)
        parsed: pdf.ParsedPdf | None = None
        note = ""
        digest: str | None = None
        if copy is not None:
            data, _origin = copy
            digest = hashlib.sha256(data).hexdigest()
            try:
                parsed = await self._parse(data)
                note = (
                    f"Full text from {meta.pdf_note or 'the address you gave'}."
                    if direct is None or not isinstance(direct, tuple)
                    else "Full text from the address you gave."
                )
            except pdf.PdfError as exc:
                note = f"An open copy was found but could not be read: {exc}"
        elif kind == "url" and not meta.title:
            await self._fail(job_id, "no_metadata", "Nothing identifying a paper was found there.")
            return
        else:
            note = (
                "No open full text was found. Only the registry's details are held; the "
                "analysis can read the abstract, not the paper."
                if meta.abstract
                else "No open full text and no abstract were found. Only the paper's details "
                "are held."
            )

        access = (
            "full_text"
            if parsed
            else meta.access
            if meta.access != "full_text"
            else "abstract_only"
        )
        title = meta.title or (parsed.metadata.get("title") if parsed else None) or value
        doi = meta.doi or (value if kind == "doi" else None)
        arxiv_id = meta.arxiv_id or (value if kind == "arxiv" else None)
        url = canonical_url(meta.url or "") if kind == "url" else None
        duplicate = await self._existing(
            doi=doi, arxiv_id=arxiv_id, url=value if kind == "url" else None, content_hash=digest
        )
        if duplicate is not None:
            if parsed is not None and duplicate.access_status != "full_text":
                await self._store_parsed(duplicate.id, parsed)
                await self._mark_source(
                    duplicate.id,
                    access_status="full_text",
                    access_detail=note,
                    extraction_status="partial" if parsed.partial else "parsed",
                    page_count=parsed.page_count,
                    content_hash=digest,
                )
            await self._set_job(
                job_id,
                source_id=duplicate.id,
                status="completed",
                progress=1.0,
                message="That paper was already in your library.",
                result={"sourceId": duplicate.id, "duplicateOf": duplicate.id},
                finished_at=utcnow(),
            )
            return

        extraction_status = (
            ("partial" if parsed.partial else "parsed") if parsed else "not_applicable"
        )
        try:
            async with self.db.session() as session:
                source = ResearchSource(
                    import_method=kind if kind in {"doi", "arxiv", "url"} else "search",
                    title=title,
                    authors=meta.authors
                    or (parsed.metadata.get("authors") if parsed else [])
                    or [],
                    abstract=meta.abstract or (parsed.metadata.get("abstract") if parsed else None),
                    canonical_url=url or (value if kind == "url" else None),
                    doi=doi,
                    arxiv_id=arxiv_id,
                    published_at=meta.published,
                    venue=meta.venue,
                    topics=meta.topics,
                    access_status=access,
                    access_detail=note,
                    license=meta.license,
                    content_hash=digest,
                    extraction_status=extraction_status,
                    page_count=parsed.page_count if parsed else None,
                    metadata_={"citations": meta.citations, "registry": meta.provider},
                    provenance={
                        "kind": kind,
                        "input": value,
                        "metadata": meta.provider,
                        "fullTextFrom": (copy[1] if copy else None),
                        "retrievedAt": _now(),
                    },
                )
                session.add(source)
                await session.flush()
                source_id = source.id
        except IntegrityError:
            await self._fail(
                job_id,
                "duplicate",
                "That paper was imported at the same moment by another request.",
            )
            return
        if parsed is not None:
            await self._store_parsed(source_id, parsed)
        await self._set_job(
            job_id,
            source_id=source_id,
            status="completed",
            progress=1.0,
            message=note,
            result={
                "sourceId": source_id,
                "accessStatus": access,
                "warnings": parsed.warnings if parsed else [],
            },
            finished_at=utcnow(),
        )

    # --- search -------------------------------------------------------------------------

    async def search(
        self, query: str, *, semantic: bool = False, include_web: bool = False
    ) -> dict[str, Any]:
        text = " ".join(query.split())
        if len(text) < 3:
            raise ResearchError("bad_query", "Type at least three characters to search.", 422)
        client = self._scholarly()
        try:
            found = await client.search(text[:300], limit=SEARCH_LIMIT, semantic=semantic)
        except ScholarlyError as exc:
            raise ResearchError("registry_unavailable", str(exc), 502) from exc
        finally:
            await client.aclose()
        results = []
        for meta in found:
            have = await self._existing(doi=meta.doi, arxiv_id=meta.arxiv_id)
            results.append(
                {
                    "title": meta.title,
                    "authors": meta.authors[:8],
                    "abstract": (meta.abstract or "")[:700] or None,
                    "doi": meta.doi,
                    "arxivId": meta.arxiv_id,
                    "url": meta.url,
                    "publishedAt": meta.published,
                    "venue": meta.venue,
                    "provider": meta.provider,
                    "citations": meta.citations,
                    "access": meta.access,
                    "openCopy": bool(meta.pdf_url),
                    "importedAs": have.id if have else None,
                }
            )
        web, notes = (await self._web_search(text)) if include_web else ([], [])
        return {"query": text, "papers": results, "web": web, "notes": notes}

    async def _web_search(self, query: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Supplementary public reading through Tavily, when configured. Never the paper."""
        if self._web is None:
            return [], ["Web search is not set up."]
        try:
            resolved = await self._web.resolve_search()
            if resolved is None:
                return [], ["No web search service is enabled. Add Tavily under API Integration."]
            if await self._web.reserve(resolved, 1) < 1:
                return [], ["The daily web-search budget is used up."]
            import httpx2

            async with httpx2.AsyncClient(
                timeout=httpx2.Timeout(20.0), follow_redirects=False
            ) as http:
                raw = await resolved.provider.search(
                    http, f"{query} academic finance research paper", count=6, freshness_days=3650
                )
            await self._web.record(resolved, error=None)
        except IntegrationError as exc:
            return [], [str(exc)]
        except Exception as exc:  # noqa: BLE001 - supplementary: never fails the search
            log.info("research.web_search_failed", error=type(exc).__name__)
            return [], ["Web search failed; academic results are shown without it."]
        hits = []
        for item in raw:
            url = safe_public_url(str(item.get("url") or ""))
            if url:
                hits.append(
                    {
                        "title": str(item.get("title") or url)[:200],
                        "url": url,
                        "snippet": str(item.get("snippet") or "")[:400],
                        "kind": "web",
                    }
                )
        return hits, ["Web results are not peer-reviewed papers."]

    # --- analysis -----------------------------------------------------------------------

    async def _text_for(self, source_id: str) -> tuple[ResearchSource, dict[str, Any], bool]:
        """The text an analysis will read, and whether it is the abstract alone."""
        async with self.db.session() as session:
            source = await session.get(ResearchSource, source_id)
            if source is None:
                raise ResearchError("source_not_found", "There is no such paper.", 404)
            document = await session.get(ResearchDocument, source_id)
        if document is not None and source.access_status == "full_text":
            pages = document.pages
            offsets = _offsets(pages)
            full = pdf.PAGE_JOINER.join(p["text"] for p in pages)
            return (
                source,
                {
                    "pages": pages,
                    "full": full,
                    "sections": document.sections,
                    "offsets": offsets,
                    "equations": document.equations,
                },
                False,
            )
        if source.abstract:
            text = source.abstract
            return (
                source,
                {
                    "pages": [{"page": 1, "text": text}],
                    "full": text,
                    "sections": [
                        {
                            "id": "s1",
                            "heading": "Abstract",
                            "level": 1,
                            "start": 0,
                            "end": len(text),
                            "page_start": 1,
                            "page_end": 1,
                        }
                    ],
                    "offsets": [(1, 0, len(text))],
                    "equations": [],
                },
                True,
            )
        raise ResearchError(
            "no_text",
            "Only the paper's details are held: no full text and no abstract. There is nothing "
            "to analyse. Upload the PDF to analyse it.",
            422,
        )

    async def preview(self, source_id: str, *, budget: int | None = None) -> dict[str, Any]:
        """What an analysis would send and cost, before anything is sent."""
        source, doc, abstract_only = await self._text_for(source_id)
        chunks, stats = extraction.chunk_document(
            doc["full"],
            doc["sections"],
            doc["offsets"],
            budget=budget or extraction.DEFAULT_CHAR_BUDGET,
        )
        return {
            "sourceId": source.id,
            "abstractOnly": abstract_only,
            "basis": "abstract only" if abstract_only else "full text",
            "sections": [
                {
                    "id": s["id"],
                    "heading": s["heading"],
                    "included": any(s["id"] in c.section_ids for c in chunks),
                }
                for s in doc["sections"]
            ],
            **extraction.estimate(chunks, stats),
        }

    async def analyze(
        self,
        source_id: str,
        *,
        scope: Tuple4 | None,
        model_ref: str | None,
        budget: int | None = None,
        force: bool = False,
    ) -> Started:
        source, doc, abstract_only = await self._text_for(source_id)
        try:
            model = self.llm.model_for(model_ref)
        except LLMError as exc:
            raise ResearchError("no_model", str(exc), 422) from exc
        if not [m for m in await self.llm.runnable() if m["ref"] == model.ref]:
            raise ResearchError(
                "no_key",
                f"{model.ref} has no enabled API key. Add one under API Integration.",
                422,
            )
        budget = min(budget or extraction.DEFAULT_CHAR_BUDGET, 200_000)
        chunks, stats = extraction.chunk_document(
            doc["full"], doc["sections"], doc["offsets"], budget=budget
        )
        if not chunks:
            raise ResearchError("no_text", "No usable text was found to analyse.", 422)
        scope_label = _scope_label(scope)
        key = extraction.cache_key(
            source_ref=source.content_hash or source.id,
            model_ref=model.ref,
            budget=budget,
            scope="",
            abstract_only=abstract_only,
        )
        if not force:
            reusable = await self._cached(source_id, key)
            if reusable is not None:
                return Started(
                    source_id,
                    reusable,
                    True,
                    "This paper was already analysed with this model and budget; showing "
                    "that result.",
                )
        estimate = extraction.estimate(chunks, stats)
        job_id = await self._new_job(
            kind="analyze",
            source_id=source_id,
            input_ref=source.title[:200],
            model_ref=model.ref,
            provider=model.provider,
            model_id=model.id,
            cache_key=key,
            params={
                "estimate": estimate,
                "budget": budget,
                "abstractOnly": abstract_only,
                "scope": scope.model_dump() if scope else None,
                "scopeLabel": scope_label,
            },
            message="Queued.",
        )

        async def work(job_id: str) -> None:
            await self._analyze_work(job_id, source, doc, abstract_only, chunks, scope, model.ref)

        self._launch(job_id, f"Analysing {source.title[:50]}", work)
        return Started(source_id, job_id, False, "Analysis started.")

    async def _cached(self, source_id: str, key: str) -> str | None:
        async with self.db.session() as session:
            job = (
                (
                    await session.execute(
                        select(ResearchJob)
                        .where(
                            ResearchJob.source_id == source_id,
                            ResearchJob.cache_key == key,
                            ResearchJob.status == "completed",
                        )
                        .order_by(ResearchJob.created_at.desc())
                    )
                )
                .scalars()
                .first()
            )
            if job is None:
                return None
            # Only while its ideas still exist: a deleted analysis is not a cache hit.
            expected = int((job.result or {}).get("ideas", 0))
            have = (
                await session.execute(
                    select(func.count(ResearchIdea.id)).where(ResearchIdea.job_id == job.id)
                )
            ).scalar_one()
        return job.id if have >= expected and expected > 0 else None

    async def _analyze_work(
        self,
        job_id: str,
        source: ResearchSource,
        doc: dict[str, Any],
        abstract_only: bool,
        chunks: list[extraction.Chunk],
        scope: Tuple4 | None,
        model_ref: str,
    ) -> None:
        index = extraction.PageIndex.build(doc["pages"])
        rows: list[dict[str, Any]] = []
        summaries: list[tuple[str, str]] = []
        reasons: list[str] = []
        usage = {"promptTokens": 0, "outputTokens": 0, "thinkingTokens": 0, "totalTokens": 0}
        label = _scope_label(scope)
        async with self._llm_gate:
            for chunk in chunks:
                await self._progress(
                    job_id,
                    0.1 + 0.7 * (chunk.index - 1) / len(chunks),
                    f"Reading part {chunk.index} of {len(chunks)} with the model.",
                )
                prompt = extraction.build_prompt(
                    title=source.title,
                    scope=label,
                    text=chunk.text,
                    abstract_only=abstract_only,
                    part=(chunk.index, len(chunks)),
                )
                try:
                    answer = await self.llm.generate(
                        system=extraction.SYSTEM,
                        user=prompt,
                        model_ref=model_ref,
                        response_schema=extraction.SCHEMA,
                        temperature=TEMPERATURE,
                    )
                except LLMError as exc:
                    await self._fail(job_id, "llm_error", str(exc))
                    return
                for k, v in answer.usage.items():
                    usage[k] = usage.get(k, 0) + int(v)
                await self._set_job(job_id, usage=usage)
                try:
                    parsed = extraction.parse_answer(answer.text)
                except extraction.ModelOutputError as exc:
                    # The whole analysis fails: ideas from the parts that did parse are not kept,
                    # since half an analysis presented as the paper's would mislead.
                    await self._fail(
                        job_id,
                        "invalid_model_output",
                        f"{exc} Part {chunk.index} of {len(chunks)}. Nothing was saved. "
                        "Try again or choose another model.",
                        detail=exc.problems,
                        raw=exc.raw,
                    )
                    return
                summaries.append((parsed.paper_summary, parsed.methodology_summary))
                if parsed.no_ideas_reason:
                    reasons.append(parsed.no_ideas_reason)
                rows.extend(
                    extraction.idea_row(
                        i, index=index, equations=doc["equations"], abstract_only=abstract_only
                    )
                    for i in parsed.ideas
                )
        merged = extraction.merge_ideas(rows)
        await self._progress(job_id, 0.85, "Saving ideas.")
        saved = await self._save_ideas(source.id, job_id, merged)
        assessed = 0
        if scope is not None and saved:
            await self._progress(job_id, 0.92, "Checking the ideas against your BRAIN scope.")
            for idea_id in saved:
                try:
                    await self.assess(idea_id, scope)
                    assessed += 1
                except Exception:
                    log.warning("research.assess_failed", idea_id=idea_id, exc_info=True)
        await self._set_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"{len(saved)} idea(s) extracted."
            if saved
            else "No testable ideas were found.",
            result={
                "ideas": len(saved),
                "ideaIds": saved,
                "assessed": assessed,
                "paperSummary": summaries[0][0] if summaries else "",
                "methodologySummary": " ".join(m for _, m in summaries)[:4000],
                "noIdeasReason": " ".join(reasons)[:600] or None,
                "abstractOnly": abstract_only,
            },
            finished_at=utcnow(),
        )

    async def _save_ideas(
        self, source_id: str, job_id: str, rows: list[dict[str, Any]]
    ) -> list[str]:
        """Store ideas, replacing the earlier unreviewed ones of this paper. Ideas the person
        saved, rejected or annotated are theirs and stay; a new idea they already have is
        skipped."""
        ids: list[str] = []
        async with self.db.session() as session:
            kept = (
                (
                    await session.execute(
                        select(ResearchIdea.title).where(
                            ResearchIdea.source_id == source_id,
                            (ResearchIdea.status != "new") | (ResearchIdea.note != ""),
                        )
                    )
                )
                .scalars()
                .all()
            )
            await session.execute(
                delete(ResearchIdea).where(
                    ResearchIdea.source_id == source_id,
                    ResearchIdea.status == "new",
                    ResearchIdea.note == "",
                )
            )
            have = {extraction.squash(t) for t in kept}
            for row in rows:
                if extraction.squash(row["title"]) in have:
                    continue
                idea = ResearchIdea(source_id=source_id, job_id=job_id, **row)
                session.add(idea)
                await session.flush()
                ids.append(idea.id)
        return ids

    # --- ideas --------------------------------------------------------------------------

    @staticmethod
    def idea_view(idea: ResearchIdea, source: ResearchSource | None = None) -> dict[str, Any]:
        return {
            "id": idea.id,
            "sourceId": idea.source_id,
            "extraSourceIds": idea.extra_source_ids,
            "source": (
                {
                    "id": source.id,
                    "title": source.title,
                    "authors": source.authors,
                    "publishedAt": source.published_at,
                    "accessStatus": source.access_status,
                    "doi": source.doi,
                }
                if source
                else None
            ),
            "jobId": idea.job_id,
            "title": idea.title,
            "topic": idea.topic,
            "mainFinding": idea.main_finding,
            "economicHypothesis": idea.economic_hypothesis,
            "anomaly": idea.anomaly,
            "signalDefinition": idea.signal_definition,
            "signalDirection": idea.signal_direction,
            "requiredVariables": idea.required_variables,
            "reportedFormula": idea.reported_formula,
            "formulaNeedsReview": idea.formula_needs_review,
            "lookback": idea.lookback,
            "holdingPeriod": idea.holding_period,
            "universe": idea.universe,
            "pointInTime": idea.point_in_time,
            "methodology": idea.methodology,
            "limitations": idea.limitations,
            "implementationVariants": idea.implementation_variants,
            "lookaheadConcerns": idea.lookahead_concerns,
            "missingInformation": idea.missing_information,
            "claims": idea.claims,
            "evidence": idea.evidence,
            "support": idea.support,
            "evidenceScope": idea.evidence_scope,
            "brainAssessment": idea.brain_assessment,
            "readiness": idea.readiness,
            "status": idea.status,
            "note": idea.note,
            "createdAt": _iso(idea.created_at),
            "updatedAt": _iso(idea.updated_at),
        }

    async def list_ideas(
        self,
        *,
        source_id: str | None = None,
        status: str | None = None,
        readiness: str | None = None,
        topic: str | None = None,
        support: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(ResearchIdea, ResearchSource)
            .join(ResearchSource, ResearchSource.id == ResearchIdea.source_id)
            .order_by(ResearchIdea.created_at.desc())
            .limit(limit)
        )
        for column, value in (
            (ResearchIdea.source_id, source_id),
            (ResearchIdea.status, status),
            (ResearchIdea.readiness, readiness),
            (ResearchIdea.topic, topic),
            (ResearchIdea.support, support),
        ):
            if value:
                stmt = stmt.where(column == value)
        if search:
            for word in search.lower().split()[:6]:
                stmt = stmt.where(ResearchIdea.search_text.like(f"%{word}%"))
        async with self.db.session() as session:
            rows = (await session.execute(stmt)).all()
        return [self.idea_view(i, s) for i, s in rows]

    async def get_idea(self, idea_id: str) -> dict[str, Any]:
        async with self.db.session() as session:
            idea = await session.get(ResearchIdea, idea_id)
            if idea is None:
                raise ResearchError("idea_not_found", "There is no such idea.", 404)
            source = await session.get(ResearchSource, idea.source_id)
            candidates = (
                (
                    await session.execute(
                        select(AlphaCandidate)
                        .where(AlphaCandidate.idea_id == idea_id)
                        .order_by(AlphaCandidate.version.desc())
                    )
                )
                .scalars()
                .all()
            )
        view = self.idea_view(idea, source)
        view["candidates"] = [self.candidate_view(c) for c in candidates]
        return view

    async def update_idea(
        self, idea_id: str, *, status: str | None, note: str | None
    ) -> dict[str, Any]:
        if status is not None and status not in {"new", "saved", "rejected"}:
            raise ResearchError("bad_status", "Status must be new, saved or rejected.", 422)
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status
        if note is not None:
            values["note"] = note[:MAX_NOTE_CHARS]
        if values:
            async with self.db.session() as session:
                result = await session.execute(
                    update(ResearchIdea).where(ResearchIdea.id == idea_id).values(**values)
                )
                if not (result.rowcount or 0):  # pyright: ignore[reportAttributeAccessIssue]
                    raise ResearchError("idea_not_found", "There is no such idea.", 404)
        return await self.get_idea(idea_id)

    # --- feasibility --------------------------------------------------------------------

    async def assess(self, idea_id: str, scope: Tuple4) -> dict[str, Any]:
        if self.queries is None:
            raise ResearchError("no_catalog", "The field catalogue is not available.", 503)
        async with self.db.session() as session:
            idea = await session.get(ResearchIdea, idea_id)
        if idea is None:
            raise ResearchError("idea_not_found", "There is no such idea.", 404)
        operators = await self._operators() if self._operators else None
        row = {
            "required_variables": idea.required_variables,
            "claims": idea.claims,
            "universe": idea.universe,
            "lookahead_concerns": idea.lookahead_concerns,
            "formula_needs_review": idea.formula_needs_review,
            "evidence_scope": idea.evidence_scope,
            "support": idea.support,
        }
        result = await feasibility.assess(self.queries, operators, scope, row, now=_now())
        async with self.db.session() as session:
            await session.execute(
                update(ResearchIdea)
                .where(ResearchIdea.id == idea_id)
                .values(brain_assessment=result, readiness=result["readiness"])
            )
        return await self.get_idea(idea_id)

    # --- handoff ------------------------------------------------------------------------

    @staticmethod
    def candidate_view(c: AlphaCandidate) -> dict[str, Any]:
        return {
            "id": c.id,
            "ideaId": c.idea_id,
            "expression": c.expression,
            "scope": c.scope,
            "validationOutcome": c.validation_outcome,
            "validationDetail": c.validation_detail,
            "version": c.version,
            "status": c.status,
            "provenance": c.provenance,
            "createdAt": _iso(c.created_at),
        }

    async def hand_off(self, idea_id: str, scope: Tuple4) -> dict[str, Any]:
        """Record a draft for the Alpha Builder. A draft only: no expression, no simulation.

        Refused for an idea that is blocked in this scope, was never assessed there, or is a
        model suggestion nothing in the paper supports: the Builder should not start from one
        without the person knowing, so that stays a choice they make on the idea itself.
        """
        idea = await self.get_idea(idea_id)
        assessment = idea.get("brainAssessment")
        if not assessment or assessment.get("scope") != {
            "instrumentType": scope.instrument_type,
            "region": scope.region,
            "delay": scope.delay,
            "universe": scope.universe,
        }:
            raise ResearchError(
                "not_assessed",
                "Check this idea against that scope first: the assessment on file is for a "
                "different scope or missing.",
                409,
            )
        if assessment["readiness"] == "blocked":
            raise ResearchError(
                "blocked",
                "This idea is blocked in that scope, so it cannot go to the Alpha Builder. "
                "It stays in your library.",
                409,
            )
        async with self.db.session() as session:
            version = (
                await session.execute(
                    select(func.coalesce(func.max(AlphaCandidate.version), 0)).where(
                        AlphaCandidate.idea_id == idea_id
                    )
                )
            ).scalar_one() + 1
            candidate = AlphaCandidate(
                idea_id=idea_id,
                scope=assessment["scope"],
                version=version,
                provenance={
                    "source": idea["sourceId"],
                    "readiness": assessment["readiness"],
                    "support": idea["support"],
                    "assessedAt": assessment["assessedAt"],
                    "createdBy": "research-paper-analyzer",
                    "note": "Draft. No expression written, no simulation run.",
                },
            )
            session.add(candidate)
            await session.flush()
            return self.candidate_view(candidate)


def extraction_text_type() -> Any:
    from sqlalchemy import Text

    return Text


def _offsets(pages: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    cursor = 0
    for p in pages:
        end = cursor + len(p["text"])
        out.append((int(p["page"]), cursor, end))
        cursor = end + len(pdf.PAGE_JOINER)
    return out


def _scope_label(scope: Tuple4 | None) -> str:
    if scope is None:
        return "not chosen"
    return f"{scope.instrument_type} {scope.region} delay {scope.delay} {scope.universe}"
