"""Running a test: research, ask the model, validate, save.

``SpcService.start_run`` writes the run and hands its id here. Everything after happens in a
detached task, so the request that started it returns at once with an id, and the screen
follows the run by reading its row. Every state change is written to that row before the next
step begins, so a restart can say exactly where a run stopped.

A run ends in one of three ways, and only one of them is a result: ``completed`` means the model
answered and the answer passed validation; ``failed`` carries a precise ``error_code`` and keeps
whatever was produced, raw answer included; ``cancelled`` means the person or a shutdown
stopped it.
"""

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, update

from ..db.models import (
    SpcEvidence,
    SpcPosition,
    SpcPromptVersion,
    SpcRun,
    SpcRunStatus,
    utcnow,
)
from ..integrations.base import SETTINGS_PATH
from ..integrations.errors import IntegrationError
from ..integrations.legacy import SETUP_HELP
from ..integrations.service import Resolved
from ..llm.keys import BudgetExhaustedError, LLMError, LLMUnavailable, NoKeysError
from ..tasks import spawn
from . import research
from .errors import SpcError
from .identifiers import load_reference
from .validation import ERROR, WARNING, Issue, validate_output

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import httpx2

    from ..db.sqlite import Database
    from ..integrations.service import WebResearchService
    from ..llm.service import LLMService
    from ..tasks import Task, TaskRegistry

log = structlog.get_logger(__name__)

#: Runs working at once. Each one spends model requests, so the rest wait their turn.
CONCURRENCY = 2
#: Sampling temperature. Low: the answer is a data structure, and a test should be repeatable.
TEMPERATURE = 0.3
#: How much of a failed answer a corrective pass is shown.
CORRECTION_ECHO_CHARS = 6_000

#: Phase -> the sentence the run row and Tasks list show while it is in that phase.
PHASES = {
    SpcRunStatus.QUEUED: "Waiting for a free slot.",
    SpcRunStatus.RESEARCHING: "Searching for current evidence.",
    SpcRunStatus.GENERATING: "Waiting for the model's portfolio.",
    SpcRunStatus.VALIDATING: "Checking the answer against the SPC schema.",
}


def _llm_failure(exc: Exception) -> tuple[str, str]:
    """``(error_code, message)`` for an LLM failure: the precise kind, not a generic one."""
    if isinstance(exc, NoKeysError):
        return "no_api_key", str(exc)
    if isinstance(exc, BudgetExhaustedError):
        return "budget_exhausted", str(exc)
    if isinstance(exc, LLMUnavailable):
        message = str(exc)
        return (
            "provider_timeout" if "did not answer" in message else "provider_unavailable"
        ), message
    if isinstance(exc, LLMError):
        return "llm_error", str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"[:500]


def _correction(run: SpcRun) -> str:
    """The text appended to the prompt when a run corrects an earlier one."""
    problems = [
        f"- entry {i['index']}, {i['field']}: {i['message']}"
        if i.get("index") is not None
        else f"- {i['message']}"
        for i in ((run.validation or {}).get("issues") or [])
        if i.get("severity") == ERROR
    ][:25]
    return (
        "CORRECTION\n"
        "Your previous answer to the task above failed validation:\n"
        + "\n".join(problems)
        + "\n\nYour previous answer was:\n"
        + (run.raw_output or "")[:CORRECTION_ECHO_CHARS]
        + "\n\nReturn the complete corrected JSON array, following the output format exactly, "
        "with no other text. Do not guess identifiers: drop any company you cannot identify."
    )


class SpcRunner:
    """Owns the detached tasks that execute runs."""

    def __init__(
        self,
        db: Database,
        llm: LLMService,
        data_dir: Path,
        tasks: TaskRegistry | None = None,
        *,
        provider: Callable[[], research.SearchProvider | None] | None = None,
        web: WebResearchService | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """``web`` is how a key is found: the application's stored keys, then the legacy
        environment. ``provider`` replaces that outright, for a caller that supplies its own
        search (tests); it is never budgeted. With neither, only the environment is read.
        ``transport`` replaces the network for the search requests, for the same callers."""
        self.db = db
        self.llm = llm
        self.data_dir = data_dir
        self.tasks = tasks
        self._provider = provider
        self._web = web
        self._transport = transport
        self._gate = asyncio.Semaphore(CONCURRENCY)
        self._jobs: dict[int, asyncio.Task[None]] = {}
        #: Why a task was cancelled, so shutdown does not read as a person's choice.
        self._why: dict[int, str] = {}
        self._stopping = False

    # -- control ---------------------------------------------------------

    def start(self, run_id: int) -> None:
        job = spawn(self._execute(run_id), name=f"spc-run-{run_id}")
        self._jobs[run_id] = job  # pyright: ignore[reportArgumentType]
        job.add_done_callback(lambda _t: self._jobs.pop(run_id, None))

    def active(self, run_id: int) -> bool:
        return run_id in self._jobs

    async def cancel(self, run_id: int) -> bool:
        """Stop a run. ``False`` when nothing was running it here, so the caller can settle a
        row an earlier process left active."""
        job = self._jobs.get(run_id)
        if job is None or job.done():
            return False
        self._why[run_id] = "Cancelled."
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        return True

    async def reconcile(self) -> int:
        """Settle runs a previous process left in flight. They cannot still be running: the
        tasks that ran them died with it, and a row saying otherwise would mislead forever."""
        async with self.db.session() as session:
            result = await session.execute(
                update(SpcRun)
                .where(SpcRun.status.in_([s.value for s in PHASES]))
                .values(
                    status=SpcRunStatus.FAILED.value,
                    error_code="interrupted",
                    error_message="Alpha Harness stopped while this run was in progress. "
                    "Nothing was lost; run it again.",
                    message="Interrupted by a restart.",
                    finished_at=utcnow(),
                )
            )
            count = result.rowcount or 0  # pyright: ignore[reportAttributeAccessIssue]
        if count:
            log.warning("spc.runs_interrupted", count=count)
        return count

    async def stop(self) -> None:
        """Called on shutdown: end every run cleanly, recording why."""
        self._stopping = True
        for run_id, job in list(self._jobs.items()):
            self._why[run_id] = "Stopped because Alpha Harness was closing."
            job.cancel()
        await asyncio.gather(*self._jobs.values(), return_exceptions=True)

    # -- one run ---------------------------------------------------------

    async def _set(self, run_id: int, **values: Any) -> None:
        async with self.db.session() as session:
            await session.execute(update(SpcRun).where(SpcRun.id == run_id).values(**values))

    async def _phase(self, run_id: int, status: SpcRunStatus, task: Task | None) -> None:
        await self._set(run_id, status=status.value, message=PHASES[status])
        if task is not None and self.tasks is not None:
            await self.tasks.update(task, detail=PHASES[status])

    async def _execute(self, run_id: int) -> None:
        task: Task | None = None
        if self.tasks is not None:
            task = await self.tasks.start("spc-run", f"SPC test run {run_id}", run_id=run_id)
        state, error = "done", None
        try:
            async with self._gate:
                await self._run(run_id, task)
        except asyncio.CancelledError:
            state = "cancelled"
            await asyncio.shield(
                self._set(
                    run_id,
                    status=SpcRunStatus.CANCELLED.value,
                    message=self._why.pop(run_id, "Cancelled."),
                    finished_at=utcnow(),
                )
            )
            raise
        except Exception as exc:
            # A bug here must not leave the run claiming to be working.
            log.exception("spc.run_crashed", run_id=run_id)
            state, error = "failed", f"{type(exc).__name__}: {exc}"[:300]
            await self._set(
                run_id,
                status=SpcRunStatus.FAILED.value,
                error_code="internal_error",
                error_message=error,
                message="Failed unexpectedly.",
                finished_at=utcnow(),
            )
        finally:
            if task is not None and self.tasks is not None:
                await asyncio.shield(self.tasks.finish(task, state=state, error=error))

    async def _fail(self, run_id: int, code: str, message: str, **extra: Any) -> None:
        await self._set(
            run_id,
            status=SpcRunStatus.FAILED.value,
            error_code=code,
            error_message=message,
            message=message,
            finished_at=utcnow(),
            **extra,
        )

    async def _run(self, run_id: int, task: Task | None) -> None:
        async with self.db.session() as session:
            run = await session.get(SpcRun, run_id)
            if run is None or SpcRunStatus(run.status).terminal:
                return
            version = await session.get(SpcPromptVersion, run.version_id)
            earlier = (
                await session.get(SpcRun, run.corrects_run_id) if run.corrects_run_id else None
            )
        settings: dict[str, Any] = (version.settings if version else None) or {}
        meta = run.meta_snapshot
        await self._set(run_id, started_at=utcnow())

        evidence_text = ""
        summary: dict[str, Any] = {"status": "off"}
        mode = str(meta.get("research", "auto"))
        if mode != "off":
            await self._phase(run_id, SpcRunStatus.RESEARCHING, task)
            outcome = await self._research(run_id, run, settings, mode)
            if outcome is None:
                return  # failed, and the row says why
            summary, evidence_text = outcome
        await self._set(run_id, research=summary)

        await self._phase(run_id, SpcRunStatus.GENERATING, task)
        user = run.prompt_snapshot
        if evidence_text:
            user += "\n\n---\n" + evidence_text
        if earlier is not None:
            user += "\n\n---\n" + _correction(earlier)
        try:
            answer = await self.llm.generate(
                system="",
                user=user,
                model_ref=run.model_ref,
                response_schema=None,
                temperature=TEMPERATURE,
                plain=True,
            )
        # Any provider failure is classified by _llm_failure and shown on the run.
        except Exception as exc:  # noqa: BLE001
            code, message = _llm_failure(exc)
            await self._fail(run_id, code, message)
            return
        usage = {**answer.usage, "requests": 1}
        if not answer.text.strip():
            await self._fail(
                run_id,
                "empty_response",
                f"{answer.model} returned an empty answer. Run it again, or choose another model.",
                raw_output=answer.text,
                usage=usage,
            )
            return

        await self._phase(run_id, SpcRunStatus.VALIDATING, task)
        reference = await asyncio.to_thread(load_reference, self.data_dir)
        report = validate_output(answer.text, reference=reference)
        report.issues.extend(self._expectations(report, settings))
        validation = report.to_dict()
        fields: dict[str, Any] = {
            "raw_output": answer.text,
            "sample_json": report.sample_json,
            "valid": report.valid,
            "validation": validation,
            "usage": usage,
            "finished_at": utcnow(),
        }
        if report.valid:
            async with self.db.session() as session:
                session.add_all(
                    SpcPosition(
                        run_id=run_id,
                        position=p.index,
                        company_name=p.company_name,
                        ticker=p.ticker,
                        isin=p.isin,
                        mic=p.mic,
                        confidence_score=p.confidence_score,
                        investment_horizon=p.investment_horizon,
                        return_prediction=p.return_prediction,
                        isin_status=p.isin_status,
                        mic_status=p.mic_status,
                        mapping_status=p.mapping_status,
                    )
                    for p in report.positions
                )
            await self._set(
                run_id,
                status=SpcRunStatus.COMPLETED.value,
                message=f"{len(report.positions)} positions, valid.",
                error_code=None,
                error_message=None,
                **fields,
            )
            return
        first = next(i for i in report.issues if i.severity == ERROR)
        await self._set(
            run_id,
            status=SpcRunStatus.FAILED.value,
            error_code="invalid_output",
            error_message=f"The answer does not match the SPC schema: {first.message}",
            message=f"{len(report.errors)} validation errors.",
            **fields,
        )

    async def _resolve(self) -> Resolved | None:
        """The search service this run will use: a caller's own, else a stored key, else the
        legacy environment (see :class:`~..integrations.service.WebResearchService`)."""
        if self._provider is not None:
            supplied = self._provider()
            return None if supplied is None else Resolved(supplied, "caller", supplied.name)
        if self._web is not None:
            return await self._web.resolve_search()
        env = research.configured_provider()
        return None if env is None else Resolved(env, "environment", env.name)

    async def _research(
        self, run_id: int, run: SpcRun, settings: dict[str, Any], mode: str
    ) -> tuple[dict[str, Any], str] | None:
        """``(summary, evidence text)``, or ``None`` after failing the run."""
        resolved = await self._resolve()
        if resolved is None:
            if mode == "required":
                await self._fail(
                    run_id,
                    "research_not_configured",
                    "This run needs web research and none is configured. " + SETUP_HELP,
                )
                return None
            return {
                "status": "not_configured",
                "help": SETUP_HELP,
                "note": "Ran without web research: the model answered from its own knowledge.",
            }, ""
        provider = resolved.provider
        now = utcnow()
        queries = research.build_queries(settings, run.prompt_snapshot, now=now)
        wanted = len(queries)
        notes: list[str] = []
        try:
            if self._web is not None and queries:
                # Counted before anything is sent, against the user's own daily budget.
                granted = await self._web.reserve(resolved, wanted)
                if granted < wanted:
                    queries = queries[:granted]
                    notes.append(
                        f"Only {granted} of {wanted} searches were sent: the daily request "
                        f"budget for {provider.name} ran out. Raise it under {SETTINGS_PATH}."
                    )
            found = await research.gather(
                provider,
                queries,
                max_age_days=int(settings.get("freshness_days") or research.DEFAULT_MAX_AGE_DAYS),
                now=now,
                transport=self._transport,
            )
        except (SpcError, IntegrationError) as exc:
            if self._web is not None:
                await self._web.record(
                    resolved, error=exc if isinstance(exc, IntegrationError) else str(exc)
                )
            code = exc.code if isinstance(exc, SpcError) else f"research_{exc.code}"
            if mode == "required":
                await self._fail(run_id, code, str(exc))
                return None
            return {"status": "failed", "provider": provider.name, "error": str(exc)}, ""
        if self._web is not None:
            await self._web.record(resolved, error=None)
        found.notes.extend(notes)
        async with self.db.session() as session:
            session.add_all(
                SpcEvidence(
                    run_id=run_id,
                    url=s.url,
                    title=s.title,
                    source_type=s.source_type,
                    published_at=s.published_at,
                    retrieved_at=s.retrieved_at,
                    snippet=s.snippet,
                    query=s.query,
                    stale=s.stale,
                )
                for s in found.sources
            )
        text = research.evidence_block(found.sources, retrieved=now) if found.sources else ""
        return found.summary(), text

    @staticmethod
    def _expectations(report: Any, settings: dict[str, Any]) -> list[Issue]:
        """Warnings about whether the answer did what the Prompt Studio options asked. Never
        blocking: the options are the author's intent, not the platform's schema."""
        out: list[Issue] = []
        direction = settings.get("direction")
        if report.valid and direction in {"long_only", "short_only"}:
            wrong = [
                p
                for p in report.positions
                if (p.confidence_score < 0) == (direction == "long_only")
            ]
            if wrong:
                out.append(
                    Issue(
                        WARNING,
                        "direction_not_followed",
                        f"The options asked for {direction.replace('_', ' ')} but "
                        f"{len(wrong)} positions go the other way.",
                    )
                )
        target = settings.get("target_equities")
        if report.valid and isinstance(target, int) and target > 0:
            count = len(report.positions)
            if count < target * 0.5 or count > target * 1.5:
                out.append(
                    Issue(
                        WARNING,
                        "target_size_missed",
                        f"The options asked for about {target} securities; the model gave {count}.",
                    )
                )
        return out


async def evidence_for(db: Database, run_id: int) -> list[SpcEvidence]:
    async with db.session() as session:
        return list(
            (
                await session.scalars(
                    select(SpcEvidence).where(SpcEvidence.run_id == run_id).order_by(SpcEvidence.id)
                )
            ).all()
        )
