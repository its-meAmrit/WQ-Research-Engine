"""Prompts, versions, runs, analytics and export: the SPC lab's one entry point.

The API layer holds no logic of its own; it calls this, which holds the rules the platform
documents (name, length, weight precision, frequency) and the rules of this application
(a version never changes, a run keeps its own copy of what it tested).
"""

import asyncio
import difflib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from ..db.models import (
    SpcEvidence,
    SpcPosition,
    SpcPrompt,
    SpcPromptVersion,
    SpcRun,
    SpcRunStatus,
    utcnow,
)
from ..llm.keys import LLMError, NoKeysError
from . import export, portfolio, research
from .errors import SpcConflictError, SpcInvalidError, SpcNotFoundError
from .execution import SpcRunner
from .identifiers import load_reference
from .prompt_generation import APPROACHES
from .schema import (
    DEFAULT_TARGET_EQUITIES,
    DOCUMENTED_MODEL_FAMILIES,
    FREQUENCIES,
    MAX_ACTIVE_PROMPTS,
    MAX_DAILY_SUBMISSIONS,
    MODEL_VERSION_MAX_CHARS,
    PROFILE,
    PROMPT_MAX_CHARS,
)
from .views import (
    Analytics,
    DiffLine,
    EvidenceOut,
    ExportOut,
    ModelChoice,
    Overlap,
    PositionOut,
    PromptCreate,
    PromptDetail,
    PromptOut,
    PromptPatch,
    PromptVersionOut,
    RunBrief,
    RunDetail,
    RunRequest,
    SpcOptions,
    VersionComparison,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from ..db.sqlite import Database
    from ..llm.service import LLMService
    from ..tasks import TaskRegistry


def hundredths(weight: float) -> int:
    """A weight as whole hundredths, refusing a third decimal place rather than rounding it."""
    scaled = weight * 100
    whole = round(scaled)
    if not 0 <= whole <= 100 or abs(scaled - whole) > 1e-6:
        raise SpcInvalidError("Weight must be between 0.00 and 1.00, with at most two decimals.")
    return whole


def _text(value: str) -> str:
    if not value.strip():
        raise SpcInvalidError("The prompt text is empty.")
    if len(value) > PROMPT_MAX_CHARS:
        raise SpcInvalidError(
            f"The prompt is {len(value):,} characters; WorldQuant accepts at most "
            f"{PROMPT_MAX_CHARS:,}. Shorten it by {len(value) - PROMPT_MAX_CHARS:,}."
        )
    return value


def _model_version(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    if len(value) > MODEL_VERSION_MAX_CHARS:
        raise SpcInvalidError(
            f"The model version is {len(value)} characters; the limit is {MODEL_VERSION_MAX_CHARS}."
        )
    return value.strip()


class SpcService:
    def __init__(
        self,
        db: Database,
        llm: LLMService,
        data_dir: Path,
        tasks: TaskRegistry | None = None,
        runner: SpcRunner | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.data_dir = data_dir
        self.runner = runner or SpcRunner(db, llm, data_dir, tasks)

    # -- lifecycle -------------------------------------------------------

    async def reconcile(self) -> int:
        return await self.runner.reconcile()

    async def stop(self) -> None:
        await self.runner.stop()

    # -- options ---------------------------------------------------------

    async def options(self) -> SpcOptions:
        models = await self.llm.runnable()
        reference = await _reference(self.data_dir)
        async with self.db.session() as session:
            active = await _active_count(session)
        return SpcOptions(
            models=[ModelChoice.model_validate(m) for m in models],
            default_model=models[0]["ref"] if models else None,
            frequencies=list(FREQUENCIES),
            approaches={key: label for key, (label, _) in APPROACHES.items()},
            research=research.status(),
            limits={
                "promptMaxChars": PROMPT_MAX_CHARS,
                "modelVersionMaxChars": MODEL_VERSION_MAX_CHARS,
                "maxActivePrompts": MAX_ACTIVE_PROMPTS,
                "maxDailySubmissions": MAX_DAILY_SUBMISSIONS,
                "targetEquities": DEFAULT_TARGET_EQUITIES,
                "activePrompts": active,
                "note": (
                    "The two maximums are WorldQuant submission limits per consultant. Local "
                    "drafts and test runs do not count against them; only the prompts you "
                    "enter on the platform do."
                ),
                "modelFamilies": list(DOCUMENTED_MODEL_FAMILIES),
            },
            schema_profile={
                "keys": list(PROFILE.keys),
                "verified": PROFILE.verified,
                "note": PROFILE.note,
            },
            reference={
                "micList": reference.has_mics,
                "securityMaster": reference.has_securities,
                "help": (
                    "Optional: put spc_mic_reference.csv (ISO 10383) and "
                    "spc_security_master.csv (isin, mic, ticker, name) in the data directory "
                    "to check identifiers against them."
                ),
            },
        )

    # -- prompts ---------------------------------------------------------

    async def list_prompts(self, *, archived: bool = False) -> list[PromptOut]:
        async with self.db.session() as session:
            rows = (
                await session.scalars(
                    select(SpcPrompt)
                    .where(SpcPrompt.archived == archived)
                    .order_by(SpcPrompt.updated_at.desc(), SpcPrompt.id.desc())
                )
            ).all()
            return [await self._prompt_out(session, p) for p in rows]

    async def get_prompt(self, prompt_id: int) -> PromptDetail:
        async with self.db.session() as session:
            return await self._detail(session, await _prompt(session, prompt_id))

    async def create_prompt(self, body: PromptCreate) -> PromptDetail:
        text = _text(body.prompt_text)
        model = await self._known_model(body.preferred_model)
        async with self.db.session() as session:
            prompt = SpcPrompt(
                name=body.name.strip(),
                description=body.description,
                universe=body.universe,
                theme=body.theme,
                update_frequency=body.update_frequency,
                weight_hundredths=hundredths(body.weight),
                preferred_model=model,
                model_version=_model_version(body.model_version),
            )
            session.add(prompt)
            await session.flush()
            session.add(
                SpcPromptVersion(
                    prompt_id=prompt.id,
                    number=1,
                    prompt_text=text,
                    settings=body.settings,
                    note=body.note,
                )
            )
            await session.flush()
            return await self._detail(session, prompt)

    async def update_prompt(self, prompt_id: int, body: PromptPatch) -> PromptDetail:
        async with self.db.session() as session:
            prompt = await _prompt(session, prompt_id)
            latest = await _latest(session, prompt_id)
            if body.name is not None:
                prompt.name = body.name.strip()
            for field in ("description", "universe", "theme"):
                value = getattr(body, field)
                if value is not None:
                    setattr(prompt, field, value)
            if body.update_frequency is not None:
                prompt.update_frequency = body.update_frequency
            if body.weight is not None:
                prompt.weight_hundredths = hundredths(body.weight)
            if body.clear_model:
                prompt.preferred_model = None
            elif body.preferred_model is not None:
                prompt.preferred_model = await self._known_model(body.preferred_model)
            if body.model_version is not None:
                prompt.model_version = _model_version(body.model_version)
            if body.archived is not None:
                prompt.archived = body.archived
            if body.preferred_version_id is not None:
                chosen = await session.get(SpcPromptVersion, body.preferred_version_id)
                if chosen is None or chosen.prompt_id != prompt_id:
                    raise SpcInvalidError("That version does not belong to this prompt.")
                prompt.preferred_version_id = chosen.id
            # A changed text is a new version; the old one is left exactly as it was.
            if body.prompt_text is not None and body.prompt_text != latest.prompt_text:
                session.add(
                    SpcPromptVersion(
                        prompt_id=prompt_id,
                        number=latest.number + 1,
                        prompt_text=_text(body.prompt_text),
                        settings=body.settings,
                        note=body.note,
                    )
                )
            prompt.updated_at = utcnow()
            await session.flush()
            return await self._detail(session, prompt)

    async def duplicate_prompt(self, prompt_id: int) -> PromptDetail:
        """A copy with its own history, starting at weight 0 so it cannot silently double the
        original's share of the portfolio."""
        async with self.db.session() as session:
            source = await _prompt(session, prompt_id)
            latest = await _latest(session, prompt_id)
            copy = SpcPrompt(
                name=f"{source.name[:190]} (copy)",
                description=source.description,
                universe=source.universe,
                theme=source.theme,
                update_frequency=source.update_frequency,
                weight_hundredths=0,
                preferred_model=source.preferred_model,
                model_version=source.model_version,
            )
            session.add(copy)
            await session.flush()
            session.add(
                SpcPromptVersion(
                    prompt_id=copy.id,
                    number=1,
                    prompt_text=latest.prompt_text,
                    settings=latest.settings,
                    note=f"Copied from '{source.name}' version {latest.number}.",
                )
            )
            await session.flush()
            return await self._detail(session, copy)

    async def restore_version(self, prompt_id: int, number: int) -> PromptDetail:
        """Bring an old text back as a new version. History is never rewritten."""
        async with self.db.session() as session:
            prompt = await _prompt(session, prompt_id)
            old = await _version(session, prompt_id, number)
            latest = await _latest(session, prompt_id)
            session.add(
                SpcPromptVersion(
                    prompt_id=prompt_id,
                    number=latest.number + 1,
                    prompt_text=old.prompt_text,
                    settings=old.settings,
                    note=f"Restored from version {old.number}.",
                )
            )
            prompt.updated_at = utcnow()
            await session.flush()
            return await self._detail(session, prompt)

    async def compare(self, prompt_id: int, a: int, b: int) -> VersionComparison:
        async with self.db.session() as session:
            await _prompt(session, prompt_id)
            va, vb = await _version(session, prompt_id, a), await _version(session, prompt_id, b)
            out_a, out_b = (
                await self._version_out(session, va),
                await self._version_out(session, vb),
            )
        lines: list[DiffLine] = []
        for raw in difflib.ndiff(va.prompt_text.splitlines(), vb.prompt_text.splitlines()):
            tag, body = raw[:2], raw[2:]
            if tag == "+ ":
                lines.append(DiffLine(kind="add", text=body))
            elif tag == "- ":
                lines.append(DiffLine(kind="del", text=body))
            elif tag == "  ":
                lines.append(DiffLine(kind="ctx", text=body))
        return VersionComparison(
            a=out_a,
            b=out_b,
            lines=lines,
            added=sum(1 for ln in lines if ln.kind == "add"),
            removed=sum(1 for ln in lines if ln.kind == "del"),
            character_change=len(vb.prompt_text) - len(va.prompt_text),
        )

    # -- runs ------------------------------------------------------------

    async def start_run(
        self, prompt_id: int, body: RunRequest, *, corrects: int | None = None
    ) -> RunDetail:
        async with self.db.session() as session:
            prompt = await _prompt(session, prompt_id)
            earlier: SpcRun | None = None
            if corrects is not None:
                earlier = await session.get(SpcRun, corrects)
                if earlier is None or earlier.prompt_id != prompt_id:
                    raise SpcNotFoundError(f"Run {corrects} is not a run of this prompt.")
                if earlier.status != SpcRunStatus.FAILED.value or not earlier.raw_output:
                    raise SpcConflictError(
                        "Only a run whose answer failed validation can be corrected."
                    )
            if earlier is not None:
                version = await session.get(SpcPromptVersion, earlier.version_id)
            elif body.version_number is not None:
                version = await _version(session, prompt_id, body.version_number)
            elif prompt.preferred_version_id is not None:
                version = await session.get(SpcPromptVersion, prompt.preferred_version_id)
            else:
                version = await _latest(session, prompt_id)
            if version is None:
                raise SpcNotFoundError("That prompt version does not exist.")
            ref = body.model or (earlier.model_ref if earlier else None) or prompt.preferred_model
            info = self.llm.model_for(ref)
            if info.ref not in {m["ref"] for m in await self.llm.runnable()}:
                raise NoKeysError(info.provider)
            busy = await session.scalar(
                select(SpcRun.id).where(
                    SpcRun.version_id == version.id,
                    SpcRun.model_ref == info.ref,
                    SpcRun.status.in_([s.value for s in SpcRunStatus if not s.terminal]),
                )
            )
            if busy is not None:
                raise SpcConflictError(
                    f"Run {busy} of this version on {info.id} is already in progress."
                )
            run = SpcRun(
                prompt_id=prompt_id,
                version_id=version.id,
                version_number=version.number,
                prompt_snapshot=version.prompt_text,
                meta_snapshot={
                    "name": prompt.name,
                    "update_frequency": prompt.update_frequency,
                    "weight_hundredths": prompt.weight_hundredths,
                    "model_version": prompt.model_version,
                    "research": body.research,
                },
                model_ref=info.ref,
                provider=info.provider,
                model_id=info.id,
                status=SpcRunStatus.QUEUED.value,
                message="Waiting for a free slot.",
                corrects_run_id=corrects,
            )
            session.add(run)
            await session.flush()
            run_id = run.id
        # Only after the row is committed: the task reads it back.
        self.runner.start(run_id)
        return await self.get_run(run_id)

    async def correct_run(self, run_id: int, body: RunRequest) -> RunDetail:
        async with self.db.session() as session:
            earlier = await session.get(SpcRun, run_id)
            if earlier is None:
                raise SpcNotFoundError(f"There is no run {run_id}.")
            prompt_id = earlier.prompt_id
        return await self.start_run(prompt_id, body, corrects=run_id)

    async def list_runs(self, *, prompt_id: int | None = None, limit: int = 100) -> list[RunBrief]:
        async with self.db.session() as session:
            query = select(SpcRun).order_by(SpcRun.id.desc()).limit(limit)
            if prompt_id is not None:
                query = query.where(SpcRun.prompt_id == prompt_id)
            runs = (await session.scalars(query)).all()
            return [await self._brief(session, r) for r in runs]

    async def get_run(self, run_id: int) -> RunDetail:
        async with self.db.session() as session:
            run = await _run(session, run_id)
            brief = await self._brief(session, run)
            evidence = (
                await session.scalars(
                    select(SpcEvidence).where(SpcEvidence.run_id == run_id).order_by(SpcEvidence.id)
                )
            ).all()
            rows = await _positions(session, run_id)
            return RunDetail(
                **brief.model_dump(),
                prompt_snapshot=run.prompt_snapshot,
                meta=run.meta_snapshot,
                raw_output=run.raw_output,
                sample_json=run.sample_json,
                validation=run.validation,
                research=run.research,
                evidence=[EvidenceOut.model_validate(e, from_attributes=True) for e in evidence],
                portfolio=[
                    PositionOut(
                        position=r.position,
                        company_name=r.company_name,
                        ticker=r.ticker,
                        isin=r.isin,
                        mic=r.mic,
                        confidence_score=r.confidence_score,
                        investment_horizon=r.investment_horizon,
                        return_prediction=r.return_prediction,
                        isin_status=r.isin_status,
                        mic_status=r.mic_status,
                        mapping_status=r.mapping_status,
                    )
                    for r in rows
                ],
            )

    async def cancel_run(self, run_id: int) -> RunDetail:
        async with self.db.session() as session:
            run = await _run(session, run_id)
            if SpcRunStatus(run.status).terminal:
                raise SpcConflictError(f"Run {run_id} has already {run.status}.")
        if not await self.runner.cancel(run_id):
            # Active on paper, but nothing here is running it: settle the row.
            async with self.db.session() as session:
                run = await _run(session, run_id)
                if not SpcRunStatus(run.status).terminal:
                    run.status = SpcRunStatus.CANCELLED.value
                    run.message = "Cancelled."
                    run.finished_at = utcnow()
        return await self.get_run(run_id)

    # -- analytics -------------------------------------------------------

    async def analytics(self, run_id: int) -> Analytics:
        async with self.db.session() as session:
            run = await _run(session, run_id)
            prompt = await _prompt(session, run.prompt_id)
            positions = [_pos(r) for r in await _positions(session, run_id)]
            previous = await session.scalar(
                select(SpcRun)
                .where(
                    SpcRun.prompt_id == run.prompt_id,
                    SpcRun.id < run.id,
                    SpcRun.status == SpcRunStatus.COMPLETED.value,
                    SpcRun.valid.is_(True),
                )
                .order_by(SpcRun.id.desc())
                .limit(1)
            )
            before = [_pos(r) for r in await _positions(session, previous.id)] if previous else []
        return Analytics(
            run_id=run_id,
            prompt_id=prompt.id,
            prompt_name=prompt.name,
            summary=portfolio.summarise(positions),
            previous_run_id=previous.id if previous else None,
            comparison=portfolio.compare(before, positions) if previous else None,
            performance={"available": False, "reason": portfolio.PERFORMANCE_UNAVAILABLE},
            scoring=list(portfolio.OFFICIAL_SCORING),
        )

    async def overlap(self, run_ids: list[int]) -> Overlap:
        if len(run_ids) < 2:
            raise SpcInvalidError("Choose at least two runs to compare.")
        async with self.db.session() as session:
            grouped: dict[int, list[portfolio.Pos]] = {}
            for run_id in dict.fromkeys(run_ids):
                run = await _run(session, run_id)
                if not run.valid:
                    raise SpcInvalidError(f"Run {run_id} has no validated portfolio to compare.")
                grouped[run_id] = [_pos(r) for r in await _positions(session, run_id)]
        return Overlap(overlap=portfolio.overlap(grouped))

    # -- export ----------------------------------------------------------

    async def export(self, prompt_id: int, run_id: int | None = None) -> ExportOut:
        async with self.db.session() as session:
            prompt = await _prompt(session, prompt_id)
            latest = await _latest(session, prompt_id)
            run: SpcRun | None
            if run_id is not None:
                run = await _run(session, run_id)
                if run.prompt_id != prompt_id:
                    raise SpcNotFoundError(f"Run {run_id} is not a run of this prompt.")
                version = await session.get(SpcPromptVersion, run.version_id)
            else:
                version = (
                    await session.get(SpcPromptVersion, prompt.preferred_version_id)
                    if prompt.preferred_version_id
                    else latest
                )
                query = (
                    select(SpcRun)
                    .where(
                        SpcRun.version_id == (version.id if version else -1),
                        SpcRun.status == SpcRunStatus.COMPLETED.value,
                        SpcRun.valid.is_(True),
                    )
                    .order_by(SpcRun.id.desc())
                )
                if prompt.preferred_model:
                    # The selected model's own test is the one that counts; fall back to any.
                    run = (
                        await session.scalars(
                            query.where(SpcRun.model_ref == prompt.preferred_model)
                        )
                    ).first() or (await session.scalars(query)).first()
                else:
                    run = (await session.scalars(query)).first()
            checks = export.readiness(
                prompt,
                version,
                run,
                latest_number=latest.number,
                active_prompts=await _active_count(session),
            )
            return export.build(
                prompt, version, run, checks, exported_at=datetime.now(UTC).isoformat()
            )

    # -- helpers ---------------------------------------------------------

    async def _known_model(self, ref: str | None) -> str | None:
        if ref is None or not ref.strip():
            return None
        info = self.llm.registry.get(ref.strip())
        if info is None:
            raise LLMError(f"{ref} is not set up. Set it up under LLM Integration › Models.")
        return info.ref

    async def _brief(self, session: AsyncSession, run: SpcRun) -> RunBrief:
        prompt = await session.get(SpcPrompt, run.prompt_id)
        count = await session.scalar(
            select(func.count(SpcPosition.id)).where(SpcPosition.run_id == run.id)
        )
        seconds = (
            (run.finished_at - run.started_at).total_seconds()
            if run.finished_at and run.started_at
            else None
        )
        return RunBrief(
            id=run.id,
            prompt_id=run.prompt_id,
            prompt_name=prompt.name if prompt else "(deleted)",
            version_number=run.version_number,
            status=run.status,
            active=self.runner.active(run.id),
            valid=run.valid,
            model_ref=run.model_ref,
            message=run.message,
            error_code=run.error_code,
            error_message=run.error_message,
            positions=count or 0,
            corrects_run_id=run.corrects_run_id,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_seconds=seconds,
            usage=run.usage,
        )

    async def _version_out(self, session: AsyncSession, v: SpcPromptVersion) -> PromptVersionOut:
        total = await session.scalar(select(func.count(SpcRun.id)).where(SpcRun.version_id == v.id))
        good = await session.scalar(
            select(func.count(SpcRun.id)).where(SpcRun.version_id == v.id, SpcRun.valid.is_(True))
        )
        return PromptVersionOut(
            id=v.id,
            number=v.number,
            prompt_text=v.prompt_text,
            characters=len(v.prompt_text),
            note=v.note,
            settings=v.settings,
            created_at=v.created_at,
            runs=total or 0,
            valid_runs=good or 0,
        )

    async def _prompt_out(self, session: AsyncSession, p: SpcPrompt) -> PromptOut:
        latest = await _latest(session, p.id)
        runs = await session.scalar(select(func.count(SpcRun.id)).where(SpcRun.prompt_id == p.id))
        last = await session.scalar(
            select(SpcRun).where(SpcRun.prompt_id == p.id).order_by(SpcRun.id.desc()).limit(1)
        )
        return PromptOut(
            id=p.id,
            name=p.name,
            description=p.description,
            universe=p.universe,
            theme=p.theme,
            update_frequency=p.update_frequency,
            weight=p.weight_hundredths / 100,
            preferred_model=p.preferred_model,
            model_version=p.model_version,
            archived=p.archived,
            preferred_version_id=p.preferred_version_id,
            latest_version=latest.number,
            latest_version_id=latest.id,
            characters=len(latest.prompt_text),
            runs=runs or 0,
            last_run=await self._brief(session, last) if last else None,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )

    async def _detail(self, session: AsyncSession, p: SpcPrompt) -> PromptDetail:
        base = await self._prompt_out(session, p)
        versions = (
            await session.scalars(
                select(SpcPromptVersion)
                .where(SpcPromptVersion.prompt_id == p.id)
                .order_by(SpcPromptVersion.number.desc())
            )
        ).all()
        return PromptDetail(
            **base.model_dump(),
            prompt_text=versions[0].prompt_text,
            versions=[await self._version_out(session, v) for v in versions],
        )


# --- queries ----------------------------------------------------------------------------


async def _prompt(session: AsyncSession, prompt_id: int) -> SpcPrompt:
    found = await session.get(SpcPrompt, prompt_id)
    if found is None:
        raise SpcNotFoundError(f"There is no SPC prompt {prompt_id}.")
    return found


async def _latest(session: AsyncSession, prompt_id: int) -> SpcPromptVersion:
    found = await session.scalar(
        select(SpcPromptVersion)
        .where(SpcPromptVersion.prompt_id == prompt_id)
        .order_by(SpcPromptVersion.number.desc())
        .limit(1)
    )
    if found is None:
        raise SpcNotFoundError(f"Prompt {prompt_id} has no versions.")
    return found


async def _version(session: AsyncSession, prompt_id: int, number: int) -> SpcPromptVersion:
    found = await session.scalar(
        select(SpcPromptVersion).where(
            SpcPromptVersion.prompt_id == prompt_id, SpcPromptVersion.number == number
        )
    )
    if found is None:
        raise SpcNotFoundError(f"Prompt {prompt_id} has no version {number}.")
    return found


async def _run(session: AsyncSession, run_id: int) -> SpcRun:
    found = await session.get(SpcRun, run_id)
    if found is None:
        raise SpcNotFoundError(f"There is no SPC run {run_id}.")
    return found


async def _positions(session: AsyncSession, run_id: int) -> list[SpcPosition]:
    return list(
        (
            await session.scalars(
                select(SpcPosition)
                .where(SpcPosition.run_id == run_id)
                .order_by(SpcPosition.position)
            )
        ).all()
    )


def _pos(row: SpcPosition) -> portfolio.Pos:
    return portfolio.Pos(
        isin=row.isin,
        mic=row.mic,
        ticker=row.ticker,
        name=row.company_name,
        confidence=row.confidence_score,
        horizon=row.investment_horizon,
        prediction=row.return_prediction,
    )


async def _active_count(session: AsyncSession) -> int:
    return (
        await session.scalar(
            select(func.count(SpcPrompt.id)).where(
                SpcPrompt.archived.is_(False), SpcPrompt.weight_hundredths > 0
            )
        )
        or 0
    )


async def _reference(data_dir: Path) -> Any:
    return await asyncio.to_thread(load_reference, data_dir)
