"""The Alpha Builder's service: designs, candidates, previews, tasks, results and refinement.

Quota discipline, as code:

* Designing, generating, validating, saving, previewing and refining never send anything to
  BRAIN. They read the local catalogue and cached metadata, and call the configured LLM.
* A task is made only from saved candidates, only with ``confirm``, and is stored *idle*
  (``run=False``): the person starts it from Tasks like any other. There is no second
  scheduler; the task is a Paste Alphas task and the existing engine runs it.
* The same candidate cannot make two tasks: the link is written in the transaction that
  checks it, and a repeated request with the same key gets the first task back.
"""

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select, update

from ..brain.settings_schema import chosen_labels, validate_settings
from ..catalog.queries import FieldFilter, Tuple4
from ..db.models import (
    AlphaBuilderDesign,
    AlphaCandidate,
    Study,
    StudyStatus,
    Trial,
    TrialState,
    utcnow,
)
from ..labs import fastexpr
from ..labs.launch import AddedTask, add_study
from ..labs.params import PASTE_ALPHAS, PasteParams
from ..labs.search import MAX_SIMULATIONS
from ..llm.service import LLMError
from ..research.service import ResearchError, ResearchService
from ..tools import paste_alphas, settings_sampler
from . import design as dsn
from . import matcher, validation

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ..brain.schemas import SimulationRequest
    from ..db.sqlite import Database

log = structlog.get_logger()

#: Versions of one idea beyond which the screen warns about repeated searching.
LINEAGE_WARNING = 6
MAX_TASK_CANDIDATES = 20
TEMPERATURE = 0.3

#: Real BRAIN metrics read back from a simulated Alpha, in the platform's own names.
METRICS = (
    "sharpe",
    "fitness",
    "turnover",
    "returns",
    "drawdown",
    "margin",
    "long_count",
    "short_count",
    "train_sharpe",
    "train_fitness",
    "test_sharpe",
    "test_fitness",
    "self_correlation",
    "prod_correlation",
    "after_cost_5bps",
)


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _fingerprint(expression: str, settings: dict[str, Any]) -> str:
    text = json.dumps([fastexpr_norm(expression), settings], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def fastexpr_norm(expression: str) -> str:
    """The expression as the parser writes it back, so spacing never makes a new simulation."""
    try:
        return fastexpr.render(fastexpr.parse(expression))
    except fastexpr.ParseError:
        return expression.strip()


class AlphaBuilderService:
    def __init__(
        self,
        db: Database,
        llm: Any,
        research: ResearchService,
        queries: Any,
        *,
        operators: Callable[[], Awaitable[list[dict[str, Any]] | None]],
        schema: Callable[[], Awaitable[dict[str, Any] | None]],
        alphas: Callable[[list[str]], Awaitable[dict[str, dict[str, Any]]]] | None = None,
        host: Any = None,
    ) -> None:
        self.db, self.llm, self.research, self.queries = db, llm, research, queries
        self._operators, self._schema, self._alphas = operators, schema, alphas
        #: The AppState, for ``add_study``. Set after construction in the application.
        self.host = host
        self._task_lock = asyncio.Lock()

    # --- scope ---------------------------------------------------------------------------

    async def check_scope(self, scope: Tuple4) -> None:
        """The scope exactly as chosen, or an error saying why it cannot be used. Never a
        different scope."""
        held = {
            (r["instrument_type"], r["region"], int(r["delay"]), r["universe"])
            for r in await self.queries.synced_tuples()
        }
        key = (scope.instrument_type, scope.region, scope.delay, scope.universe)
        if key not in held:
            raise ResearchError(
                "scope_not_synced",
                f"The field catalogue holds no data for {scope.label}. Sync that market in the "
                "Data Explorer first. Alpha Builder never switches to another scope.",
                409,
            )
        schema = await self._schema()
        if not schema:
            raise ResearchError(
                "no_settings_metadata",
                "BRAIN's settings list is not loaded. Sign in again to load it.",
                409,
            )
        problems = validate_settings(
            schema,
            {
                "instrumentType": scope.instrument_type,
                "region": scope.region,
                "delay": scope.delay,
                "universe": scope.universe,
            },
        )
        if problems:
            raise ResearchError("scope_unavailable", " ".join(problems), 409)

    @staticmethod
    def _scope_dict(scope: Tuple4) -> dict[str, Any]:
        return {
            "instrumentType": scope.instrument_type,
            "region": scope.region,
            "delay": scope.delay,
            "universe": scope.universe,
        }

    async def options(self) -> dict[str, Any]:
        """What the market picker may offer: BRAIN's own choices and what is downloaded."""
        schema = await self._schema() or {}
        synced = [
            {
                "instrumentType": r["instrument_type"],
                "region": r["region"],
                "delay": int(r["delay"]),
                "universe": r["universe"],
            }
            for r in await self.queries.synced_tuples()
        ]
        return {"settingsSchema": schema, "syncedScopes": synced, "maxVariants": dsn.MAX_VARIANTS}

    # --- reading -------------------------------------------------------------------------

    async def _idea(self, idea_id: str) -> dict[str, Any]:
        return await self.research.get_idea(idea_id)

    async def _model(self, model_ref: str | None) -> Any:
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
        return model

    async def design_row(self, design_id: str) -> AlphaBuilderDesign:
        async with self.db.session() as session:
            row = await session.get(AlphaBuilderDesign, design_id)
        if row is None:
            raise ResearchError("design_not_found", "There is no such signal design.", 404)
        return row

    async def _candidate_row(self, candidate_id: str) -> AlphaCandidate:
        async with self.db.session() as session:
            row = await session.get(AlphaCandidate, candidate_id)
        if row is None:
            raise ResearchError("candidate_not_found", "There is no such candidate.", 404)
        return row

    async def design_view(self, row: AlphaBuilderDesign) -> dict[str, Any]:
        resolved = await matcher.resolve(
            self.queries,
            validation.scope_of(row.scope),
            row.mapping,
            row.substitutions,
            row.accepted_proxies,
        )
        return {
            "id": row.id,
            "ideaId": row.idea_id,
            "scope": row.scope,
            "design": row.design,
            "mapping": resolved,
            "substitutions": row.substitutions,
            "acceptedProxies": row.accepted_proxies,
            "modelRef": row.model_ref,
            "jobId": row.job_id,
            "createdAt": _iso(row.created_at),
            "blocked": [
                {"variable": e["name"], "reason": e["blockedReason"], "essential": e["essential"]}
                for e in resolved
                if not e["usable"]
            ],
        }

    async def list_designs(self, idea_id: str) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            rows = (
                await session.scalars(
                    select(AlphaBuilderDesign)
                    .where(AlphaBuilderDesign.idea_id == idea_id)
                    .order_by(AlphaBuilderDesign.created_at.desc())
                )
            ).all()
        return [await self.design_view(r) for r in rows]

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
            "designId": c.design_id,
            "parentId": c.parent_id,
            "variantKind": c.variant_kind,
            "settings": c.settings,
            "explanation": c.explanation,
            "changes": c.changes,
            "limitations": c.limitations,
            "fieldsUsed": c.fields_used,
            "operatorsUsed": c.operators_used,
            "savedAt": _iso(c.saved_at),
            "studyId": c.study_id,
            "note": c.note,
        }

    async def list_candidates(self, idea_id: str) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            rows = (
                await session.scalars(
                    select(AlphaCandidate)
                    .where(AlphaCandidate.idea_id == idea_id)
                    .order_by(AlphaCandidate.version)
                )
            ).all()
        return [self.candidate_view(r) for r in rows]

    async def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        return self.candidate_view(await self._candidate_row(candidate_id))

    # --- step 1: the signal design ---------------------------------------------------------

    async def start_design(
        self, idea_id: str, scope: Tuple4, model_ref: str | None
    ) -> dict[str, Any]:
        idea = await self._idea(idea_id)
        await self.check_scope(scope)
        model = await self._model(model_ref)
        operators = await self._operators() or []
        if not operators:
            raise ResearchError(
                "operators_unread",
                "Your BRAIN operators are not loaded. Sign in again, then reload.",
                409,
            )
        job_id = await self.research.new_job(
            kind="design",
            source_id=idea["sourceId"],
            input_ref=idea["title"][:200],
            model_ref=model.ref,
            provider=model.provider,
            model_id=model.id,
            params={"ideaId": idea_id, "scope": self._scope_dict(scope)},
            message="Queued.",
        )

        async def work(job_id: str) -> None:
            await self._design_work(job_id, idea, scope, model.ref, operators)

        self.research.launch(job_id, f"Signal design · {idea['title'][:40]}", work)
        return {"jobId": job_id, "message": "Signal design started."}

    async def _generate(
        self, job_id: str, *, user: str, model_ref: str, schema: dict[str, Any], kind: type[Any]
    ) -> Any:
        """One model call: the answer parsed and validated, or the job failed and ``None``."""
        async with self.research.llm_gate:
            try:
                answer = await self.llm.generate(
                    system=dsn.SYSTEM,
                    user=user,
                    model_ref=model_ref,
                    response_schema=schema,
                    temperature=TEMPERATURE,
                )
            except LLMError as exc:
                await self.research.fail(job_id, "llm_error", str(exc))
                return None
        await self.research.set_job(job_id, usage={k: int(v) for k, v in answer.usage.items()})
        try:
            return dsn.parse(kind, answer.text)
        except dsn.ModelOutputError as exc:
            await self.research.set_job(job_id, raw_output=exc.raw)
            await self.research.fail(
                job_id,
                "invalid_model_output",
                f"{exc} Nothing was saved. Try again or choose another model.",
                detail=exc.problems,
            )
            return None

    async def _design_work(
        self,
        job_id: str,
        idea: dict[str, Any],
        scope: Tuple4,
        model_ref: str,
        operators: list[dict[str, Any]],
    ) -> None:
        table = fastexpr.operator_table(operators, "REGULAR")
        await self.research.progress(job_id, 0.15, "Asking the model for a signal design.")
        parsed = await self._generate(
            job_id,
            user=dsn.design_prompt(
                idea=idea, scope_label=scope.label, operator_names=sorted(table)
            ),
            model_ref=model_ref,
            schema=dsn.DESIGN_SCHEMA,
            kind=dsn.SignalDesign,
        )
        if parsed is None:
            return
        await self.research.progress(job_id, 0.6, "Matching its data to the catalogue.")
        mapping = await matcher.match(
            self.queries, scope, [v.model_dump() for v in parsed.variables]
        )
        design = parsed.model_dump()
        # An operator the model names that the account lacks is said so, not dropped.
        design["unknown_operators"] = [o for o in parsed.relevant_operators if o not in table]
        async with self.db.session() as session:
            row = AlphaBuilderDesign(
                idea_id=idea["id"],
                scope=self._scope_dict(scope),
                design=design,
                mapping=mapping,
                model_ref=model_ref,
                job_id=job_id,
            )
            session.add(row)
            await session.flush()
            design_id = row.id
        await self.research.set_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Signal design ready for review.",
            result={"designId": design_id},
            finished_at=utcnow(),
        )

    async def update_mapping(
        self,
        design_id: str,
        substitutions: dict[str, str] | None,
        accepted_proxies: list[str] | None,
    ) -> dict[str, Any]:
        row = await self.design_row(design_id)
        scope = validation.scope_of(row.scope)
        names = {m["name"] for m in row.mapping}
        subs = dict(row.substitutions if substitutions is None else substitutions)
        for variable, field_id in list(subs.items()):
            if variable not in names:
                raise ResearchError(
                    "unknown_variable", f"The design has no variable {variable!r}.", 422
                )
            if not field_id:
                del subs[variable]
                continue
            if await matcher.verify_field(self.queries, scope, field_id) is None:
                raise ResearchError(
                    "field_not_in_scope",
                    f"{field_id} is not a field in {scope.label}. Nothing was changed.",
                    422,
                )
        accepted = list(row.accepted_proxies if accepted_proxies is None else accepted_proxies)
        if unknown := [a for a in accepted if a not in names]:
            raise ResearchError(
                "unknown_variable", f"The design has no variable {unknown[0]!r}.", 422
            )
        async with self.db.session() as session:
            await session.execute(
                update(AlphaBuilderDesign)
                .where(AlphaBuilderDesign.id == design_id)
                .values(substitutions=subs, accepted_proxies=accepted)
            )
        return await self.design_view(await self.design_row(design_id))

    async def search_fields(self, scope: Tuple4, text: str) -> list[dict[str, Any]]:
        """Fields in exactly this scope, for the person choosing a substitution."""
        await self.check_scope(scope)
        page = await self.queries.fields(
            scope, FieldFilter(search=text, limit=25, sort_by="alpha_count"), counted=False
        )
        return [matcher.view(r) for r in page["results"]]

    # --- step 2: candidate variants --------------------------------------------------------

    async def start_variants(
        self,
        design_id: str,
        *,
        settings: dict[str, Any],
        settings_confirmed: bool,
        count: int,
        kinds: list[str] | None,
        model_ref: str | None,
    ) -> dict[str, Any]:
        row = await self.design_row(design_id)
        scope = validation.scope_of(row.scope)
        if not settings_confirmed:
            raise ResearchError(
                "settings_unconfirmed",
                "Confirm the simulation settings first. They are never chosen for you.",
                422,
            )
        await self.check_scope(scope)
        schema = await self._schema()
        problems = [
            c
            for c in validation.check_settings(settings, row.scope, schema)
            if c["severity"] == "block"
        ]
        if problems:
            raise ResearchError("settings_blocked", problems[0]["message"], 422)
        view = await self.design_view(row)
        missing = [b for b in view["blocked"] if b["essential"]]
        if missing:
            raise ResearchError(
                "field_mapping_blocked",
                f"{missing[0]['variable']}: {missing[0]['reason']}",
                409,
            )
        chosen = [k for k in (kinds or dsn.VARIANT_KINDS) if k in dsn.VARIANT_KINDS]
        if not chosen:
            raise ResearchError("no_kinds", "Choose at least one kind of variant.", 422)
        count = max(1, min(count, dsn.MAX_VARIANTS))
        model = await self._model(model_ref)
        operators = await self._operators() or []
        if not operators:
            raise ResearchError("operators_unread", "Your BRAIN operators are not loaded.", 409)
        idea = await self._idea(row.idea_id)
        job_id = await self.research.new_job(
            kind="variants",
            source_id=idea["sourceId"],
            input_ref=idea["title"][:200],
            model_ref=model.ref,
            provider=model.provider,
            model_id=model.id,
            params={"designId": design_id, "count": count, "kinds": chosen, "scope": row.scope},
            message="Queued.",
        )

        async def work(job_id: str) -> None:
            await self._variants_work(
                job_id, idea, row, view, settings, count, chosen, model.ref, operators
            )

        self.research.launch(job_id, f"Alpha variants · {idea['title'][:40]}", work)
        return {"jobId": job_id, "message": "Candidate generation started."}

    async def _variants_work(
        self,
        job_id: str,
        idea: dict[str, Any],
        design: AlphaBuilderDesign,
        view: dict[str, Any],
        settings: dict[str, Any],
        count: int,
        kinds: list[str],
        model_ref: str,
        operators: list[dict[str, Any]],
    ) -> None:
        table = fastexpr.operator_table(operators, "REGULAR")
        wanted = {
            o for t in design.design.get("transformations", []) for o in t.get("operators", [])
        }
        wanted |= set(design.design.get("relevant_operators", []))
        shown = [
            {
                "name": n,
                "definition": next((o["definition"] for o in operators if o.get("name") == n), ""),
            }
            for n in sorted(table)
            if n in wanted
            or n.startswith(("rank", "zscore", "group_", "vec_", "ts_backfill", "winsorize"))
        ][:80]
        await self.research.progress(job_id, 0.2, "Asking the model for candidate expressions.")
        parsed = await self._generate(
            job_id,
            user=dsn.variants_prompt(
                idea=idea,
                design=design.design,
                fields=matcher.usable_fields(view["mapping"]),
                operators=shown,
                scope_label=validation.scope_of(design.scope).label,
                kinds=kinds,
                count=count,
            ),
            model_ref=model_ref,
            schema=dsn.VARIANTS_SCHEMA,
            kind=dsn.Variants,
        )
        if parsed is None:
            return
        await self.research.progress(
            job_id, 0.7, "Checking every candidate against BRAIN metadata."
        )
        ids: list[str] = []
        base: str | None = None
        for index, variant in enumerate(parsed.variants[:count]):
            created = await self._store_candidate(
                idea=idea,
                scope=design.scope,
                expression=variant.expression,
                settings=settings,
                design_id=design.id,
                parent_id=base,
                kind=variant.kind,
                explanation={
                    "name": variant.name,
                    "relationToSource": variant.relation_to_source,
                    "implementation": variant.implementation_explanation,
                    "sourceBackedParts": variant.source_backed_parts,
                    "adaptations": [a.model_dump() for a in variant.adaptations],
                    "directionJustification": variant.direction_justification,
                },
                changes=variant.changes_from_base,
                limitations=variant.limitations,
                origin="alpha-builder",
                model_ref=model_ref,
                mapping=view["mapping"],
            )
            if index == 0:
                base = created
            ids.append(created)
        await self.research.set_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"{len(ids)} candidate(s) written as drafts.",
            result={"candidateIds": ids, "fewerReason": parsed.fewer_reason},
            finished_at=utcnow(),
        )

    # --- storing and validating ------------------------------------------------------------

    async def _validate_row(
        self,
        expression: str | None,
        settings: dict[str, Any] | None,
        scope: dict[str, Any],
        idea: dict[str, Any],
    ) -> dict[str, Any]:
        return await validation.validate(
            expression=expression,
            settings=settings,
            scope=scope,
            operators=await self._operators(),
            schema=await self._schema(),
            queries=self.queries,
            lookahead_concerns=idea.get("lookaheadConcerns") or [],
        )

    async def _store_candidate(
        self,
        *,
        idea: dict[str, Any],
        scope: dict[str, Any],
        expression: str,
        settings: dict[str, Any],
        design_id: str | None,
        parent_id: str | None,
        kind: str,
        explanation: dict[str, Any],
        changes: list[str],
        limitations: list[str],
        origin: str,
        model_ref: str | None = None,
        mapping: list[dict[str, Any]] | None = None,
        extra_provenance: dict[str, Any] | None = None,
    ) -> str:
        detail = await self._validate_row(expression, settings, scope, idea)
        proxies = [
            {
                "variable": e["name"],
                "fieldId": e["final"]["fieldId"],
                "status": e["status"],
                "approximation": e["approximation"],
            }
            for e in (mapping or [])
            if e.get("final")
            and e["status"] in {matcher.PROXY, matcher.MANUAL}
            and e["final"]["fieldId"] in detail["fieldsUsed"]
        ]
        warnings = list(limitations)
        if proxies:
            warnings.append(
                "Uses proxy or substituted data: "
                + ", ".join(f"{p['fieldId']} for {p['variable']}" for p in proxies)
                + "."
            )
        async with self.db.session() as session:
            version = (
                await session.execute(
                    select(func.coalesce(func.max(AlphaCandidate.version), 0)).where(
                        AlphaCandidate.idea_id == idea["id"]
                    )
                )
            ).scalar_one() + 1
            row = AlphaCandidate(
                idea_id=idea["id"],
                scope=scope,
                expression=expression.strip(),
                settings=settings,
                version=version,
                status="draft",
                design_id=design_id,
                parent_id=parent_id,
                variant_kind=kind,
                explanation=explanation,
                changes=changes,
                limitations=warnings,
                validation_outcome=detail["outcome"],
                validation_detail=detail,
                fields_used=detail["fieldsUsed"],
                operators_used=detail["operatorsUsed"],
                fingerprint=_fingerprint(expression, settings),
                provenance={
                    "createdBy": origin,
                    "modelRef": model_ref,
                    "source": idea["sourceId"],
                    "sourceTitle": (idea.get("source") or {}).get("title"),
                    "hypothesis": idea.get("economicHypothesis"),
                    "evidence": [
                        {"quote": (e.get("quote") or "")[:300], "page": e.get("page")}
                        for e in (idea.get("evidence") or [])[:6]
                    ],
                    "support": idea.get("support"),
                    "proxies": proxies,
                    "note": "Draft. Static checks only: no simulation has been run.",
                    **(extra_provenance or {}),
                },
            )
            session.add(row)
            await session.flush()
            return row.id

    async def revalidate(self, candidate_id: str) -> dict[str, Any]:
        """Run the static checks again against current metadata. The expression and settings
        are not touched; only the reading is."""
        row = await self._candidate_row(candidate_id)
        idea = await self._idea(row.idea_id)
        detail = await self._validate_row(row.expression, row.settings, row.scope, idea)
        async with self.db.session() as session:
            await session.execute(
                update(AlphaCandidate)
                .where(AlphaCandidate.id == candidate_id)
                .values(validation_outcome=detail["outcome"], validation_detail=detail)
            )
        return await self.get_candidate(candidate_id)

    async def check_edit(
        self, candidate_id: str, expression: str | None, settings: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Validate a proposed edit without keeping anything."""
        row = await self._candidate_row(candidate_id)
        idea = await self._idea(row.idea_id)
        return await self._validate_row(
            row.expression if expression is None else expression,
            row.settings if settings is None else settings,
            row.scope,
            idea,
        )

    async def revise(
        self,
        candidate_id: str,
        *,
        expression: str | None,
        settings: dict[str, Any] | None,
        note: str = "",
    ) -> dict[str, Any]:
        """An edit: always a new version with this one as parent. The original is untouched."""
        row = await self._candidate_row(candidate_id)
        new_expression = row.expression if expression is None else expression
        new_settings = row.settings if settings is None else settings
        if not (new_expression or "").strip():
            raise ResearchError("no_expression", "There is no expression.", 422)
        if _fingerprint(new_expression or "", new_settings or {}) == row.fingerprint:
            raise ResearchError("unchanged", "Nothing changed: this is the same simulation.", 409)
        idea = await self._idea(row.idea_id)
        changes = []
        if expression is not None and fastexpr_norm(expression) != fastexpr_norm(
            row.expression or ""
        ):
            changes.append("Expression edited by hand.")
        if settings is not None and settings != row.settings:
            diff = sorted(
                k
                for k in {*settings, *(row.settings or {})}
                if settings.get(k) != (row.settings or {}).get(k)
            )
            changes.append(f"Settings edited: {', '.join(diff)}.")
        new_id = await self._store_candidate(
            idea=idea,
            scope=row.scope,
            expression=new_expression or "",
            settings=new_settings or {},
            design_id=row.design_id,
            parent_id=row.id,
            kind="manual_edit",
            explanation=dict(row.explanation or {}),
            changes=changes,
            limitations=[x for x in row.limitations if not x.startswith("Uses proxy")],
            origin="manual-edit",
        )
        if note:
            async with self.db.session() as session:
                await session.execute(
                    update(AlphaCandidate)
                    .where(AlphaCandidate.id == new_id)
                    .values(note=note[:2000])
                )
        return await self.get_candidate(new_id)

    async def save(self, candidate_id: str) -> dict[str, Any]:
        """Keep a candidate for task creation. Blocked candidates are refused."""
        row = await self._candidate_row(candidate_id)
        detail = await self._validate_row(
            row.expression, row.settings, row.scope, await self._idea(row.idea_id)
        )
        async with self.db.session() as session:
            await session.execute(
                update(AlphaCandidate)
                .where(AlphaCandidate.id == candidate_id)
                .values(validation_outcome=detail["outcome"], validation_detail=detail)
            )
        if detail["outcome"] == validation.BLOCKED:
            raise ResearchError(
                "candidate_blocked",
                "This candidate is blocked: "
                + next(c["message"] for c in detail["checks"] if c["severity"] == "block"),
                409,
            )
        if row.status == "draft":
            async with self.db.session() as session:
                await session.execute(
                    update(AlphaCandidate)
                    .where(AlphaCandidate.id == candidate_id, AlphaCandidate.status == "draft")
                    .values(status="saved", saved_at=utcnow())
                )
        return await self.get_candidate(candidate_id)

    # --- preview and task ------------------------------------------------------------------

    async def _plan(self, candidate_ids: list[str]) -> dict[str, Any]:
        """The task as it would be made, from current metadata. Reads only."""
        if not candidate_ids:
            raise ResearchError("no_candidates", "Choose at least one candidate.", 422)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ResearchError("duplicate_candidates", "A candidate is listed twice.", 422)
        if len(candidate_ids) > MAX_TASK_CANDIDATES:
            raise ResearchError(
                "too_many", f"At most {MAX_TASK_CANDIDATES} candidates per task.", 422
            )
        schema = await self._schema()
        items: list[dict[str, Any]] = []
        requests: list[SimulationRequest] = []
        owners: list[str] = []
        problems: list[str] = []
        seen: dict[str, str] = {}
        for candidate_id in candidate_ids:
            row = await self._candidate_row(candidate_id)
            idea = await self._idea(row.idea_id)
            detail = await self._validate_row(row.expression, row.settings, row.scope, idea)
            blockers: list[str] = [
                c["message"] for c in detail["checks"] if c["severity"] == "block"
            ]
            if row.status == "draft":
                blockers.append("Save the candidate first.")
            if row.study_id is not None:
                blockers.append("A task was already made from this candidate.")
            duplicate = seen.get(row.fingerprint or "")
            if duplicate:
                blockers.append(
                    f"Same expression and settings as candidate {duplicate} in this task."
                )
            request = None
            if not blockers:
                request = paste_alphas.request_of(row.expression or "", row.settings or {})
                dumped = request.settings.model_dump(by_alias=True, exclude_none=True)
                # The regression guard: what would be sent must equal the chosen scope.
                for key in ("instrumentType", "region", "delay", "universe"):
                    if dumped.get(key) != row.scope.get(key):
                        blockers.append(
                            f"The request would run in {key}={dumped.get(key)!r}, not the chosen "
                            f"{row.scope.get(key)!r}. Refused."
                        )
                        request = None
                        break
            if request is not None:
                requests.append(request)
                owners.append(row.id)
                seen[row.fingerprint or ""] = row.id
            problems.extend(f"{row.id}: {b}" for b in blockers)
            items.append(
                {
                    "candidateId": row.id,
                    "version": row.version,
                    "expression": row.expression,
                    "scope": row.scope,
                    "settings": request.settings.model_dump(by_alias=True, exclude_none=True)
                    if request
                    else row.settings,
                    "labels": chosen_labels(schema, row.settings or {}) if schema else {},
                    "outcome": detail["outcome"],
                    "needsReview": detail["outcome"] == validation.REVIEW,
                    "reviewReasons": [
                        c["message"] for c in detail["checks"] if c["severity"] == "review"
                    ],
                    "blockers": blockers,
                    "fields": detail["fieldsUsed"],
                    "runnable": request is not None,
                }
            )
        # BRAIN's batching: the same grouping the Paste Alphas preview shows.
        ordered = paste_alphas.ordered(requests)
        return {
            "items": items,
            "requests": requests,
            "owners": owners,
            "simulations": len(requests),
            "batches": len(paste_alphas.chunks(ordered)),
            "problems": problems,
            "needsReview": any(i["needsReview"] and i["runnable"] for i in items),
        }

    async def preview(self, candidate_ids: list[str]) -> dict[str, Any]:
        """Free and read-only: nothing is written and nothing is sent anywhere."""
        slots = getattr(getattr(self.host, "engine", None), "slots", None)
        plan = await self._plan(candidate_ids)
        return {
            "items": plan["items"],
            "simulations": plan["simulations"],
            "batches": plan["batches"],
            "problems": plan["problems"],
            "needsReview": plan["needsReview"],
            "maxCores": slots,
            "quota": (
                f"{plan['simulations']} simulation(s) in {plan['batches']} batch(es). Each "
                "counts against your BRAIN simulation allowance when it runs. Alpha Harness does "
                "not read your remaining allowance; the task starts only when you start it in "
                "Tasks."
            ),
            "starts": False,
        }

    async def create_task(
        self,
        candidate_ids: list[str],
        *,
        key: str,
        confirm: bool,
        acknowledge_review: bool,
        cores: int | None,
    ) -> dict[str, Any]:
        if not confirm:
            raise ResearchError(
                "not_confirmed", "Confirm the task to create it. Nothing was created.", 422
            )
        if self.host is None:
            raise ResearchError("no_engine", "Tasks are not available.", 503)
        async with self._task_lock:
            existing = await self._task_for_key(key)
            if existing is not None:
                return existing | {"existing": True}
            plan = await self._plan(candidate_ids)
            if plan["problems"] or not plan["requests"]:
                raise ResearchError(
                    "preview_blocked",
                    plan["problems"][0] if plan["problems"] else "Nothing here can run.",
                    409,
                )
            if plan["needsReview"] and not acknowledge_review:
                raise ResearchError(
                    "review_not_acknowledged",
                    "A candidate needs review. Read its warnings and acknowledge them to continue.",
                    409,
                )
            if len(plan["requests"]) > MAX_SIMULATIONS:
                raise ResearchError("too_many", f"At most {MAX_SIMULATIONS:,} per task.", 422)
            requests: list[SimulationRequest] = plan["requests"]
            owners: list[str] = plan["owners"]
            by_fingerprint = dict(zip(owners, requests, strict=True))
            ordered = paste_alphas.ordered(requests)
            owner_of = {id(r): c for c, r in by_fingerprint.items()}
            engine_slots = int(getattr(getattr(self.host, "engine", None), "slots", 1))
            cores = max(1, min(cores or engine_slots, engine_slots))
            seeds = settings_sampler.seed_trials(ordered)

            def build(study_id: int) -> list[Trial]:
                trials = seeds(study_id)
                for trial, request in zip(trials, ordered, strict=True):
                    trial.params = {"candidateId": owner_of[id(request)], "origin": "alpha-builder"}
                return trials

            first = ordered[0].settings
            added: AddedTask = await add_study(
                self.host,
                now=utcnow(),
                sampler=PASTE_ALPHAS,
                params=PasteParams(
                    region=first.region,
                    delay=first.delay,
                    markets=len(
                        {
                            (r.settings.region, r.settings.delay, r.settings.universe)
                            for r in ordered
                        }
                    ),
                    cores=cores,
                ),
                simulations=len(ordered),
                batch_size=(cores + 1) * 10,
                template_source=await self._copy_text(owners),
                template_name=f"Alpha Builder · {len(ordered)} candidate(s)",
                seeds=build,
                run=False,
            )
            async with self.db.session() as session:
                await session.execute(
                    update(AlphaCandidate)
                    .where(AlphaCandidate.id.in_(owners))
                    .values(study_id=added.id, task_key=key, status="tasked")
                )
            return {
                "taskId": added.id,
                "name": added.name,
                "candidateIds": owners,
                "existing": False,
                "started": False,
            }

    async def _task_for_key(self, key: str) -> dict[str, Any] | None:
        async with self.db.session() as session:
            rows = (
                await session.scalars(select(AlphaCandidate).where(AlphaCandidate.task_key == key))
            ).all()
            if not rows:
                return None
            study = await session.get(Study, rows[0].study_id) if rows[0].study_id else None
        return {
            "taskId": rows[0].study_id,
            "name": study.name if study else "",
            "candidateIds": [r.id for r in rows],
            "started": False,
        }

    async def _copy_text(self, candidate_ids: list[str]) -> str:
        """The task's source in Copy Alphas form, so it reads like any Paste Alphas task."""
        parts: list[str] = []
        for candidate_id in candidate_ids:
            row = await self._candidate_row(candidate_id)
            settings = row.settings or {}
            lines = [f"Alpha Expression:\n{row.expression}", "Simulation Settings:"]
            lines += [
                f"{title}: {settings[name]}"
                for name, title in paste_alphas.SETTINGS
                if name in settings
            ]
            parts.append(
                "\n".join(lines) + f"\n(Alpha Builder candidate {row.id}, version {row.version})"
            )
        return "\n\n".join(parts)

    # --- results ---------------------------------------------------------------------------

    async def results(self, candidate_id: str) -> dict[str, Any]:
        """What BRAIN returned for this candidate's simulation, exactly as stored, or the
        state it is waiting in. Nothing is estimated."""
        row = await self._candidate_row(candidate_id)
        base = {
            "candidateId": row.id,
            "version": row.version,
            "ideaId": row.idea_id,
            "settings": row.settings,
            "expression": row.expression,
        }
        if row.study_id is None:
            return base | {
                "state": "no_task",
                "message": "No task has been made from this candidate.",
                "trials": [],
                "metrics": None,
                "checks": [],
                "diagnostics": [],
            }
        async with self.db.session() as session:
            study = await session.get(Study, row.study_id)
            trials = (
                await session.scalars(
                    select(Trial)
                    .where(
                        Trial.study_id == row.study_id,
                        func.json_extract(Trial.params, "$.candidateId") == row.id,
                    )
                    .order_by(Trial.number)
                )
            ).all()
        alpha_ids = [t.alpha_id for t in trials if t.alpha_id]
        stored = await self._alphas(alpha_ids) if (self._alphas and alpha_ids) else {}
        shown: list[dict[str, Any]] = []
        metrics: dict[str, Any] | None = None
        checks: list[dict[str, Any]] = []
        for trial in trials:
            alpha = stored.get(trial.alpha_id or "")
            got = {m: alpha.get(m) for m in METRICS} if alpha else None
            shown.append(
                {
                    "trial": trial.number,
                    "state": trial.state,
                    "message": trial.message,
                    "alphaId": trial.alpha_id,
                    "settingsSent": trial.settings,
                    "metrics": got,
                }
            )
            if got is not None and metrics is None:
                metrics = got
                checks = _checks(alpha.get("checks") if alpha else None)
        state = "waiting"
        if metrics is not None:
            state = "complete"
        elif trials and all(t.state == TrialState.FAIL for t in trials):
            state = "failed"
        elif study is not None and study.status in {StudyStatus.IDLE, StudyStatus.QUEUED}:
            state = "not_started"
        # The settings BRAIN actually ran must be the ones the candidate carries.
        mismatch = [
            k
            for t in trials
            for k, v in (row.settings or {}).items()
            if t.settings and k in t.settings and t.settings[k] != v
        ]
        return base | {
            "state": state,
            "taskId": row.study_id,
            "taskStatus": study.status if study else None,
            "message": {
                "not_started": "The task exists but has not been started. Start it in Tasks.",
                "waiting": "The simulation has not returned a result yet.",
                "failed": "BRAIN did not return a result for this candidate.",
                "complete": "BRAIN's metrics, as returned.",
            }[state],
            "trials": shown,
            "metrics": metrics,
            "checks": checks,
            "settingsMismatch": sorted(set(mismatch)),
            "diagnostics": diagnose(metrics, checks) if metrics else [],
            "source": "BRAIN" if metrics else None,
        }

    async def lineage(self, candidate_id: str) -> dict[str, Any]:
        """Every version of the idea, with the real metrics where there are any, and the
        multiple-testing warning. A difference shown is observed, never called an improvement."""
        row = await self._candidate_row(candidate_id)
        versions = await self.list_candidates(row.idea_id)
        tested = []
        for v in versions:
            if v["studyId"] is not None:
                r = await self.results(v["id"])
                tested.append(
                    {
                        "candidateId": v["id"],
                        "version": v["version"],
                        "parentId": v["parentId"],
                        "kind": v["variantKind"],
                        "state": r["state"],
                        "metrics": r["metrics"],
                    }
                )
        count = len([v for v in versions if v["expression"]])
        warning = (
            f"{count} expressions have been written for this idea. Each one you simulate and "
            "choose between is another test; the best of many is likely to look better "
            "in-sample than it will later. Prefer changes you can justify economically."
            if count >= LINEAGE_WARNING
            else None
        )
        return {
            "ideaId": row.idea_id,
            "versions": len(versions),
            "expressions": count,
            "tested": tested,
            "multipleTestingWarning": warning,
            "note": "Differences between in-sample metrics are observations, not evidence of "
            "improvement.",
        }

    # --- refinement ------------------------------------------------------------------------

    async def start_refinement(self, candidate_id: str, model_ref: str | None) -> dict[str, Any]:
        row = await self._candidate_row(candidate_id)
        result = await self.results(candidate_id)
        if result["state"] != "complete":
            raise ResearchError(
                "not_simulated",
                "Refinement starts from real results. This candidate has none yet: "
                f"{result['message']}",
                409,
            )
        scope = validation.scope_of(row.scope)
        await self.check_scope(scope)
        model = await self._model(model_ref)
        operators = await self._operators() or []
        if not operators:
            raise ResearchError("operators_unread", "Your BRAIN operators are not loaded.", 409)
        idea = await self._idea(row.idea_id)
        lineage = await self.lineage(candidate_id)
        design = await self.design_row(row.design_id) if row.design_id else None
        fields = (
            matcher.usable_fields(
                await matcher.resolve(
                    self.queries,
                    scope,
                    design.mapping,
                    design.substitutions,
                    design.accepted_proxies,
                )
            )
            if design
            else []
        )
        job_id = await self.research.new_job(
            kind="refine",
            source_id=idea["sourceId"],
            input_ref=idea["title"][:200],
            model_ref=model.ref,
            provider=model.provider,
            model_id=model.id,
            params={"candidateId": candidate_id, "scope": row.scope},
            message="Queued.",
        )

        async def work(job_id: str) -> None:
            await self._refine_work(
                job_id, idea, row, result, lineage, fields, operators, model.ref
            )

        self.research.launch(job_id, f"Refinement · {idea['title'][:40]}", work)
        return {"jobId": job_id, "message": "Refinement started. Its proposals will be drafts."}

    async def _refine_work(
        self,
        job_id: str,
        idea: dict[str, Any],
        row: AlphaCandidate,
        result: dict[str, Any],
        lineage: dict[str, Any],
        fields: list[dict[str, Any]],
        operators: list[dict[str, Any]],
        model_ref: str,
    ) -> None:
        shown = [
            {"name": o["name"], "definition": o.get("definition", "")}
            for o in operators
            if o.get("name")
        ][:80]
        await self.research.progress(job_id, 0.2, "Asking the model for refinements.")
        parsed = await self._generate(
            job_id,
            user=dsn.refinement_prompt(
                idea=idea,
                candidate={
                    "expression": row.expression,
                    "settings": row.settings,
                    "version": row.version,
                },
                metrics=result["metrics"] or {},
                checks=result["checks"],
                lineage={
                    k: lineage[k] for k in ("versions", "expressions", "multipleTestingWarning")
                },
                fields=fields,
                operators=shown,
            ),
            model_ref=model_ref,
            schema=dsn.REFINEMENT_SCHEMA,
            kind=dsn.Refinement,
        )
        if parsed is None:
            return
        ids: list[str] = []
        for proposal in parsed.proposals:
            settings = {**(row.settings or {}), **proposal.settings_changes}
            ids.append(
                await self._store_candidate(
                    idea=idea,
                    scope=row.scope,
                    expression=proposal.expression,
                    settings=settings,
                    design_id=row.design_id,
                    parent_id=row.id,
                    kind="refinement",
                    explanation={
                        "name": proposal.name,
                        "refinementKind": proposal.kind,
                        "rationale": proposal.rationale,
                        "economicJustification": proposal.economic_justification,
                        "overfittingRisk": proposal.overfitting_risk,
                    },
                    changes=proposal.changes_from_parent
                    + (
                        [f"Settings changed: {', '.join(sorted(proposal.settings_changes))}."]
                        if proposal.settings_changes
                        else []
                    ),
                    limitations=[
                        "Proposed after seeing the parent's in-sample results: it is not "
                        "independent "
                        "of them and its metrics will be optimistic.",
                        f"Model's overfitting note: {proposal.overfitting_risk}",
                    ],
                    origin="refinement",
                    model_ref=model_ref,
                    extra_provenance={"refinedFrom": row.id, "parentMetricsAreInSample": True},
                )
            )
        await self.research.set_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"{len(ids)} draft refinement(s) written."
            if ids
            else "No refinement proposed.",
            result={
                "candidateIds": ids,
                "commentary": parsed.commentary,
                "commentarySource": "LLM",
                "noProposalReason": parsed.no_proposal_reason,
            },
            finished_at=utcnow(),
        )

    # --- export ----------------------------------------------------------------------------

    async def export(self, candidate_id: str) -> dict[str, Any]:
        row = await self._candidate_row(candidate_id)
        idea = await self._idea(row.idea_id)
        return {
            "candidate": self.candidate_view(row),
            "idea": {
                "id": idea["id"],
                "title": idea["title"],
                "hypothesis": idea.get("economicHypothesis"),
                "source": idea.get("source"),
            },
            "results": await self.results(candidate_id),
            "exportedAt": utcnow().isoformat(),
            "note": "Static validation only unless results.metrics is present. Metrics are "
            "BRAIN's.",
        }


def _checks(raw: Any) -> list[dict[str, Any]]:
    """BRAIN's own check results for an Alpha, as stored, or nothing."""
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [
        {
            "name": c.get("name"),
            "result": c.get("result"),
            "limit": c.get("limit"),
            "value": c.get("value"),
        }
        for c in data
        if isinstance(c, dict)
    ]


def _note(source: str, text: str) -> dict[str, str]:
    return {"source": source, "text": text}


def diagnose(metrics: dict[str, Any] | None, checks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Plain observations read from BRAIN's returned numbers and checks. Rule-based, not a
    model: each says which number it reads. Absent numbers are said to be absent."""
    if not metrics:
        return []
    notes = [
        _note(
            "BRAIN check",
            f"BRAIN's {c.get('name')} check failed "
            f"(value {c.get('value')}, limit {c.get('limit')}).",
        )
        for c in checks
        if str(c.get("result")).upper() == "FAIL"
    ]

    def num(key: str) -> float | None:
        v = metrics.get(key)
        return float(v) if isinstance(v, int | float) else None

    sharpe, turnover, fitness = num("sharpe"), num("turnover"), num("fitness")
    if sharpe is None:
        notes.append(_note("missing", "BRAIN did not return a Sharpe for this simulation."))
    elif sharpe < 1.0:
        notes.append(
            _note(
                "Sharpe",
                f"Sharpe {sharpe:.2f} is low in-sample; BRAIN's usual submission bar is higher, "
                "so the signal may be weak or noisy here.",
            )
        )
    if turnover is not None and turnover > 0.7:
        notes.append(
            _note(
                "Turnover",
                f"Turnover {turnover:.0%} is high: trading costs would erode returns. More decay "
                "or a slower window may help; it is not known until simulated.",
            )
        )
    if fitness is not None and sharpe is not None and fitness < 1.0 <= sharpe:
        notes.append(
            _note(
                "Fitness",
                f"Fitness {fitness:.2f} is low relative to Sharpe, usually because of turnover "
                "or small returns.",
            )
        )
    test = num("test_sharpe")
    if test is not None and sharpe is not None and test < sharpe * 0.5:
        notes.append(
            _note(
                "Out-of-sample",
                f"Test Sharpe {test:.2f} is well below in-sample {sharpe:.2f}, "
                "a sign of overfitting.",
            )
        )
    for key, label in (("self_correlation", "self"), ("prod_correlation", "production")):
        v = num(key)
        if v is not None and v > 0.7:
            notes.append(
                _note(
                    "Correlation",
                    f"Correlation with your {label} alphas is {v:.2f}, which is high.",
                )
            )
    return notes
