"""Operational state, stored in SQLite.

Everything here is small, transactional, and must survive a crash. The bulk analytical
data — the data-field catalog — lives in DuckDB instead (see :mod:`.duck`).

Schema evolution is additive only, applied by :func:`.sqlite.migrate` on every startup:
missing columns and indexes as well as missing tables, which ``create_all`` alone does
*not* add.

Never drop or rename a column on ``simulation_record``. Losing a row there means losing
the ability to cancel a running simulation, which is the exact failure this project
exists to prevent.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..brain.platform import INSTRUMENT_TYPE


def utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp stored as UTC and read back UTC-aware.

    SQLite has no zone: the stock type drops an aware value's offset without converting,
    so ``12:00-04:00`` came back as ``12:00``. Converting on the way in makes the stored
    text UTC; a naive value is refused rather than guessed at.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"Refusing to store naive datetime {value!r}; use utcnow().")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON, datetime: UtcDateTime}


# --- credentials & session ------------------------------------------------


class Credential(Base):
    """A BRAIN login. The password is sealed by :class:`~alpha_harness.sealing.Sealer`."""

    __tablename__ = "credential"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    #: Set when BRAIN refused this login or the person signed out: the app does not sign in
    #: with it by itself again until the person does. Each refused sign-in spends BRAIN's
    #: lockout budget.
    silent_off_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class BrainSessionRow(Base):
    """A cached cookie jar, so a restart does not cost another proof-of-work solve.

    Cookies are bearer credentials, so the jar is sealed just like the password.
    """

    __tablename__ = "brain_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    credential_id: Mapped[int] = mapped_column(ForeignKey("credential.id", ondelete="CASCADE"))
    cookies_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    #: Never read. Kept because it is NOT NULL with no server default: dropped here, a previous
    #: build the launcher falls back to could not add it again, and would not start.
    permissions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


# --- simulations ----------------------------------------------------------


class SimStatus(StrEnum):
    """Local lifecycle. Distinct from the platform's own status.

    ``PENDING`` is the crash window: the request is in flight and the platform id is not
    known yet, so a row left there after a restart may be a real running simulation and
    must be reconciled rather than discarded.

    ``ORPHANED`` means it was sent but no id came back. Not final:
    :mod:`alpha_harness.engine.reconcile` adopts the alpha it produced or queues the
    request again, because treating it as final either loses the alpha or spends the
    quota twice.
    """

    QUEUED = "QUEUED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    ORPHANED = "ORPHANED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self not in ACTIVE


#: Not yet finished; every other status is final. One exception: a cancel BRAIN refuses
#: moves ``CANCELLED`` back to ``RUNNING``, because the simulation is in fact still running
#: and still has to be polled (:func:`.engine.lifecycle.cancel_after_lost_race`).
#: ``ORPHANED`` belongs here because an identical request arriving while its outcome is
#: reconciled must share the row rather than pay for a second run.
ACTIVE = (SimStatus.QUEUED, SimStatus.PENDING, SimStatus.RUNNING, SimStatus.ORPHANED)


class SimulationRecord(Base):
    """One simulation we asked BRAIN to run.

    The row is written **before** the HTTP request and updated with ``platform_id`` the
    moment a ``201`` lands, so a simulation can always be cancelled.

    For a multi-simulation, ``platform_id`` is the *parent* id — the only cancellable
    handle. Child ids only exist once the parent completes and are stored in
    ``child_ids``.
    """

    __tablename__ = "simulation_record"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Platform identity
    platform_id: Mapped[str | None] = mapped_column(String(64), index=True)
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    parent_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL")
    )
    child_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_batch: Mapped[bool] = mapped_column(default=False)
    #: Child ids exist only once the parent finishes, and resolving them costs one
    #: request each. This marks that work as done so it happens exactly once.
    children_expanded: Mapped[bool] = mapped_column(default=False)

    # What we asked for
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    expression: Mapped[str | None] = mapped_column(Text)

    # Batch-packing key: children of one multi-simulation must agree on all of these.
    sim_type: Mapped[str] = mapped_column(String(16), default="REGULAR")
    instrument_type: Mapped[str] = mapped_column(String(16), default=INSTRUMENT_TYPE)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(16), default="FASTEXPR")
    universe: Mapped[str | None] = mapped_column(String(32))

    # Where it came from — lets the UI group work and apply per-task slot quotas.
    task: Mapped[str] = mapped_column(String(64), default="manual", index=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(16), default=SimStatus.PENDING, index=True)
    platform_status: Mapped[str | None] = mapped_column(String(16))
    progress: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(Text)
    #: Times reconcile queued this row again after its send's answer was lost.
    orphan_requeues: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, index=True)
    submitted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    #: When the POST that carried this row began — ``created_at`` is when it was queued,
    #: which can be hours earlier. Bounds the search for a send whose answer was lost.
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_polled_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index("ix_sim_active", "status", "created_at"),
        Index("ix_sim_batch_key", "sim_type", "instrument_type", "region", "delay", "language"),
        # For the engine's every-tick reads, which otherwise scan every row ever simulated.
        Index("ix_sim_unexpanded", "is_batch", "children_expanded"),
        Index("ix_sim_finished", "status", "finished_at"),
        Index("ix_sim_sent_day", "is_batch", "submitted_at"),
    )

    @property
    def elapsed_seconds(self) -> float | None:
        """Wall time since submission — what the matrix cell counts up."""
        if self.submitted_at is None:
            return None
        return ((self.finished_at or utcnow()) - self.submitted_at).total_seconds()


class DedupEntry(Base):
    """Hash of a canonical simulation payload -> the alpha it produced.

    Guards the daily quota against re-simulating something already run, as BRAIN recommends.
    """

    __tablename__ = "dedup_entry"

    request_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    alpha_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class TaskQuota(Base):
    """How many of the concurrent slots a named task may hold at once.

    Research runs in parallel streams — a manual experiment, a template sweep, an
    optimizer study — and without a cap one of them starves the others. A task with no
    row here competes freely for whatever is unallocated.
    """

    __tablename__ = "task_quota"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    max_slots: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


class Preference(Base):
    """One choice made in Settings, by name. A missing row means the default.

    A row per choice rather than a column each: a choice added later needs no migration, and
    a build older than it simply never reads the row.
    """

    __tablename__ = "preference"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


class QuotaSnapshot(Base):
    """A reading of ``X-RateLimit-*`` from a simulation POST.

    Kept as a series rather than a single row so the UI can show quota burn over the day.
    """

    __tablename__ = "quota_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    limit_total: Mapped[int | None] = mapped_column(Integer)
    remaining: Mapped[int | None] = mapped_column(Integer)
    reset_seconds: Mapped[float | None] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, index=True)


class BrainCache(Base):
    """A BRAIN read kept so it is not asked for again: an Alpha's series, its correlations.

    Correlations and recordsets are Retry-After jobs, while an Alpha's in-sample history
    never changes until the period rolls. So the Alpha page reads from here and only goes
    back to BRAIN when asked to refresh.
    """

    __tablename__ = "brain_cache"

    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    body: Mapped[Any] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


# --- catalog sync ---------------------------------------------------------


class SyncStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SyncRun(Base):
    """One catalog crawl of a single (instrumentType, region, delay, universe) tuple."""

    __tablename__ = "sync_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_type: Mapped[str] = mapped_column(String(16), default=INSTRUMENT_TYPE)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(16), default=SyncStatus.RUNNING, index=True)
    phase: Mapped[str | None] = mapped_column(String(32))
    #: Unused. Retired once every install has run a build with these server defaults: a build
    #: without them cannot put the columns back, so the launcher could not fall back to it.
    cursor_offset: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    truncated_datasets: Mapped[list[Any]] = mapped_column(JSON, default=list, server_default="[]")

    datasets_synced: Mapped[int] = mapped_column(Integer, default=0)
    fields_synced: Mapped[int] = mapped_column(Integer, default=0)
    #: No longer counted. Kept because it is NOT NULL with no server default: dropped here, a
    #: previous build the launcher falls back to could not add it again, and would not start.
    categories_synced: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (Index("ix_sync_tuple", "instrument_type", "region", "delay", "universe"),)


# --- platform metadata cache ---------------------------------------------


class MetadataCache(Base):
    """Cached platform metadata (``OPTIONS /simulations``, ``/operators``).

    Refreshed on login rather than hardcoded — regions and universes change.
    """

    __tablename__ = "metadata_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


# --- template studio (Phase 3) -------------------------------------------


class Template(Base):
    """A YAML alpha template with typed variables."""

    __tablename__ = "template"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    origin: Mapped[str] = mapped_column(String(16), default="human")  # human | llm
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


# --- optimization (Phase 4) ----------------------------------------------


class StudyStatus(StrEnum):
    IDLE = "IDLE"
    #: Told to run, waiting for its cores to fit in the free slots.
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class TrialState(StrEnum):
    """Mirrors Optuna's states, plus the two this application adds.

    ``QUEUED`` and ``RUNNING`` both mean "asked for, not yet told" as far as Optuna is
    concerned; the distinction is whether a simulation has actually started, which is
    what the UI needs to show.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"


class Study(Base):
    """One optimization run over one template.

    ``template_source`` is a *snapshot*, not a reference. A study is a record of a search
    over a particular space; if the template is edited afterwards the trials stop meaning
    anything, so the space is frozen at the moment the study is created.

    Optuna itself is rebuilt from the trial rows on demand rather than persisted
    separately. One source of truth, and the trials stay joinable to the simulations that
    produced them.
    """

    __tablename__ = "study"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    template_id: Mapped[int | None] = mapped_column(ForeignKey("template.id", ondelete="SET NULL"))
    template_name: Mapped[str | None] = mapped_column(String(128))
    #: What the user calls the task, shown in place of its lab and template.
    label: Mapped[str | None] = mapped_column(String(128))
    template_source: Mapped[str] = mapped_column(Text)

    sampler: Mapped[str] = mapped_column(String(32), default="nsga3")
    sampler_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    objectives: Mapped[list[Any]] = mapped_column(JSON, default=list)
    directions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    seed: Mapped[int | None] = mapped_column(Integer)

    #: Trials asked for per round. Kept a multiple of ten so a round fills whole
    #: multi-simulations rather than leaving slots holding part-empty batches.
    batch_size: Mapped[int] = mapped_column(Integer, default=10)
    max_trials: Mapped[int] = mapped_column(Integer, default=80)
    #: Slot-quota group, so a study cannot starve manual work.
    task: Mapped[str] = mapped_column(String(64), default="study")

    status: Mapped[str] = mapped_column(String(16), default=StudyStatus.IDLE, index=True)
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    #: First time it began running. A task can wait as IDLE for days, so this is what "how
    #: long did it take" means; ``created_at`` would count the waiting too.
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class Trial(Base):
    """One point in the search space, and what it turned into.

    The link to ``simulation_record`` is the whole reason trials live here rather than in
    Optuna's own storage: "which parameters produced this alpha" and "which simulation is
    this trial waiting on" are the two questions actually asked of a running study.
    """

    __tablename__ = "trial"

    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("study.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)

    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: What Optuna drew each value from: every choice of every categorical, so a field searched
    #: over a few thousand fields stores them all, on every trial — 127 KB a trial was measured.
    #: Deferred, so only the two readers that rebuild the search load it; decoding it on every
    #: row a screen or a harvest reads held the event loop for seconds.
    distributions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, deferred=True)
    expression: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    state: Mapped[str] = mapped_column(String(16), default=TrialState.QUEUED, index=True)
    values: Mapped[list[Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    message: Mapped[str | None] = mapped_column(Text)

    simulation_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL"), index=True
    )
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Evolution Lab only: 0 for the seeds, then one per bred generation.
    generation: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        UniqueConstraint("study_id", "number", name="uq_trial_number"),
        Index("ix_trial_open", "study_id", "state"),
    )


# --- conversations --------------------------------------------------------


class ChatThread(Base):
    """One conversation with the assistant.

    Persisted because a consultant who closes the laptop mid-thought should not lose it,
    and because the scope is part of the conversation: "which fields" means nothing
    without knowing which market is being talked about.
    """

    __tablename__ = "chat_thread"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(160))
    instrument_type: Mapped[str] = mapped_column(String(16), default=INSTRUMENT_TYPE)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, index=True
    )


class ChatMessage(Base):
    """One turn. ``meta`` carries the picks, model and cost of an assistant turn."""

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("chat_thread.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class ApiKey(Base):
    """An assistant key. Sealed at rest; only a masked hint is ever shown.

    ``provider`` says whose key it is. Rotation is per provider: a Groq key cannot answer
    a request for a Gemini model, and offering it would spend a retry to learn that.
    """

    __tablename__ = "api_key"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32), default="google", server_default="google")
    key_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    hint: Mapped[str] = mapped_column(String(32))
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    #: The user's own daily request ceiling, overriding the model's when set. Required for a
    #: paid provider, which has no ceiling of its own, and optional for a free one.
    daily_limit: Mapped[int | None] = mapped_column(nullable=True)
    last_ok_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class KeyUsage(Base):
    """Local budget ledger per (key, model, quota day).

    Providers expose no remaining-quota endpoint, so requests per minute and per day are
    tracked client-side or rotation is guesswork. ``day`` is the date in the model's own
    reset time zone, since providers do not agree on when a day starts.
    """

    __tablename__ = "key_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key_id: Mapped[int] = mapped_column(ForeignKey("api_key.id", ondelete="CASCADE"))
    model: Mapped[str] = mapped_column(String(64))
    day: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD, America/Los_Angeles
    requests: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (UniqueConstraint("api_key_id", "model", "day", name="uq_key_model_day"),)


class LLMModel(Base):
    """A model the user has set up, with the limits they read off their provider.

    Nothing about models is built in: which ones exist and what each allows change faster
    than any table here could keep up. Limits apply per key, so a second account doubles
    them. A model id is unique per provider: two providers may serve the same one.
    """

    __tablename__ = "llm_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(200))
    requests_per_minute: Mapped[int] = mapped_column(Integer)
    requests_per_day: Mapped[int] = mapped_column(Integer)
    #: The IANA time zone whose midnight starts the provider's new day for this model.
    reset_timezone: Mapped[str] = mapped_column(
        String(64), default="America/Los_Angeles", server_default="America/Los_Angeles"
    )
    #: The most tokens one Power Pool prompt may spend on this model, or null for the default.
    max_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    # An index rather than a table constraint, so a database that already has the table
    # gains it on startup: SQLite cannot add a constraint to a live table.
    __table_args__ = (Index("uq_llm_model_provider_model", "provider", "model", unique=True),)


# --- SPC Research Lab -----------------------------------------------------
#
# Independent of every simulation table above: an SPC prompt is a text thesis an LLM turns
# into a stock portfolio, not a Fast Expression, and no BRAIN simulation ever runs for one.


class SpcRunStatus(StrEnum):
    """Local lifecycle of one SPC test run. Distinct from anything BRAIN reports."""

    QUEUED = "queued"
    RESEARCHING = "researching"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (SpcRunStatus.COMPLETED, SpcRunStatus.FAILED, SpcRunStatus.CANCELLED)


class SpcPrompt(Base):
    """A prompt the consultant is developing for the Systematic Prediction Challenge.

    Holds the submission metadata that can change without changing the prompt's text. The text
    itself lives in :class:`SpcPromptVersion`, so editing it never rewrites history.
    """

    __tablename__ = "spc_prompt"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    universe: Mapped[str] = mapped_column(String(200), default="", server_default="")
    theme: Mapped[str] = mapped_column(String(200), default="", server_default="")
    #: daily | weekly | monthly | quarterly
    update_frequency: Mapped[str] = mapped_column(
        String(16), default="weekly", server_default="weekly"
    )
    #: Portfolio weight in hundredths (0..100), so two decimal places never meet a float.
    weight_hundredths: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    #: A model as :func:`~alpha_harness.llm.registry.model_ref` spells it.
    preferred_model: Mapped[str | None] = mapped_column(String(300))
    model_version: Mapped[str | None] = mapped_column(String(100))
    archived: Mapped[bool] = mapped_column(default=False, server_default="0")
    #: The version the consultant marked as the one to submit. Not a foreign key: versions
    #: reference their prompt, and a cycle between the two tables helps nothing.
    preferred_version_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


class SpcPromptVersion(Base):
    """One immutable text of a prompt. Edits add a version; none is ever changed."""

    __tablename__ = "spc_prompt_version"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("spc_prompt.id", ondelete="CASCADE"))
    number: Mapped[int] = mapped_column(Integer)
    prompt_text: Mapped[str] = mapped_column(Text)
    #: The Prompt Studio options that generated this text, or null for one written by hand.
    settings: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    note: Mapped[str] = mapped_column(String(300), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (Index("uq_spc_version_number", "prompt_id", "number", unique=True),)


class SpcRun(Base):
    """One test of a prompt version: what was sent, what came back, and whether it validates.

    The prompt text, model and submission metadata are copied in, not referenced, so editing the
    prompt afterwards cannot change what this run says it tested.
    """

    __tablename__ = "spc_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("spc_prompt.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("spc_prompt_version.id", ondelete="CASCADE"))
    version_number: Mapped[int] = mapped_column(Integer)
    prompt_snapshot: Mapped[str] = mapped_column(Text)
    #: Name, frequency, weight and model version as they were when this run started.
    meta_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    model_ref: Mapped[str] = mapped_column(String(300))
    provider: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(16), default=SpcRunStatus.QUEUED.value)
    #: What it is doing now, or why it stopped.
    message: Mapped[str] = mapped_column(Text, default="", server_default="")
    #: The model's answer, byte for byte.
    raw_output: Mapped[str | None] = mapped_column(Text)
    #: The JSON array text taken out of ``raw_output``: a substring of it, never reformatted.
    sample_json: Mapped[str | None] = mapped_column(Text)
    #: True when the answer passed every blocking check; null until validated.
    valid: Mapped[bool | None] = mapped_column()
    validation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    research: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: The earlier run this one corrects, when it came from a corrective pass.
    corrects_run_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index("ix_spc_run_prompt", "prompt_id", "created_at"),
        Index("ix_spc_run_status", "status"),
    )


class SpcPosition(Base):
    """One security in a run's validated portfolio."""

    __tablename__ = "spc_position"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("spc_run.id", ondelete="CASCADE"))
    #: Its place in the model's array.
    position: Mapped[int] = mapped_column(Integer)
    company_name: Mapped[str] = mapped_column(String(300))
    ticker: Mapped[str] = mapped_column(String(40))
    isin: Mapped[str] = mapped_column(String(12))
    mic: Mapped[str] = mapped_column(String(4))
    confidence_score: Mapped[float] = mapped_column(Float)
    investment_horizon: Mapped[float] = mapped_column(Float)
    return_prediction: Mapped[float] = mapped_column(Float)
    #: How far each identifier was checked: see ``spc.identifiers``.
    isin_status: Mapped[str] = mapped_column(String(24))
    mic_status: Mapped[str] = mapped_column(String(24))
    #: ``verified`` or ``mismatch`` only when a security master the user supplied covered the
    #: ISIN; otherwise ``unverified``. A valid checksum never makes a mapping ``verified``.
    mapping_status: Mapped[str] = mapped_column(
        String(24), default="unverified", server_default="unverified"
    )

    __table_args__ = (
        UniqueConstraint("run_id", "position", name="uq_spc_position_run_pos"),
        Index("ix_spc_position_isin", "isin"),
    )


class SpcEvidence(Base):
    """One source the research step retrieved for a run."""

    __tablename__ = "spc_evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("spc_run.id", ondelete="CASCADE"))
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="", server_default="")
    source_type: Mapped[str] = mapped_column(String(32), default="web", server_default="web")
    #: As the source gave it, or null when it did not say. Never guessed.
    published_at: Mapped[str | None] = mapped_column(String(40))
    retrieved_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    snippet: Mapped[str] = mapped_column(Text, default="", server_default="")
    query: Mapped[str] = mapped_column(Text, default="", server_default="")
    stale: Mapped[bool] = mapped_column(default=False, server_default="0")
    companies: Mapped[list[Any] | None] = mapped_column(JSON)

    __table_args__ = (Index("ix_spc_evidence_run", "run_id"),)


class IntegrationCredential(Base):
    """A key for a web service that is not an LLM (a research or search API).

    Kept apart from :class:`ApiKey` on purpose: that table is built around models, per-model
    budgets and key rotation, and none of it applies to a search API. One row per provider,
    sealed at rest; the UI only ever receives ``hint``.

    ``provider`` is validated against the registry in code rather than a CHECK constraint:
    the migrator never alters a constraint, so adding a provider later must not need one.
    """

    __tablename__ = "integration_credential"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), unique=True)
    key_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    hint: Mapped[str] = mapped_column(String(32))
    #: Salted hash of the key, so the same one cannot be stored twice without keeping a
    #: second recoverable copy of it.
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True, server_default="1")
    #: The user's own ceiling on requests per UTC day. The only thing between a key and an
    #: open-ended bill, so it is always set.
    daily_request_budget: Mapped[int] = mapped_column(Integer, default=100, server_default="100")
    last_checked_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_ok_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_code: Mapped[str | None] = mapped_column(String(32))
    #: What the provider itself said about usage the last time it was asked. Provider's
    #: figures, as of ``checked_at`` inside it; never mixed with the local count.
    reported_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


class IntegrationUsage(Base):
    """Requests this application sent with a credential, per UTC day.

    A local count of what *we* sent. It is not the provider's balance: other tools may share
    the key, and providers bill by credits, not by request.
    """

    __tablename__ = "integration_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    credential_id: Mapped[int] = mapped_column(
        ForeignKey("integration_credential.id", ondelete="CASCADE")
    )
    day: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD, UTC
    requests: Mapped[int] = mapped_column(Integer, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (UniqueConstraint("credential_id", "day", name="uq_integration_usage_day"),)


# --- Research Intelligence ---------------------------------------------------
#
# A library of sources (papers now, scraped articles later) and the testable ideas taken from
# them. Independent of the SPC tables above and of every simulation table: nothing here is a
# prompt, a portfolio or a simulation, and nothing in this phase ever starts one. Ids are
# opaque strings so a source keeps its identity whatever kind of source it is.


def research_id(prefix: str) -> str:
    """``src_3f9a...``: a stable id that says what it names when it turns up in a log."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class ResearchSource(Base):
    """One paper (or, later, article): what it is, where it came from, how much of it we hold.

    ``access_status`` is the honest answer to "did we read this?": ``full_text`` only when the
    whole document was available (an upload, or an open-access copy), otherwise
    ``abstract_only`` or ``metadata_only``. Nothing downstream may treat the lesser two as the
    paper.
    """

    __tablename__ = "research_source"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: research_id("src")
    )
    #: ``academic_paper`` now; ``web_article`` when scraped articles join the library.
    source_type: Mapped[str] = mapped_column(
        String(24), default="academic_paper", server_default="academic_paper"
    )
    #: How it came in: upload | doi | arxiv | url | search.
    import_method: Mapped[str] = mapped_column(
        String(16), default="upload", server_default="upload"
    )
    title: Mapped[str] = mapped_column(Text, default="", server_default="")
    authors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    abstract: Mapped[str | None] = mapped_column(Text)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(255))
    arxiv_id: Mapped[str | None] = mapped_column(String(64))
    #: As the source gave it (``2019``, ``2019-05``, ``2019-05-17``); never completed by guessing.
    published_at: Mapped[str | None] = mapped_column(String(32))
    venue: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[list[Any]] = mapped_column(JSON, default=list)
    retrieved_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    #: full_text | abstract_only | metadata_only
    access_status: Mapped[str] = mapped_column(
        String(24), default="metadata_only", server_default="metadata_only"
    )
    #: Why: "uploaded by you", "open access copy via OpenAlex", "no open full text found".
    access_detail: Mapped[str] = mapped_column(Text, default="", server_default="")
    license: Mapped[str | None] = mapped_column(String(200))
    #: SHA-256 of the PDF bytes, so the same file uploaded twice is one source.
    content_hash: Mapped[str | None] = mapped_column(String(64))
    #: pending | parsed | partial | unreadable | failed | not_applicable
    extraction_status: Mapped[str] = mapped_column(
        String(24), default="pending", server_default="pending"
    )
    page_count: Mapped[int | None] = mapped_column(Integer)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    # Unique where present: SQLite allows many NULLs in a unique index, and a source without a
    # DOI is not a duplicate of another without one.
    __table_args__ = (
        Index("uq_research_source_doi", "doi", unique=True),
        Index("uq_research_source_arxiv", "arxiv_id", unique=True),
        Index("uq_research_source_url", "canonical_url", unique=True),
        Index("uq_research_source_hash", "content_hash", unique=True),
        Index("ix_research_source_created", "created_at"),
    )


class ResearchDocument(Base):
    """A source's parsed text, kept so an analysis can be run again without the file.

    The PDF itself is not kept: the text, its page boundaries and the flags below are what the
    pipeline reads, and a copy of someone's paper has no use after that.
    """

    __tablename__ = "research_document"

    source_id: Mapped[str] = mapped_column(
        ForeignKey("research_source.id", ondelete="CASCADE"), primary_key=True
    )
    #: ``[{"page": 1, "text": "..."}]``, in order, 1-based.
    pages: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: ``[{"id": "s1", "heading": "Data", "level": 1, "page_start": 3, "page_end": 5}]``
    sections: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: Tables PyMuPDF recovered cleanly: ``{"page", "rows"}``.
    tables: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: Equation-looking text, each with ``needs_review``: never turned into a formula by us.
    equations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    parser: Mapped[str] = mapped_column(String(40), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class ResearchJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: Left running by a process that ended: not a failure of the work, and not a result.
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self not in (ResearchJobStatus.QUEUED, ResearchJobStatus.RUNNING)


class ResearchJob(Base):
    """One unit of work on a source: an import or an analysis.

    Written before the work starts and updated as it goes, so a restart can say where it
    stopped and a screen can follow it by reading the row.
    """

    __tablename__ = "research_job"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: research_id("job")
    )
    #: import | analyze
    kind: Mapped[str] = mapped_column(String(16))
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_source.id", ondelete="CASCADE")
    )
    #: What was asked for: a file name, a DOI, an arXiv id or a URL.
    input_ref: Mapped[str] = mapped_column(Text, default="", server_default="")
    status: Mapped[str] = mapped_column(
        String(16), default=ResearchJobStatus.QUEUED.value, server_default="queued"
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    #: What it is doing now, or why it stopped.
    message: Mapped[str] = mapped_column(Text, default="", server_default="")
    #: Stable, machine-readable: ``no_full_text``, ``invalid_model_output``, ``interrupted``, ...
    error_category: Mapped[str | None] = mapped_column(String(48))
    error_message: Mapped[str | None] = mapped_column(Text)
    #: Validation problems found in the model's answer, shown as they are.
    error_detail: Mapped[list[Any] | None] = mapped_column(JSON)
    model_ref: Mapped[str | None] = mapped_column(String(300))
    provider: Mapped[str | None] = mapped_column(String(32))
    model_id: Mapped[str | None] = mapped_column(String(200))
    #: Chunks, character budget and the estimate shown before it started.
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: The model's last answer when it could not be used, clipped: what to read to see why.
    raw_output: Mapped[str | None] = mapped_column(Text)
    #: Same source, model, prompt and budget: a finished job with this key is the answer.
    cache_key: Mapped[str | None] = mapped_column(String(64))
    cancel_requested: Mapped[bool] = mapped_column(default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index("ix_research_job_source", "source_id", "created_at"),
        Index("ix_research_job_status", "status"),
        Index("ix_research_job_cache", "cache_key"),
    )


class ResearchIdea(Base):
    """One testable idea taken from a source, and what we know about running it on BRAIN.

    The fields hold the paper's own content and the model's, apart: ``claims`` keeps what the
    authors said, what the model inferred and what is a new proposal in three lists, and
    ``support`` says whether any author claim was found in the text. An adaptation the model
    wrote is never stored as the authors'.
    """

    __tablename__ = "research_idea"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: research_id("idea")
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("research_source.id", ondelete="CASCADE"))
    #: Other sources that report the same idea; empty until a second source joins.
    extra_source_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    job_id: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text)
    #: value | momentum | quality | growth | size | volatility | liquidity | sentiment |
    #: analyst | event | macro | other
    topic: Mapped[str] = mapped_column(String(24), default="other", server_default="other")
    main_finding: Mapped[str] = mapped_column(Text, default="", server_default="")
    economic_hypothesis: Mapped[str] = mapped_column(Text, default="", server_default="")
    anomaly: Mapped[str] = mapped_column(Text, default="", server_default="")
    signal_definition: Mapped[str] = mapped_column(Text, default="", server_default="")
    #: long_high | long_low | long_short_by_sign | unclear
    signal_direction: Mapped[str] = mapped_column(
        String(24), default="unclear", server_default="unclear"
    )
    required_variables: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reported_formula: Mapped[str | None] = mapped_column(Text)
    #: Whether the formula was read from clean text; a doubtful one is never "fixed" by us.
    formula_needs_review: Mapped[bool] = mapped_column(default=False, server_default="0")
    lookback: Mapped[str] = mapped_column(Text, default="", server_default="")
    holding_period: Mapped[str] = mapped_column(Text, default="", server_default="")
    universe: Mapped[str] = mapped_column(Text, default="", server_default="")
    point_in_time: Mapped[str] = mapped_column(Text, default="", server_default="")
    methodology: Mapped[str] = mapped_column(Text, default="", server_default="")
    limitations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    implementation_variants: Mapped[list[Any]] = mapped_column(JSON, default=list)
    lookahead_concerns: Mapped[list[Any]] = mapped_column(JSON, default=list)
    missing_information: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: ``{"authorClaims": [...], "modelInterpretations": [...], "proposedHypotheses": [...]}``
    claims: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Every cited passage, each marked verified or not against the stored text.
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    #: source_backed | partially_supported | speculative
    support: Mapped[str] = mapped_column(
        String(24), default="speculative", server_default="speculative"
    )
    #: paper | abstract: how much of the source the idea was read from.
    evidence_scope: Mapped[str] = mapped_column(String(12), default="paper", server_default="paper")
    brain_assessment: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: Copied out of the assessment so the library can filter on it. Null: never assessed.
    readiness: Mapped[str | None] = mapped_column(String(24))
    #: new | saved | rejected
    status: Mapped[str] = mapped_column(String(12), default="new", server_default="new")
    note: Mapped[str] = mapped_column(Text, default="", server_default="")
    #: Lower-cased title, hypothesis and variables, for the library's text search.
    search_text: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_research_idea_source", "source_id"),
        Index("ix_research_idea_status", "status", "readiness"),
    )


class AlphaCandidate(Base):
    """A draft handed to the Alpha Builder: an idea, a scope and (later) an expression.

    Draft only in this phase. Nothing creates one with a simulation attached, and no code path
    from here reaches BRAIN: ``validation_outcome`` is a local check, never a result.
    """

    __tablename__ = "alpha_candidate"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: research_id("cand")
    )
    idea_id: Mapped[str] = mapped_column(ForeignKey("research_idea.id", ondelete="CASCADE"))
    #: Null until the Alpha Builder writes one.
    expression: Mapped[str | None] = mapped_column(Text)
    #: ``{"instrumentType", "region", "delay", "universe"}``: exactly where it would run.
    scope: Mapped[dict[str, Any]] = mapped_column(JSON)
    #: not_validated | valid | invalid
    validation_outcome: Mapped[str] = mapped_column(
        String(16), default="not_validated", server_default="not_validated"
    )
    validation_detail: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(12), default="draft", server_default="draft")
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (Index("uq_alpha_candidate_version", "idea_id", "version", unique=True),)
