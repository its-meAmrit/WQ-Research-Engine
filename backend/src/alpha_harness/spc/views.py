"""What the SPC API sends and takes, in the camelCase the screens read."""

from datetime import datetime  # noqa: TC003  (pydantic resolves annotations at runtime)
from typing import Any, Literal

from pydantic import Field

from ..schemas import Out
from .schema import FREQUENCIES

Frequency = Literal["daily", "weekly", "monthly", "quarterly"]
ResearchMode = Literal["off", "auto", "required"]
CheckStatus = Literal["pass", "warn", "fail"]

assert set(FREQUENCIES) == {"daily", "weekly", "monthly", "quarterly"}  # noqa: S101


# --- requests ---------------------------------------------------------------------------


class PromptCreate(Out):
    name: str = Field(min_length=1, max_length=200)
    prompt_text: str = Field(min_length=1)
    description: str = Field(default="", max_length=2000)
    universe: str = Field(default="", max_length=200)
    theme: str = Field(default="", max_length=200)
    update_frequency: Frequency = "weekly"
    #: 0.00 to 1.00, two decimal places. 0 means inactive.
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    preferred_model: str | None = None
    model_version: str | None = None
    #: The Prompt Studio options that produced the text, kept with version 1.
    settings: dict[str, Any] | None = None
    note: str = Field(default="", max_length=300)


class PromptPatch(Out):
    """Every field optional. A changed ``prompt_text`` adds a version; the rest edit the prompt."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    prompt_text: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, max_length=2000)
    universe: str | None = Field(default=None, max_length=200)
    theme: str | None = Field(default=None, max_length=200)
    update_frequency: Frequency | None = None
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    preferred_model: str | None = None
    #: Send an empty string to clear the model.
    clear_model: bool = False
    model_version: str | None = None
    archived: bool | None = None
    preferred_version_id: int | None = None
    settings: dict[str, Any] | None = None
    note: str = Field(default="", max_length=300)


class RunRequest(Out):
    version_number: int | None = Field(default=None, ge=1)
    model: str | None = None
    research: ResearchMode = "auto"


# --- responses --------------------------------------------------------------------------


class EvidenceOut(Out):
    url: str
    title: str
    source_type: str
    published_at: str | None
    retrieved_at: datetime
    snippet: str
    query: str
    stale: bool


class PositionOut(Out):
    position: int
    company_name: str
    ticker: str
    isin: str
    mic: str
    confidence_score: float
    investment_horizon: float
    return_prediction: float
    isin_status: str
    mic_status: str
    mapping_status: str


class RunBrief(Out):
    id: int
    prompt_id: int
    prompt_name: str
    version_number: int
    status: str
    #: Still being worked on in this process.
    active: bool
    valid: bool | None
    model_ref: str
    message: str
    error_code: str | None
    error_message: str | None
    positions: int
    corrects_run_id: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    usage: dict[str, Any] | None


class RunDetail(RunBrief):
    prompt_snapshot: str
    meta: dict[str, Any]
    raw_output: str | None
    sample_json: str | None
    validation: dict[str, Any] | None
    research: dict[str, Any] | None
    evidence: list[EvidenceOut]
    portfolio: list[PositionOut]


class PromptVersionOut(Out):
    id: int
    number: int
    prompt_text: str
    characters: int
    note: str
    settings: dict[str, Any] | None
    created_at: datetime
    runs: int
    valid_runs: int


class PromptOut(Out):
    id: int
    name: str
    description: str
    universe: str
    theme: str
    update_frequency: str
    weight: float
    preferred_model: str | None
    model_version: str | None
    archived: bool
    preferred_version_id: int | None
    latest_version: int
    latest_version_id: int
    characters: int
    runs: int
    last_run: RunBrief | None
    created_at: datetime
    updated_at: datetime


class PromptDetail(PromptOut):
    prompt_text: str
    versions: list[PromptVersionOut]


class DiffLine(Out):
    kind: Literal["add", "del", "ctx"]
    text: str


class VersionComparison(Out):
    a: PromptVersionOut
    b: PromptVersionOut
    lines: list[DiffLine]
    added: int
    removed: int
    character_change: int


class Check(Out):
    id: str
    label: str
    status: CheckStatus
    detail: str


class ExportOut(Out):
    prompt_id: int
    prompt_name: str
    version_number: int | None
    run_id: int | None
    #: No failing check: the package can be taken to the platform.
    ready: bool
    checks: list[Check]
    schema_verified: bool
    schema_note: str
    prompt_text: str | None
    sample_json: str | None
    package: dict[str, Any] | None


class ModelChoice(Out):
    id: str
    ref: str
    provider: str
    remaining_today: int


class SpcOptions(Out):
    models: list[ModelChoice]
    default_model: str | None
    frequencies: list[str]
    approaches: dict[str, str]
    research: dict[str, Any]
    #: The documented limits, and how many of the local prompts count against them.
    limits: dict[str, Any]
    schema_profile: dict[str, Any]
    reference: dict[str, Any]


class Analytics(Out):
    run_id: int
    prompt_id: int
    prompt_name: str
    summary: dict[str, Any]
    previous_run_id: int | None
    comparison: dict[str, Any] | None
    performance: dict[str, Any]
    scoring: list[str]


class Overlap(Out):
    overlap: dict[str, Any]
