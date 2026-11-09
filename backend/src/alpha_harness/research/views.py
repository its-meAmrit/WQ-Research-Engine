"""The wire shapes of the Research Intelligence API.

The service builds plain camelCase dicts; these models are what the generated frontend types
come from. Free-form JSON the pipeline stores whole (an assessment, the evidence list) is
typed loosely here and narrowed on the frontend.
"""

from typing import Any, Literal

from pydantic import Field

from ..schemas import Out

AccessStatus = Literal["full_text", "abstract_only", "metadata_only"]
Readiness = Literal["ready", "adaptation", "blocked"]
Support = Literal["source_backed", "partially_supported", "speculative"]


class SectionOut(Out):
    id: str
    heading: str
    level: int
    page_start: int
    page_end: int


class DocumentOut(Out):
    pages: int
    sections: list[SectionOut]
    tables: list[dict[str, Any]]
    equations: list[dict[str, Any]]
    stats: dict[str, Any]
    parser: str


class SourceOut(Out):
    id: str
    source_type: str
    import_method: str
    title: str
    authors: list[str]
    abstract: str | None
    canonical_url: str | None
    doi: str | None
    arxiv_id: str | None
    published_at: str | None
    venue: str | None
    topics: list[str]
    retrieved_at: str | None
    access_status: AccessStatus
    access_detail: str
    license: str | None
    content_hash: str | None
    extraction_status: str
    page_count: int | None
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    idea_count: int
    saved_idea_count: int
    has_text: bool
    created_at: str | None


class SourceDetail(SourceOut):
    document: DocumentOut | None


class PageOut(Out):
    page: int
    text: str
    pages: int


class JobOut(Out):
    id: str
    kind: str
    source_id: str | None
    input_ref: str
    status: str
    active: bool
    progress: float
    message: str
    error_category: str | None
    error_message: str | None
    error_detail: list[dict[str, Any]] | None
    model_ref: str | None
    provider: str | None
    model_id: str | None
    params: dict[str, Any]
    usage: dict[str, Any] | None
    result: dict[str, Any] | None
    raw_output: str | None
    cancel_requested: bool
    created_at: str | None
    started_at: str | None
    finished_at: str | None


class StartedOut(Out):
    source_id: str | None
    job_id: str | None
    #: True when it was already there and nothing new was started.
    existing: bool
    message: str


class IdeaSource(Out):
    id: str
    title: str
    authors: list[str]
    published_at: str | None
    access_status: AccessStatus
    doi: str | None


class CandidateOut(Out):
    id: str
    idea_id: str
    expression: str | None
    scope: dict[str, Any]
    validation_outcome: str
    validation_detail: dict[str, Any] | None
    version: int
    status: str
    provenance: dict[str, Any]
    created_at: str | None


class IdeaOut(Out):
    id: str
    source_id: str
    extra_source_ids: list[str]
    source: IdeaSource | None
    job_id: str | None
    title: str
    topic: str
    main_finding: str
    economic_hypothesis: str
    anomaly: str
    signal_definition: str
    signal_direction: str
    required_variables: list[dict[str, Any]]
    reported_formula: str | None
    formula_needs_review: bool
    lookback: str
    holding_period: str
    universe: str
    point_in_time: str
    methodology: str
    limitations: list[str]
    implementation_variants: list[str]
    lookahead_concerns: list[str]
    missing_information: list[str]
    claims: dict[str, Any]
    evidence: list[dict[str, Any]]
    support: Support
    evidence_scope: Literal["paper", "abstract"]
    brain_assessment: dict[str, Any] | None
    readiness: Readiness | None
    status: Literal["new", "saved", "rejected"]
    note: str
    created_at: str | None
    updated_at: str | None


class IdeaDetail(IdeaOut):
    candidates: list[CandidateOut]


class SearchPaper(Out):
    title: str
    authors: list[str]
    abstract: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    published_at: str | None
    venue: str | None
    provider: str
    citations: int | None
    access: AccessStatus
    open_copy: bool
    imported_as: str | None


class WebHit(Out):
    title: str
    url: str
    snippet: str
    kind: str


class SearchOut(Out):
    query: str
    papers: list[SearchPaper]
    web: list[WebHit]
    notes: list[str]


class PreviewSection(Out):
    id: str
    heading: str
    included: bool


class PreviewOut(Out):
    source_id: str
    abstract_only: bool
    basis: str
    sections: list[PreviewSection]
    chunks: int
    requests: int
    chars_sent: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    truncated: bool
    chars_left_out: int


class ModelChoice(Out):
    id: str
    ref: str
    provider: str
    remaining_today: int


class Limits(Out):
    max_pdf_mb: int
    max_pages: int
    default_char_budget: int
    max_ideas: int


class ResearchOptions(Out):
    models: list[ModelChoice]
    limits: Limits
    web_search: bool
    topics: list[str]


class ScopeIn(Out):
    instrument_type: str = "EQUITY"
    region: str = Field(min_length=2, max_length=8)
    delay: int = Field(ge=0, le=1)
    universe: str = Field(min_length=2, max_length=24)


class ImportRequest(Out):
    reference: str = Field(min_length=3, max_length=2048)


class CandidateImport(Out):
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None


class AnalyzeRequest(Out):
    model_ref: str | None = None
    char_budget: int | None = Field(default=None, ge=2_000, le=200_000)
    scope: ScopeIn | None = None
    force: bool = False


class IdeaPatch(Out):
    status: Literal["new", "saved", "rejected"] | None = None
    note: str | None = Field(default=None, max_length=5_000)


class ScopeRequest(Out):
    scope: ScopeIn
