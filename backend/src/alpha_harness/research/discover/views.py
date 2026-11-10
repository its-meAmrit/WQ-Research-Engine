"""Wire shapes of the Discover API. The service builds plain camelCase dicts; these models are
what the generated frontend types come from."""

from typing import Any, Literal

from pydantic import Field

from ...schemas import Out
from ..views import ModelChoice, ScopeIn  # noqa: TC001 - pydantic reads annotations at runtime


class RegistryEntryOut(Out):
    id: str
    label: str
    domain: str
    category: str
    adapter: str
    enabled: bool
    retrieval_policy: dict[str, Any]
    config: dict[str, Any]
    min_interval_seconds: int
    quality: dict[str, Any]
    builtin: bool
    last_attempt_at: str | None
    last_success_at: str | None
    last_error: str | None
    last_error_at: str | None


class RegistryPatch(Out):
    enabled: bool | None = None
    min_interval_seconds: int | None = Field(default=None, ge=0)
    domains: list[str] | None = Field(default=None, max_length=30)
    quality_note: str | None = Field(default=None, max_length=400)


class FeedIn(Out):
    label: str = Field(max_length=120)
    feed_url: str = Field(min_length=8, max_length=2048)
    min_interval_seconds: int = Field(default=86_400, ge=3_600)


class SearchProviderOut(Out):
    id: str
    label: str
    configured: bool
    enabled: bool
    is_default: bool
    usage: dict[str, Any] | None


class DiscoverLimits(Out):
    max_results: int
    max_queries: int
    max_requests_per_job: int
    max_extract_per_job: int


class DiscoverOptions(Out):
    topics: list[str]
    sources: list[RegistryEntryOut]
    providers: list[SearchProviderOut]
    models: list[ModelChoice]
    limits: DiscoverLimits


class QueryProposalIn(Out):
    topic: str = Field(min_length=3, max_length=300)
    market: str = Field(default="", max_length=120)


class QueryProposal(Out):
    angle: str
    text: str


class EstimateOut(Out):
    provider: str | None
    requests: int
    free: int
    search_credits: float
    extract_credits_if_all_read: float | None
    max_results: int
    note: str
    queries: list[str]
    notes: list[str]


class DiscoverStarted(Out):
    job_id: str
    existing: bool
    message: str


class HitOut(Out):
    id: str
    job_id: str
    query: str
    rank: int
    provider: str
    registry_id: str | None
    url: str
    canonical_url: str | None
    domain: str | None
    title: str
    snippet: str
    published: str | None
    score: float | None
    retrieved_at: str | None
    status: Literal[
        "found", "duplicate", "restricted", "extracted", "rejected", "failed", "imported"
    ]
    status_detail: str
    source_id: str | None
    duplicate_of: str | None
    from_cache: bool
    is_paper: bool


class ExtractIn(Out):
    hit_ids: list[str] = Field(min_length=1, max_length=10)
    model_ref: str | None = None
    scope: ScopeIn | None = None
    depth: Literal["basic", "advanced"] = "basic"


class ExtractStarted(Out):
    job_id: str | None
    existing: bool
    message: str
    imports: list[dict[str, Any]]
