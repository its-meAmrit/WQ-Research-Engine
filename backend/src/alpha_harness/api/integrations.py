"""Web research services: Tavily and Brave Search.

Not generative models, so none of this touches ``/api/llm`` (which keeps its routes). Keys are
sealed at rest and never returned: the only thing that leaves is a masked hint.
"""

from typing import Literal

from fastapi import APIRouter, Response
from pydantic import Field

from ..integrations.base import Capability, ProviderId
from ..integrations.credentials import MAX_DAILY_BUDGET
from ..schemas import Out
from .deps import State

router = APIRouter(prefix="/api/integrations/web", tags=["integrations"])


class CheckCostInfo(Out):
    may_consume_credit: bool
    #: What a connectivity test does to the quota, to show before it is sent.
    description: str


class LocalUsage(Out):
    #: The UTC day counted, YYYY-MM-DD.
    day: str
    requests: int
    daily_budget: int
    remaining_today: int
    resets_in_seconds: int
    #: Says what this number is and is not.
    note: str


class ReportedUsageInfo(Out):
    """What the provider said, as of ``checked_at``. A different thing from :class:`LocalUsage`."""

    checked_at: str
    used: float | None
    limit: float | None
    remaining: float | None
    plan: str | None
    period: str | None


class WebProviderStatus(Out):
    id: ProviderId
    label: str
    description: str
    capabilities: list[Capability]
    key_url: str
    docs_url: str
    check_cost: CheckCostInfo
    configured: bool
    enabled: bool
    #: Masked, e.g. ``tvly-…a1b2``. The key itself is never sent.
    hint: str | None
    is_preferred: bool
    #: The one a search would use right now, from the stored keys.
    is_active: bool
    last_checked_at: str | None
    last_ok_at: str | None
    last_error: str | None
    last_error_code: str | None
    created_at: str | None
    usage: LocalUsage | None
    reported_usage: ReportedUsageInfo | None


class ActiveResearch(Out):
    configured: bool
    provider: ProviderId | None
    label: str | None
    #: ``store`` is the key stored here; ``environment`` is the legacy variable.
    source: Literal["store", "environment"] | None
    #: Something worth knowing about the choice, e.g. that it comes from the environment.
    message: str | None
    #: How to set research up, when it is not.
    help: str | None


class LegacyEnvironment(Out):
    present: bool
    provider: ProviderId | None
    usable: bool
    in_use: bool
    note: str | None


class WebResearchStatus(Out):
    providers: list[WebProviderStatus]
    #: The user's chosen default, or null to let the order decide.
    preferred_provider: ProviderId | None
    active: ActiveResearch
    legacy_environment: LegacyEnvironment


@router.get("")
async def status(state: State) -> WebResearchStatus:
    """Every service, whether it has a key, how it last fared, and which one a search uses."""
    return WebResearchStatus.model_validate(await state.web.status())


class PutKey(Out):
    key: str = Field(max_length=4096, description="The service's API key. Sealed; never returned.")
    daily_request_budget: int | None = Field(
        default=None,
        ge=1,
        le=MAX_DAILY_BUDGET,
        description="Most requests per UTC day this app may send with the key.",
    )


@router.put("/providers/{provider}/key")
async def put_key(provider: ProviderId, body: PutKey, state: State) -> WebProviderStatus:
    """Store a key for a service, or replace the one it has.

    The key is checked for shape only. Whether the service accepts it is what the connectivity
    test is for, because that may cost a credit and so is the user's to start.
    """
    await state.web.store.put(provider, body.key, body.daily_request_budget)
    return WebProviderStatus.model_validate(await state.web.provider_status(provider))


class UpdateProvider(Out):
    enabled: bool | None = None
    daily_request_budget: int | None = Field(default=None, ge=1, le=MAX_DAILY_BUDGET)


@router.patch("/providers/{provider}")
async def update_provider(
    provider: ProviderId, body: UpdateProvider, state: State
) -> WebProviderStatus:
    """Turn a service on or off, or move its daily budget. Omitted fields are left alone."""
    await state.web.store.update(
        provider, enabled=body.enabled, daily_budget=body.daily_request_budget
    )
    return WebProviderStatus.model_validate(await state.web.provider_status(provider))


@router.delete("/providers/{provider}/key", status_code=204)
async def remove_key(provider: ProviderId, state: State) -> Response:
    """Delete a service's key and its usage count. Saved runs and their evidence stay."""
    await state.web.store.remove(provider)
    return Response(status_code=204)


class CheckRequest(Out):
    acknowledge_cost: bool = Field(
        default=False,
        description="The user has seen what the test may cost. Without it, a test that may "
        "spend a credit is refused with 409 confirmation_required.",
    )


class CheckWorks(Out):
    ok: Literal[True]
    provider: ProviderId
    detail: str
    reported: ReportedUsageInfo | None


class CheckFailed(Out):
    ok: Literal[False]
    provider: ProviderId
    #: Stable: ``auth_failed``, ``quota_exceeded``, ``rate_limited``, ``timeout``,
    #: ``unavailable``, ``network``, ``bad_response`` or ``rejected``.
    code: str
    message: str
    retry_after: int | None


@router.post("/providers/{provider}/check")
async def check(provider: ProviderId, body: CheckRequest, state: State) -> CheckWorks | CheckFailed:
    """Test a stored key against the service. Counts as a request, and may cost a credit."""
    outcome = await state.web.check(provider, acknowledged=body.acknowledge_cost)
    if outcome.ok:
        return CheckWorks(
            ok=True,
            provider=provider,
            detail=outcome.detail or "",
            reported=(
                ReportedUsageInfo.model_validate(outcome.reported.to_dict())
                if outcome.reported
                else None
            ),
        )
    return CheckFailed(
        ok=False,
        provider=provider,
        code=outcome.code or "failed",
        message=outcome.message or "The check failed.",
        retry_after=round(outcome.retry_after) if outcome.retry_after is not None else None,
    )


class PreferredBody(Out):
    provider: ProviderId | None = Field(
        description="The service to prefer, or null to use the built-in order."
    )


@router.put("/preferred")
async def set_preferred(body: PreferredBody, state: State) -> WebResearchStatus:
    """Choose which service searches use when more than one has an enabled key."""
    await state.web.store.set_preferred(body.provider)
    return WebResearchStatus.model_validate(await state.web.status())


class ExtractRequest(Out):
    urls: list[str] = Field(min_length=1, max_length=20, description="Public http(s) pages")
    depth: Literal["basic", "advanced"] = "basic"


class ExtractedPageInfo(Out):
    url: str
    content: str
    truncated: bool


class FailedExtraction(Out):
    url: str
    reason: str


class ExtractResponse(Out):
    provider: ProviderId
    pages: list[ExtractedPageInfo]
    failed: list[FailedExtraction]


@router.post("/extract")
async def extract(body: ExtractRequest, state: State) -> ExtractResponse:
    """Page text for up to 20 public URLs, fetched by the service. Spends credits there."""
    provider, result = await state.web.extract(body.urls, depth=body.depth)
    return ExtractResponse.model_validate(
        {
            "provider": provider,
            "pages": [
                {"url": p.url, "content": p.content, "truncated": p.truncated} for p in result.pages
            ],
            "failed": [{"url": url, "reason": why} for url, why in result.failed],
        }
    )
