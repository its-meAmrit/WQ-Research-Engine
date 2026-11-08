"""Brave Search: web search, through its official API.

https://api-dashboard.search.brave.com. ``GET /res/v1/web/search`` with the key in the
``X-Subscription-Token`` header. Brave has no endpoint that only validates a key, so the
connectivity test sends one real search; Brave bills successful requests, so the UI says so
first. Its answers carry ``X-RateLimit-*`` headers, which is where reported usage comes from.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from .base import (
    CheckCost,
    CheckReport,
    ExtractResult,
    ProviderSpec,
    ReportedUsage,
    read_json,
    refusal,
    send,
)
from .errors import UnsupportedCapabilityError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx2

URL = "https://api.search.brave.com/res/v1/web/search"
LABEL = "Brave Search"

SPEC = ProviderSpec(
    id="brave",
    label=LABEL,
    description=(
        "Independent web search index. Returns result titles, snippets and ages; it does not "
        "extract page content. Plans differ in monthly allowance and billing."
    ),
    capabilities=("search",),
    key_url="https://api-dashboard.search.brave.com/app/keys",
    docs_url="https://api-dashboard.search.brave.com/documentation",
    check_cost=CheckCost(
        may_consume_credit=True,
        description=(
            "Sends one real search, because Brave has no free key check. Brave counts "
            "successful requests against your plan, so it uses one request there and here."
        ),
    ),
)


def _windows(value: str | None) -> list[int]:
    """``"1, 15000"`` as ``[1, 15000]``; anything unreadable as an empty list."""
    try:
        return [int(part.strip()) for part in (value or "").split(",") if part.strip()]
    except ValueError:
        return []


def reported_from(headers: Mapping[str, str]) -> ReportedUsage | None:
    """Usage from Brave's rate-limit headers, using the longest window each lists.

    ``X-RateLimit-Limit`` and ``-Remaining`` list one number per window, shortest first
    (``1, 15000``: per second, then per plan period). Only when both agree on a longest
    window is anything reported, so a header this does not understand reports nothing rather
    than a wrong figure.
    """
    limits = _windows(headers.get("x-ratelimit-limit"))
    remaining = _windows(headers.get("x-ratelimit-remaining"))
    if not limits or len(limits) != len(remaining) or len(limits) < 2:
        return None
    limit, left = limits[-1], remaining[-1]
    if limit <= 0 or not 0 <= left <= limit:
        return None
    return ReportedUsage(
        checked_at=datetime.now(UTC).isoformat(),
        used=float(limit - left),
        limit=float(limit),
        remaining=float(left),
        period="plan window, from Brave's rate-limit headers",
    )


@dataclass(slots=True)
class BraveSearch:
    #: Out of ``repr``: a dataclass would otherwise print it into any log line or traceback.
    key: str = field(repr=False)
    name: str = "brave"
    spec: ProviderSpec = SPEC

    async def _get(
        self, client: httpx2.AsyncClient, params: dict[str, Any]
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        response = await send(
            client,
            "GET",
            URL,
            label=LABEL,
            params=params,
            headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
        )
        if response.status_code != 200:
            raise refusal(LABEL, response)
        return read_json(LABEL, response), response.headers

    async def search(
        self, client: httpx2.AsyncClient, query: str, *, count: int, freshness_days: int
    ) -> list[dict[str, Any]]:
        freshness = (
            "pd"
            if freshness_days <= 1
            else "pw"
            if freshness_days <= 7
            else "pm"
            if freshness_days <= 31
            else "py"
        )
        body, _ = await self._get(client, {"q": query, "count": count, "freshness": freshness})
        found = (body.get("web") or {}).get("results") or []
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("description", ""),
                "published": r.get("page_age") or r.get("age"),
            }
            for r in found
            if isinstance(r, dict)
        ]

    async def extract(
        self,
        client: httpx2.AsyncClient,  # noqa: ARG002 - the signature every service shares
        urls: list[str],  # noqa: ARG002
        *,
        depth: Literal["basic", "advanced"],  # noqa: ARG002
    ) -> ExtractResult:
        raise UnsupportedCapabilityError(f"{LABEL} searches the web but does not extract pages.")

    async def check(self, client: httpx2.AsyncClient) -> CheckReport:
        _, headers = await self._get(client, {"q": "connectivity check", "count": 1})
        return CheckReport(detail="Brave Search accepted the key.", reported=reported_from(headers))
