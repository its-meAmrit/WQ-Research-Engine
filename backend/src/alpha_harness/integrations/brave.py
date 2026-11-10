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
    DocumentQuery,
    ExtractResult,
    ProviderSpec,
    ReportedUsage,
    SearchHit,
    SearchPage,
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


def _window(start: str) -> str:
    """The Brave window that reaches back to ``start`` (``pd``, ``pw``, ``pm`` or ``py``)."""
    try:
        age = (datetime.now(UTC).date() - datetime.fromisoformat(start).date()).days
    except ValueError:
        return "py"
    return "pd" if age <= 1 else "pw" if age <= 7 else "pm" if age <= 31 else "py"


def _published(result: dict[str, Any]) -> str | None:
    """The day a result says it was published, as ``YYYY-MM-DD``, or ``None``.

    Brave's ``page_age`` is an ISO timestamp; its ``age`` is a phrase ("3 weeks ago") that is
    not a date and is not guessed at.
    """
    raw = result.get("page_age")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw).date().isoformat()
        except ValueError:
            return None
    return None


def _within(published: str | None, start: str | None, end: str | None) -> bool:
    """Whether a result with this date is inside the range. A result with no date is kept:
    dropping it would hide a page for having no metadata."""
    if published is None:
        return True
    return (not start or published >= start) and (not end or published <= end)


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

    max_results_cap: int = 10

    async def search_documents(
        self, client: httpx2.AsyncClient, request: DocumentQuery
    ) -> SearchPage:
        """Documents for a query. Brave has no date range or domain list that this adapter
        relies on, so both are approximated and the page says how: the date range becomes the
        nearest of Brave's day/week/month/year windows, ``site:`` limits the domains, and the
        results are then trimmed to the range by their reported age where one is given."""
        notes: list[str] = []
        query = request.query
        if request.include_domains:
            query += " (" + " OR ".join(f"site:{d}" for d in request.include_domains) + ")"
            notes.append("Domains were limited with site: operators in the query.")
        for domain in request.exclude_domains:
            query += f" -site:{domain}"
        params: dict[str, Any] = {"q": query, "count": min(max(request.max_results, 1), 10)}
        if request.start_date:
            params["freshness"] = _window(request.start_date)
            notes.append(
                "Brave has no exact date range here: the nearest window was used and results "
                "outside the dates, where Brave gives an age, were dropped."
            )
        body, _ = await self._get(client, params)
        found = (body.get("web") or {}).get("results") or []
        hits = [
            SearchHit(
                url=str(r.get("url") or ""),
                title=str(r.get("title") or ""),
                snippet=str(r.get("description") or "")[:600],
                published=_published(r),
            )
            for r in found
            if isinstance(r, dict) and r.get("url")
        ]
        if request.start_date or request.end_date:
            hits = [h for h in hits if _within(h.published, request.start_date, request.end_date)]
        return SearchPage(hits=hits, credits=1.0, notes=notes)

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
