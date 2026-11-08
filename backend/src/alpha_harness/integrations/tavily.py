"""Tavily: search and content extraction, through its official API.

https://docs.tavily.com. Search is ``POST /search`` and extraction ``POST /extract``; both take
the key as a bearer token. The connectivity test calls ``GET /usage``, which also reports the
key's usage; Tavily does not document whether that call costs a credit, so the UI treats it as
one that might and asks first.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from .base import (
    CheckCost,
    CheckReport,
    ExtractedPage,
    ExtractResult,
    ProviderSpec,
    ReportedUsage,
    read_json,
    refusal,
    safe_public_url,
    send,
)
from .errors import IntegrationRejected

if TYPE_CHECKING:
    import httpx2

API = "https://api.tavily.com"
LABEL = "Tavily"
#: Tavily accepts up to 20 URLs a call; fewer keeps one call's cost easy to predict.
MAX_EXTRACT_URLS = 20
#: A page can be enormous; this is more than a model will read.
MAX_PAGE_CHARS = 100_000

SPEC = ProviderSpec(
    id="tavily",
    label=LABEL,
    description=(
        "Search built for AI workflows, with page content extraction. Has a free monthly "
        "allowance; beyond it requests are billed to your account."
    ),
    capabilities=("search", "extract"),
    key_url="https://app.tavily.com/home",
    docs_url="https://docs.tavily.com/documentation/api-reference/introduction",
    check_cost=CheckCost(
        may_consume_credit=True,
        description=(
            "Asks Tavily for this key's usage. Tavily does not say whether that call uses a "
            "credit, so it is counted as one request against your budget here and may cost one "
            "credit there."
        ),
    ),
)


def _section(body: dict[str, Any], name: str) -> dict[str, Any]:
    found = body.get(name)
    return found if isinstance(found, dict) else {}


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


@dataclass(slots=True)
class TavilySearch:
    #: Out of ``repr``: a dataclass would otherwise print it into any log line or traceback.
    key: str = field(repr=False)
    name: str = "tavily"
    spec: ProviderSpec = SPEC

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    async def search(
        self, client: httpx2.AsyncClient, query: str, *, count: int, freshness_days: int
    ) -> list[dict[str, Any]]:
        response = await send(
            client,
            "POST",
            f"{API}/search",
            label=LABEL,
            json={
                "query": query,
                "max_results": count,
                "topic": "news" if freshness_days <= 31 else "general",
                "days": freshness_days,
                "include_answer": False,
            },
            headers=self._auth(),
        )
        if response.status_code != 200:
            raise refusal(LABEL, response)
        found = read_json(LABEL, response).get("results") or []
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "published": r.get("published_date"),
            }
            for r in found
            if isinstance(r, dict)
        ]

    async def extract(
        self, client: httpx2.AsyncClient, urls: list[str], *, depth: Literal["basic", "advanced"]
    ) -> ExtractResult:
        """Page text for up to :data:`MAX_EXTRACT_URLS` public URLs, fetched by Tavily.

        Tavily does the fetching, so this machine never requests an arbitrary URL; the ones
        sent are still screened, because a link to a private address has no business in a
        request that spends credits.
        """
        if not 1 <= len(urls) <= MAX_EXTRACT_URLS:
            raise IntegrationRejected(f"Give between 1 and {MAX_EXTRACT_URLS} URLs to extract.")
        safe = [u for u in (safe_public_url(u) for u in urls) if u]
        if len(safe) != len(urls):
            raise IntegrationRejected(
                "Only public http(s) pages can be extracted; "
                f"{len(urls) - len(safe)} of the URLs were not."
            )
        response = await send(
            client,
            "POST",
            f"{API}/extract",
            label=LABEL,
            json={"urls": safe, "extract_depth": depth, "format": "markdown"},
            headers=self._auth(),
        )
        if response.status_code != 200:
            raise refusal(LABEL, response)
        body = read_json(LABEL, response)
        result = ExtractResult()
        for item in body.get("results") or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("raw_content") or "")
            result.pages.append(
                ExtractedPage(
                    url=str(item.get("url") or ""),
                    content=content[:MAX_PAGE_CHARS],
                    truncated=len(content) > MAX_PAGE_CHARS,
                )
            )
        for item in body.get("failed_results") or []:
            if isinstance(item, dict):
                result.failed.append(
                    (str(item.get("url") or ""), str(item.get("error") or "unknown")[:200])
                )
        return result

    async def check(self, client: httpx2.AsyncClient) -> CheckReport:
        response = await send(client, "GET", f"{API}/usage", label=LABEL, headers=self._auth())
        if response.status_code != 200:
            raise refusal(LABEL, response)
        body = read_json(LABEL, response)
        account, key = _section(body, "account"), _section(body, "key")
        used = _number(account.get("plan_usage"))
        limit = _number(account.get("plan_limit"))
        period = "plan, current billing cycle"
        if used is None and limit is None:
            used, limit, period = _number(key.get("usage")), _number(key.get("limit")), "this key"
        plan = account.get("current_plan")
        return CheckReport(
            detail="Tavily accepted the key.",
            reported=ReportedUsage(
                checked_at=datetime.now(UTC).isoformat(),
                used=used,
                limit=limit,
                remaining=None if used is None or limit is None else max(0.0, limit - used),
                plan=plan if isinstance(plan, str) else None,
                period=period,
            ),
        )
