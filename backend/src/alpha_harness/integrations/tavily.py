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
    DocumentQuery,
    ExtractedPage,
    ExtractResult,
    ProviderSpec,
    ReportedUsage,
    SearchHit,
    SearchPage,
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


#: Tavily's documented prices (https://docs.tavily.com/guides/api-credits), used only to show an
#: estimate before a call. A search is 1 credit, or 2 at ``advanced`` depth; extraction is 1
#: credit (2 advanced) per 5 URLs read successfully, and a failed URL is not charged.
SEARCH_CREDITS = {"basic": 1, "advanced": 2}
EXTRACT_CREDITS_PER_FIVE = {"basic": 1, "advanced": 2}
MAX_RESULTS = 20


def search_credits(depth: Literal["basic", "advanced"]) -> int:
    return SEARCH_CREDITS[depth]


def extract_credits(urls: int, depth: Literal["basic", "advanced"]) -> int:
    """The most an extraction of ``urls`` pages can cost: every one succeeds."""
    return -(-urls // 5) * EXTRACT_CREDITS_PER_FIVE[depth] if urls > 0 else 0


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

    max_results_cap: int = MAX_RESULTS

    async def search_documents(
        self, client: httpx2.AsyncClient, request: DocumentQuery
    ) -> SearchPage:
        """Documents for a query, with the filters Tavily documents: ``start_date`` and
        ``end_date``, ``include_domains`` and ``exclude_domains``, and ``search_depth``.

        The topic is ``general``: Tavily's ``news`` topic favours recent articles, which is the
        wrong shape for research. Publication dates are asked for and kept as Tavily reports
        them; Tavily calls its figure a best estimate, so a missing one is left missing.
        """
        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": min(max(request.max_results, 1), MAX_RESULTS),
            "search_depth": request.depth,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
            "include_published_date": True,
            "include_usage": True,
        }
        if request.start_date:
            payload["start_date"] = request.start_date
        if request.end_date:
            payload["end_date"] = request.end_date
        if request.include_domains:
            payload["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["exclude_domains"] = list(request.exclude_domains)
        response = await send(
            client, "POST", f"{API}/search", label=LABEL, json=payload, headers=self._auth()
        )
        if response.status_code != 200:
            raise refusal(LABEL, response)
        body = read_json(LABEL, response)
        hits = [
            SearchHit(
                url=str(r.get("url") or ""),
                title=str(r.get("title") or ""),
                snippet=str(r.get("content") or "")[:600],
                published=r.get("published_date")
                if isinstance(r.get("published_date"), str)
                else None,
                score=_number(r.get("score")),
            )
            for r in body.get("results") or []
            if isinstance(r, dict) and r.get("url")
        ]
        usage = _section(body, "usage")
        return SearchPage(
            hits=hits,
            credits=_number(usage.get("credits")),
            notes=(
                ["Publication dates are Tavily's own estimates and are missing for some results."]
                if request.start_date or request.end_date
                else []
            ),
        )

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
