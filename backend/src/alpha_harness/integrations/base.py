"""The contract every web research service meets, and what they share.

A service is a :class:`ProviderSpec` (what it is and can do, shown in the UI) plus an adapter
that talks to its official API. Adapters are small and hold the key only for the life of one
call; nothing here reads or stores credentials.
"""

import ipaddress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx2

from .errors import (
    IntegrationAuthError,
    IntegrationBadResponse,
    IntegrationError,
    IntegrationNetworkError,
    IntegrationQuotaError,
    IntegrationRateLimited,
    IntegrationRejected,
    IntegrationTimeout,
    IntegrationUnavailable,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The services that exist. Mirrors the registry's keys (a test keeps the two in step); it is a
#: ``Literal`` so the API validates a path or body before any code runs.
ProviderId = Literal["tavily", "brave"]
Capability = Literal["search", "extract"]

#: Where the key a search used came from. ``caller`` is a test or a caller that supplied its own
#: provider object; it is never tracked against a budget.
Source = Literal["store", "environment", "caller"]

SETTINGS_PATH = "API Integration › Web Research"


@dataclass(frozen=True, slots=True)
class CheckCost:
    """What a connectivity test does to the user's quota, said before it is sent."""

    may_consume_credit: bool
    description: str


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    id: ProviderId
    label: str
    description: str
    capabilities: tuple[Capability, ...]
    #: Where to get a key.
    key_url: str
    docs_url: str
    check_cost: CheckCost


@dataclass(frozen=True, slots=True)
class ReportedUsage:
    """What the provider itself reported, as of ``checked_at``. Not our count."""

    checked_at: str
    used: float | None = None
    limit: float | None = None
    remaining: float | None = None
    plan: str | None = None
    #: What the figures cover, in the provider's terms ("billing cycle").
    period: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkedAt": self.checked_at,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "plan": self.plan,
            "period": self.period,
        }


@dataclass(slots=True)
class CheckReport:
    """A successful connectivity test."""

    detail: str
    reported: ReportedUsage | None = None


@dataclass(slots=True)
class ExtractedPage:
    url: str
    content: str
    truncated: bool = False


@dataclass(slots=True)
class ExtractResult:
    pages: list[ExtractedPage] = field(default_factory=list)
    #: ``(url, reason)`` for each URL the provider could not read. The reason is the provider's
    #: own and is shown, never trusted.
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class DocumentQuery:
    """One search for documents to read, with the filters a provider can honour.

    A provider ignores what it cannot do and says so in :attr:`SearchPage.notes`; it never
    pretends a filter was applied.
    """

    query: str
    max_results: int = 10
    depth: Literal["basic", "advanced"] = "basic"
    #: ``YYYY-MM-DD``. Inclusive bounds on publication date.
    start_date: str | None = None
    end_date: str | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()


@dataclass(slots=True)
class SearchHit:
    url: str
    title: str
    #: The provider's short excerpt. Never the page.
    snippet: str
    published: str | None = None
    score: float | None = None


@dataclass(slots=True)
class SearchPage:
    hits: list[SearchHit] = field(default_factory=list)
    #: What the provider said the call cost, when it said. ``None`` means it did not.
    credits: float | None = None
    #: What was applied and what was not, in words for the person.
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class DocumentSearch(Protocol):
    """A service that can search for documents with date and domain filters."""

    #: The most results one call can return.
    max_results_cap: int

    async def search_documents(
        self, client: httpx2.AsyncClient, request: DocumentQuery
    ) -> SearchPage: ...


class SearchProvider(Protocol):
    """What the SPC research step needs: raw results for a query."""

    name: str

    async def search(
        self, client: httpx2.AsyncClient, query: str, *, count: int, freshness_days: int
    ) -> list[dict[str, Any]]:
        """Raw results as ``{url, title, snippet, published}`` dicts."""
        ...


@runtime_checkable
class WebProvider(SearchProvider, Protocol):
    """A full adapter: search, an optional extract, and a connectivity test."""

    spec: ProviderSpec

    async def extract(
        self, client: httpx2.AsyncClient, urls: list[str], *, depth: Literal["basic", "advanced"]
    ) -> ExtractResult: ...

    async def check(self, client: httpx2.AsyncClient) -> CheckReport: ...


# --- URLs -------------------------------------------------------------------------------


def safe_public_url(url: str) -> str | None:
    """The URL if it is a plain http(s) link to a public host, else ``None``.

    Refuses other schemes, embedded credentials, ``localhost``-style names, names without a dot
    and any literal address that is private, loopback, link-local, reserved or multicast.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
        return None
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan")):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url.strip() if "." in host else None
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return None
    return url.strip()


# --- talking to a service ---------------------------------------------------------------


async def send(
    client: httpx2.AsyncClient, method: str, url: str, *, label: str, **kwargs: Any
) -> httpx2.Response:
    """One request, with transport failures turned into messages that carry no request.

    The chain is cut (``from None``): an exception that remembered the request would remember
    the header the key travels in.
    """
    try:
        return await client.request(method, url, **kwargs)
    except httpx2.TimeoutException:
        raise IntegrationTimeout(f"{label} did not answer in time. Try again shortly.") from None
    except httpx2.TransportError as exc:
        raise IntegrationNetworkError(
            f"{label} could not be reached ({type(exc).__name__}). Check your connection."
        ) from None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait, from ``Retry-After`` or the first window of ``X-RateLimit-Reset``."""
    for name in ("retry-after", "x-ratelimit-reset"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            value = float(raw.split(",")[0].strip())
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def refusal(label: str, response: httpx2.Response) -> IntegrationError:
    """The typed error for a non-200 answer. Built from the status only, never the body or the
    request: the body is the provider's to word and the request holds the key."""
    status = response.status_code
    if status in {401, 403}:
        return IntegrationAuthError(
            f"{label} refused the API key (HTTP {status}). Check it, or replace it under "
            f"{SETTINGS_PATH}. A key supplied by the legacy environment variable is fixed there."
        )
    if status in {402, 432, 433}:
        return IntegrationQuotaError(
            f"{label} says the plan or usage limit is used up (HTTP {status}). Check your plan "
            f"in the {label} dashboard; waiting a few minutes will not clear it."
        )
    if status == 429:
        wait = _retry_after(response.headers)
        when = f" Try again in about {wait:.0f} seconds." if wait is not None else " Try shortly."
        return IntegrationRateLimited(
            f"{label} is rate limiting requests (HTTP 429).{when}", retry_after=wait
        )
    if status in {408, 504}:
        return IntegrationTimeout(f"{label} timed out on its side (HTTP {status}). Try again.")
    if status >= 500:
        return IntegrationUnavailable(
            f"{label} is having trouble (HTTP {status}). This is on their side; try again later."
        )
    if status in {400, 422}:
        return IntegrationRejected(f"{label} rejected the request (HTTP {status}).")
    return IntegrationBadResponse(f"{label} answered HTTP {status}, which was not expected.")


def read_json(label: str, response: httpx2.Response) -> dict[str, Any]:
    """The body as a JSON object, or a typed error that says it could not be read."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise IntegrationBadResponse(f"{label} sent an answer that could not be read.")
    return body
