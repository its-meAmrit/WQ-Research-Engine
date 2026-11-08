"""Web research services: sealed keys, precedence, budget, and the two adapters.

Every request goes to an ``httpx2.MockTransport``: nothing here reaches Tavily or Brave, so no
test spends a credit.
"""

import json
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_args

import httpx2
import pytest
from sqlalchemy import select

from alpha_harness.db.models import IntegrationCredential, IntegrationUsage, Preference
from alpha_harness.db.sqlite import Database
from alpha_harness.integrations import legacy, registry
from alpha_harness.integrations.base import ProviderId, safe_public_url
from alpha_harness.integrations.brave import BraveSearch, reported_from
from alpha_harness.integrations.credentials import (
    DEFAULT_DAILY_BUDGET,
    clean_key,
    fingerprint,
    hint,
    seal_context,
)
from alpha_harness.integrations.errors import (
    ConfirmationRequiredError,
    DuplicateKeyError,
    IntegrationRejected,
    InvalidKeyError,
    NotConfiguredError,
    RequestBudgetExhaustedError,
    UnknownProviderError,
    UnreadableKeyError,
    UnsupportedCapabilityError,
)
from alpha_harness.integrations.service import WebResearchService
from alpha_harness.integrations.tavily import TavilySearch
from alpha_harness.sealing import Sealer, SealError

if TYPE_CHECKING:
    from pathlib import Path

TAVILY_KEY = "tvly-dev-SECRET-0123456789abcdef"
BRAVE_KEY = "BSA-SECRET-ABCDEFGHIJKLMNOP"
ENV_KEY = "tvly-ENVIRONMENT-KEY-9876543210"
HOSTS = {"tavily": "api.tavily.com", "brave": "api.search.brave.com"}


class Server:
    """A stand-in for the providers that records what it was sent."""

    def __init__(self, handler: Callable[[httpx2.Request], httpx2.Response] | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler = handler or self.ok

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @staticmethod
    def ok(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == HOSTS["brave"]:
            return httpx2.Response(
                200,
                json={"web": {"results": []}},
                headers={"X-RateLimit-Limit": "1, 15000", "X-RateLimit-Remaining": "1, 14000"},
            )
        if request.url.path == "/usage":
            return httpx2.Response(
                200,
                json={
                    "key": {"usage": 5, "limit": None},
                    "account": {
                        "current_plan": "Researcher",
                        "plan_usage": 120,
                        "plan_limit": 1000,
                    },
                },
            )
        return httpx2.Response(200, json={"results": []})


def respond(status: int, **kwargs: Any) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda _r: httpx2.Response(status, **kwargs)


@pytest.fixture
def sealer(tmp_path: Path) -> Sealer:
    return Sealer(tmp_path / "key")


def build(
    db: Database,
    sealer: Sealer,
    handler: Callable[[httpx2.Request], httpx2.Response] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[WebResearchService, Server]:
    server = Server(handler)
    service = WebResearchService(
        db, sealer, env={} if env is None else env, transport=httpx2.MockTransport(server)
    )
    return service, server


# --- registry ---------------------------------------------------------------------------


def test_the_provider_literal_and_the_registry_agree() -> None:
    assert set(registry.REGISTRY) == set(get_args(ProviderId))


def test_tavily_supports_search_and_extraction_and_brave_search_only() -> None:
    assert set(registry.spec_for("tavily").capabilities) == {"search", "extract"}
    assert registry.spec_for("brave").capabilities == ("search",)


def test_unknown_providers_are_refused_by_name() -> None:
    with pytest.raises(UnknownProviderError, match="tavily, brave"):
        registry.spec_for("bing")
    with pytest.raises(UnknownProviderError):
        registry.build("../etc/passwd", "k")


# --- key hygiene ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "short",
        "has a space in the middle 123",
        "line\nbreak-1234567",
        "tvly-ünïcode-key-123456",
        "Bearer tvly-abcdefghijkl",
        "https://api.tavily.com/?key=abcdefghi",
        "x" * 300,
    ],
)
def test_malformed_keys_are_refused(raw: str) -> None:
    with pytest.raises(InvalidKeyError):
        clean_key(raw)


def test_a_pasted_key_is_trimmed() -> None:
    assert clean_key(f"  {TAVILY_KEY}\n") == TAVILY_KEY


def test_the_hint_reveals_little_and_the_fingerprint_none() -> None:
    assert TAVILY_KEY not in hint(TAVILY_KEY)
    assert len(hint(TAVILY_KEY)) < len(TAVILY_KEY) - 12
    assert hint("abc-defg-hij") == "…hij"  # a short key shows only its last characters
    assert TAVILY_KEY not in fingerprint(TAVILY_KEY)
    assert fingerprint(TAVILY_KEY) == fingerprint(TAVILY_KEY)
    assert fingerprint(TAVILY_KEY) != fingerprint(BRAVE_KEY)


# --- storage ----------------------------------------------------------------------------


async def test_a_key_is_encrypted_at_rest_and_survives_a_restart(
    tmp_path: Path, sealer: Sealer
) -> None:
    path = tmp_path / "restart.db"
    first = Database(path)
    await first.create_all()
    service, _ = build(first, sealer)
    await service.store.put("tavily", TAVILY_KEY, 40)
    await first.dispose()

    raw = path.read_bytes()
    assert TAVILY_KEY.encode() not in raw
    assert b"SECRET" not in raw
    with sqlite3.connect(path) as conn:
        sealed, shown = conn.execute(
            "SELECT key_sealed, hint FROM integration_credential"
        ).fetchone()
    assert TAVILY_KEY.encode() not in sealed
    assert shown == hint(TAVILY_KEY)

    # A new process: a new engine, and a sealer that reads the same key file.
    second = Database(path)
    await second.create_all()
    again, _ = build(second, Sealer(tmp_path / "key"))
    assert await again.store.secret("tavily") == TAVILY_KEY
    row = await again.store.require("tavily")
    assert (row.enabled, row.daily_request_budget) == (True, 40)
    await second.dispose()


async def test_a_key_sealed_for_one_purpose_will_not_open_for_another(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    row, _ = await service.store.put("tavily", TAVILY_KEY)
    with pytest.raises(SealError):
        sealer.open(row.key_sealed, context=seal_context("brave"))
    with pytest.raises(SealError):
        sealer.open(row.key_sealed, context="google-api-key")
    assert sealer.open(row.key_sealed, context=seal_context("tavily")) == TAVILY_KEY


async def test_a_row_moved_to_another_provider_cannot_be_read(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    async with db.session() as session:
        row = await session.scalar(select(IntegrationCredential))
        assert row is not None
        row.provider = "brave"
    with pytest.raises(UnreadableKeyError, match="could not be read"):
        await service.store.secret("brave")


async def test_the_same_key_cannot_be_stored_twice_or_under_two_services(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    with pytest.raises(DuplicateKeyError, match="already stored for Tavily"):
        await service.store.put("tavily", TAVILY_KEY)
    with pytest.raises(DuplicateKeyError, match="Tavily key"):
        await service.store.put("brave", TAVILY_KEY)
    assert len(await service.store.all_rows()) == 1


async def test_replacing_a_key_keeps_the_choices_and_drops_the_old_keys_history(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY, 30)
    await service.store.update("tavily", enabled=False)
    await service.store.mark(
        "tavily", ok=False, code="auth_failed", message="refused", checked=True
    )

    row, replaced = await service.store.put("tavily", "tvly-dev-REPLACEMENT-0123456789")
    assert replaced is True
    assert (row.enabled, row.daily_request_budget) == (False, 30)
    assert (row.last_error, row.last_checked_at, row.last_ok_at) == (None, None, None)
    assert await service.store.secret("tavily") == "tvly-dev-REPLACEMENT-0123456789"
    assert len(await service.store.all_rows()) == 1


async def test_a_new_key_starts_with_a_budget_and_a_bad_budget_is_refused(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    row, _ = await service.store.put("brave", BRAVE_KEY)
    assert row.daily_request_budget == DEFAULT_DAILY_BUDGET
    for bad in (0, -5, 1_000_000):
        with pytest.raises(InvalidKeyError, match="between 1"):
            await service.store.update("brave", daily_budget=bad)


async def test_invalid_provider_ids_never_reach_the_database(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    for bad in ("bing", "", "TAVILY", "tavily; DROP TABLE api_key"):
        with pytest.raises(UnknownProviderError):
            await service.store.put(bad, TAVILY_KEY)
    assert await service.store.all_rows() == []


async def test_removing_a_key_removes_its_usage_and_its_default_status(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.put("brave", BRAVE_KEY)
    await service.store.set_preferred("tavily")
    await service.store.reserve("tavily", 3)

    await service.store.remove("tavily")

    assert await service.store.preferred() is None
    assert [r.provider for r in await service.store.all_rows()] == ["brave"]
    async with db.session() as session:
        assert (await session.scalars(select(IntegrationUsage))).all() == []
    with pytest.raises(NotConfiguredError):
        await service.store.remove("tavily")


async def test_removing_the_other_services_key_leaves_the_default_alone(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.put("brave", BRAVE_KEY)
    await service.store.set_preferred("tavily")
    await service.store.remove("brave")
    assert await service.store.preferred() == "tavily"


# --- precedence -------------------------------------------------------------------------


async def test_nothing_configured_says_how_to_set_it_up(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    assert await service.resolve_search() is None
    active = await service.active()
    assert active["configured"] is False
    assert "API Integration › Web Research" in active["help"]
    assert "ALPHA_HARNESS_SPC_SEARCH" in active["help"]


async def test_a_stored_key_is_used(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    found = await service.resolve_search()
    assert found is not None
    assert (found.source, found.provider_id, found.provider.name) == ("store", "tavily", "tavily")
    assert found.credential_id is not None
    assert (await service.active())["source"] == "store"


async def test_the_environment_is_the_fallback_and_is_never_saved(
    db: Database, sealer: Sealer
) -> None:
    env = {legacy.ENV_PROVIDER: "Tavily", legacy.ENV_KEY: ENV_KEY}
    service, _ = build(db, sealer, env=env)

    found = await service.resolve_search()
    assert found is not None
    assert (found.source, found.provider_id, found.credential_id) == ("environment", "tavily", None)
    assert isinstance(found.provider, TavilySearch)
    assert found.provider.key == ENV_KEY

    active = await service.active()
    assert active["source"] == "environment"
    assert "environment" in active["message"]
    assert ENV_KEY not in json.dumps(await service.status())
    # Looking at it, resolving it and describing it wrote nothing.
    assert await service.store.all_rows() == []
    with sqlite3.connect(db._engine.url.database) as conn:
        assert conn.execute("SELECT count(*) FROM integration_credential").fetchone() == (0,)
        assert ENV_KEY not in "\n".join(conn.iterdump())


async def test_a_stored_key_beats_the_environment(db: Database, sealer: Sealer) -> None:
    env = {legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY}
    service, _ = build(db, sealer, env=env)
    await service.store.put("tavily", TAVILY_KEY)
    found = await service.resolve_search()
    assert found is not None
    assert found.source == "store"
    assert isinstance(found.provider, TavilySearch)
    assert found.provider.key == TAVILY_KEY
    assert (await service.status())["legacy_environment"]["in_use"] is False


async def test_a_disabled_key_is_skipped_so_the_environment_applies(
    db: Database, sealer: Sealer
) -> None:
    env = {legacy.ENV_PROVIDER: "brave", legacy.ENV_KEY: ENV_KEY}
    service, _ = build(db, sealer, env=env)
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.update("tavily", enabled=False)
    found = await service.resolve_search()
    assert found is not None
    assert (found.source, found.provider_id) == ("environment", "brave")

    await service.store.update("tavily", enabled=True)
    found = await service.resolve_search()
    assert found is not None
    assert found.source == "store"


async def test_disabling_the_only_key_with_no_environment_is_not_configured(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("brave", BRAVE_KEY)
    await service.store.update("brave", enabled=False)
    assert await service.resolve_search() is None
    status = await service.status()
    assert status["active"]["configured"] is False
    assert status["providers"][1]["configured"] is True
    assert status["providers"][1]["enabled"] is False


async def test_a_broken_environment_is_explained_not_guessed_at(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer, env={legacy.ENV_PROVIDER: "bing", legacy.ENV_KEY: "k"})
    assert await service.resolve_search() is None
    active = await service.active()
    assert "'bing' is not a search provider" in active["help"]
    assert (await service.status())["legacy_environment"]["usable"] is False

    service, _ = build(db, sealer, env={legacy.ENV_PROVIDER: "brave"})
    assert "ALPHA_HARNESS_SPC_SEARCH_KEY is not set" in (await service.active())["help"]


async def test_the_default_provider_decides_between_two_keys(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    await service.store.put("brave", BRAVE_KEY)
    await service.store.put("tavily", TAVILY_KEY)

    first = await service.resolve_search()
    assert first is not None
    assert first.provider_id == "tavily"  # registry order, no default chosen

    await service.store.set_preferred("brave")
    chosen = await service.resolve_search()
    assert chosen is not None
    assert chosen.provider_id == "brave"
    assert (await service.status())["preferred_provider"] == "brave"

    await service.store.set_preferred(None)
    again = await service.resolve_search()
    assert again is not None
    assert again.provider_id == "tavily"


async def test_a_default_must_have_a_key_and_falls_back_with_a_reason_when_disabled(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    with pytest.raises(NotConfiguredError):
        await service.store.set_preferred("brave")
    with pytest.raises(UnknownProviderError):
        await service.store.set_preferred("bing")

    await service.store.put("brave", BRAVE_KEY)
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.set_preferred("brave")
    await service.store.update("brave", enabled=False)
    active = await service.active()
    assert active["provider"] == "tavily"
    assert "Brave Search, has no enabled key" in active["message"]


async def test_the_default_is_stored_as_a_preference_row_and_ignored_if_unknown(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    async with db.session() as session:
        session.add(Preference(key="web_research_provider", value="bing"))
    assert await service.store.preferred() is None


async def test_an_unreadable_stored_key_is_marked_and_the_next_one_is_used(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.put("brave", BRAVE_KEY)
    async with db.session() as session:
        row = await session.scalar(
            select(IntegrationCredential).where(IntegrationCredential.provider == "tavily")
        )
        assert row is not None
        row.key_sealed = b"\x00" * 40
    found = await service.resolve_search()
    assert found is not None
    assert found.provider_id == "brave"
    broken = (await service.status())["providers"][0]
    assert broken["last_error_code"] == "key_unreadable"
    assert "Add the key again" in broken["last_error"]


# --- secrecy ----------------------------------------------------------------------------


async def test_no_key_appears_in_anything_the_ui_receives(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer, env={legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY})
    await service.store.put("tavily", TAVILY_KEY)
    await service.store.put("brave", BRAVE_KEY)
    await service.check("tavily", acknowledged=True)
    dumped = json.dumps(await service.status(), ensure_ascii=False)
    for secret in (TAVILY_KEY, BRAVE_KEY, ENV_KEY):
        assert secret not in dumped
    assert hint(TAVILY_KEY) in dumped


def test_a_legacy_config_never_prints_its_key() -> None:
    config = legacy.read({legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY})
    assert config is not None
    assert ENV_KEY not in repr(config)
    assert ENV_KEY not in str(config)


def test_adapters_do_not_print_their_key() -> None:
    for adapter in (TavilySearch(TAVILY_KEY), BraveSearch(BRAVE_KEY)):
        assert "SECRET" not in repr(adapter)
        assert "SECRET" not in str(adapter)


# --- connectivity test ------------------------------------------------------------------


async def test_a_check_that_may_cost_a_credit_asks_first_and_sends_nothing(
    db: Database, sealer: Sealer
) -> None:
    service, server = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    with pytest.raises(ConfirmationRequiredError, match="credit"):
        await service.check("tavily", acknowledged=False)
    assert server.requests == []
    assert await service.store.usage_today() == {}
    assert registry.spec_for("tavily").check_cost.may_consume_credit
    assert registry.spec_for("brave").check_cost.may_consume_credit


async def test_a_check_of_a_missing_key_says_to_add_one(db: Database, sealer: Sealer) -> None:
    service, server = build(db, sealer)
    with pytest.raises(NotConfiguredError, match="Add one under API Integration › Web Research"):
        await service.check("brave", acknowledged=True)
    assert server.requests == []


async def test_a_successful_tavily_check_records_when_and_what_tavily_reported(
    db: Database, sealer: Sealer
) -> None:
    service, server = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY)
    outcome = await service.check("tavily", acknowledged=True)

    assert outcome.ok
    assert outcome.reported is not None
    assert (outcome.reported.used, outcome.reported.limit, outcome.reported.remaining) == (
        120,
        1000,
        880,
    )
    assert outcome.reported.plan == "Researcher"
    (request,) = server.requests
    assert (request.method, request.url.path) == ("GET", "/usage")
    assert request.headers["authorization"] == f"Bearer {TAVILY_KEY}"
    # The key travels in a header, never in the URL.
    assert TAVILY_KEY not in str(request.url)

    status = (await service.status())["providers"][0]
    assert status["last_ok_at"] and status["last_checked_at"]
    assert status["last_error"] is None
    assert status["reported_usage"]["used"] == 120
    # The check is a request like any other against the user's budget, counted locally.
    assert status["usage"]["requests"] == 1


async def test_a_successful_brave_check_reports_from_the_rate_limit_headers(
    db: Database, sealer: Sealer
) -> None:
    service, server = build(db, sealer)
    await service.store.put("brave", BRAVE_KEY)
    outcome = await service.check("brave", acknowledged=True)
    assert outcome.ok
    assert outcome.reported is not None
    assert (outcome.reported.used, outcome.reported.limit) == (1000, 15000)
    (request,) = server.requests
    assert request.headers["x-subscription-token"] == BRAVE_KEY
    assert BRAVE_KEY not in str(request.url)


@pytest.mark.parametrize(
    ("status", "headers", "code", "needle"),
    [
        (401, {}, "auth_failed", "refused the API key"),
        (403, {}, "auth_failed", "refused the API key"),
        (429, {"Retry-After": "12"}, "rate_limited", "about 12 seconds"),
        (432, {}, "quota_exceeded", "limit is used up"),
        (433, {}, "quota_exceeded", "limit is used up"),
        (500, {}, "unavailable", "on their side"),
        (503, {}, "unavailable", "on their side"),
        (504, {}, "timeout", "timed out"),
        (422, {}, "rejected", "rejected the request"),
        (418, {}, "bad_response", "HTTP 418"),
    ],
)
async def test_each_failure_is_typed_actionable_and_free_of_the_key(
    db: Database, sealer: Sealer, status: int, headers: dict[str, str], code: str, needle: str
) -> None:
    service, _ = build(db, sealer, respond(status, headers=headers, text=f"echo {TAVILY_KEY}"))
    await service.store.put("tavily", TAVILY_KEY)
    outcome = await service.check("tavily", acknowledged=True)

    assert outcome.ok is False
    assert outcome.code == code
    assert needle in (outcome.message or "")
    assert TAVILY_KEY not in (outcome.message or "")  # not even a body that echoes it
    row = (await service.status())["providers"][0]
    assert row["last_error_code"] == code
    assert row["last_ok_at"] is None
    assert row["last_checked_at"] is not None
    assert TAVILY_KEY not in json.dumps(row)


async def test_a_rate_limit_carries_how_long_to_wait(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer, respond(429, headers={"Retry-After": "30"}))
    await service.store.put("tavily", TAVILY_KEY)
    outcome = await service.check("tavily", acknowledged=True)
    assert (outcome.code, outcome.retry_after) == ("rate_limited", 30)


async def test_transport_failures_are_timeouts_or_network_errors_without_the_request(
    db: Database, sealer: Sealer
) -> None:
    def timeout(_r: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=httpx2.Request("GET", "https://x"))

    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"refused {request.headers['authorization']}", request=request)

    service, _ = build(db, sealer, timeout)
    await service.store.put("tavily", TAVILY_KEY)
    slow = await service.check("tavily", acknowledged=True)
    assert (slow.code, slow.message) == (
        "timeout",
        "Tavily did not answer in time. Try again shortly.",
    )

    service, _ = build(db, sealer, down)
    gone = await service.check("tavily", acknowledged=True)
    assert gone.code == "network"
    assert "could not be reached" in (gone.message or "")
    assert TAVILY_KEY not in (gone.message or "")


async def test_an_answer_that_is_not_json_is_a_bad_response(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer, respond(200, text="<html>nope</html>"))
    await service.store.put("brave", BRAVE_KEY)
    outcome = await service.check("brave", acknowledged=True)
    assert (outcome.code, outcome.message) == (
        "bad_response",
        "Brave Search sent an answer that could not be read.",
    )


async def test_a_success_after_a_failure_clears_the_error(db: Database, sealer: Sealer) -> None:
    failing = [True]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401) if failing[0] else Server.ok(request)

    service, _ = build(db, sealer, handler)
    await service.store.put("tavily", TAVILY_KEY)
    assert (await service.check("tavily", acknowledged=True)).ok is False
    failing[0] = False
    assert (await service.check("tavily", acknowledged=True)).ok is True
    row = (await service.status())["providers"][0]
    assert row["last_error"] is None
    assert row["last_ok_at"] is not None


# --- budget -----------------------------------------------------------------------------


async def test_requests_are_counted_against_the_users_own_budget(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY, 5)
    assert await service.store.reserve("tavily", 2) == 2
    assert await service.store.reserve("tavily", 2) == 2

    row = (await service.status())["providers"][0]
    assert (row["usage"]["requests"], row["usage"]["remaining_today"]) == (4, 1)
    # Said to be what it is: a local count, not the provider's balance.
    assert "Not the provider's balance" in row["usage"]["note"]

    assert await service.store.reserve("tavily", 4) == 1  # what is left, not what was asked
    with pytest.raises(RequestBudgetExhaustedError, match="5 of 5") as spent:
        await service.store.reserve("tavily", 1)
    assert spent.value.retry_after is not None
    assert "00:00 UTC" in str(spent.value)


async def test_all_or_nothing_does_not_take_a_partial_slice(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY, 3)
    await service.store.reserve("tavily", 2)
    with pytest.raises(RequestBudgetExhaustedError):
        await service.store.reserve("tavily", 2, all_or_nothing=True)
    assert (await service.store.usage_today()) == {1: 2}


async def test_a_spent_budget_stops_a_check_before_anything_is_sent(
    db: Database, sealer: Sealer
) -> None:
    service, server = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY, 1)
    assert (await service.check("tavily", acknowledged=True)).ok
    with pytest.raises(RequestBudgetExhaustedError):
        await service.check("tavily", acknowledged=True)
    assert len(server.requests) == 1


async def test_raising_the_budget_lets_requests_through_again(db: Database, sealer: Sealer) -> None:
    service, _ = build(db, sealer)
    await service.store.put("brave", BRAVE_KEY, 1)
    await service.store.reserve("brave", 1)
    with pytest.raises(RequestBudgetExhaustedError):
        await service.store.reserve("brave", 1)
    await service.store.update("brave", daily_budget=10)
    assert await service.store.reserve("brave", 1) == 1


async def test_concurrent_callers_cannot_both_take_the_last_request(
    db: Database, sealer: Sealer
) -> None:
    import asyncio

    service, _ = build(db, sealer)
    await service.store.put("tavily", TAVILY_KEY, 5)
    results = await asyncio.gather(
        *(service.store.reserve("tavily", 1) for _ in range(8)), return_exceptions=True
    )
    assert sum(1 for r in results if r == 1) == 5
    assert sum(isinstance(r, RequestBudgetExhaustedError) for r in results) == 3


async def test_environment_keys_are_not_budgeted_and_the_status_says_so(
    db: Database, sealer: Sealer
) -> None:
    service, _ = build(db, sealer, env={legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY})
    found = await service.resolve_search()
    assert found is not None
    assert await service.reserve(found, 40) == 40
    assert "not counted against a budget" in (await service.active())["message"]


# --- Tavily and Brave adapters ----------------------------------------------------------


async def test_tavily_search_sends_the_documented_request(db: Database, sealer: Sealer) -> None:
    server = Server(
        respond(
            200,
            json={
                "results": [
                    {
                        "url": "https://a.example.com/x",
                        "title": "A",
                        "content": "body",
                        "published_date": "2026-11-01",
                    }
                ]
            },
        )
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        found = await TavilySearch(TAVILY_KEY).search(client, "q", count=6, freshness_days=30)
    (request,) = server.requests
    assert (request.method, str(request.url)) == ("POST", "https://api.tavily.com/search")
    assert request.headers["authorization"] == f"Bearer {TAVILY_KEY}"
    sent = json.loads(request.content)
    assert (sent["query"], sent["max_results"], sent["topic"]) == ("q", 6, "news")
    assert TAVILY_KEY not in request.content.decode()
    assert found == [
        {
            "url": "https://a.example.com/x",
            "title": "A",
            "snippet": "body",
            "published": "2026-11-01",
        }
    ]


async def test_tavily_extract_sends_urls_and_reads_pages_and_failures(
    db: Database, sealer: Sealer
) -> None:
    body = {
        "results": [{"url": "https://a.example.com/x", "raw_content": "page text"}],
        "failed_results": [{"url": "https://b.example.com/y", "error": "blocked"}],
    }
    server = Server(respond(200, json=body))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        result = await TavilySearch(TAVILY_KEY).extract(
            client, ["https://a.example.com/x", "https://b.example.com/y"], depth="advanced"
        )
    (request,) = server.requests
    assert (request.method, request.url.path) == ("POST", "/extract")
    sent = json.loads(request.content)
    assert sent["urls"] == ["https://a.example.com/x", "https://b.example.com/y"]
    assert (sent["extract_depth"], sent["format"]) == ("advanced", "markdown")
    assert [(p.url, p.content, p.truncated) for p in result.pages] == [
        ("https://a.example.com/x", "page text", False)
    ]
    assert result.failed == [("https://b.example.com/y", "blocked")]


async def test_extraction_refuses_private_addresses_and_bad_counts_before_sending() -> None:
    server = Server()
    adapter = TavilySearch(TAVILY_KEY)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        for urls in (["http://127.0.0.1/admin"], ["https://ok.example.com", "http://10.0.0.5/"]):
            with pytest.raises(IntegrationRejected, match="public http"):
                await adapter.extract(client, urls, depth="basic")
        with pytest.raises(IntegrationRejected, match="between 1 and 20"):
            await adapter.extract(client, [], depth="basic")
        with pytest.raises(IntegrationRejected, match="between 1 and 20"):
            await adapter.extract(
                client, [f"https://e{n}.example.com" for n in range(21)], depth="basic"
            )
    assert server.requests == []
    assert safe_public_url("https://ok.example.com") is not None


async def test_very_long_pages_are_cut_and_say_so() -> None:
    server = Server(
        respond(
            200, json={"results": [{"url": "https://a.example.com", "raw_content": "x" * 150_000}]}
        )
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        result = await TavilySearch(TAVILY_KEY).extract(
            client, ["https://a.example.com"], depth="basic"
        )
    assert (len(result.pages[0].content), result.pages[0].truncated) == (100_000, True)


async def test_the_service_extracts_through_tavily_and_counts_it(
    db: Database, sealer: Sealer
) -> None:
    server_body = {"results": [{"url": "https://a.example.com", "raw_content": "hi"}]}
    service, server = build(db, sealer, respond(200, json=server_body))
    await service.store.put("tavily", TAVILY_KEY, 10)
    provider, result = await service.extract(["https://a.example.com"])
    assert (provider, [p.content for p in result.pages]) == ("tavily", ["hi"])
    assert (await service.status())["providers"][0]["usage"]["requests"] == 1
    assert server.requests[0].url.host == HOSTS["tavily"]


async def test_extraction_needs_a_service_that_can_do_it(db: Database, sealer: Sealer) -> None:
    service, server = build(db, sealer)
    with pytest.raises(NotConfiguredError, match="Tavily can"):
        await service.extract(["https://a.example.com"])
    await service.store.put("brave", BRAVE_KEY)  # search only
    with pytest.raises(NotConfiguredError):
        await service.extract(["https://a.example.com"])
    assert server.requests == []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        with pytest.raises(UnsupportedCapabilityError, match="does not extract"):
            await BraveSearch(BRAVE_KEY).extract(client, ["https://a.example.com"], depth="basic")


def test_brave_usage_is_only_reported_when_the_headers_make_sense() -> None:
    assert (
        reported_from({"x-ratelimit-limit": "1, 15000", "x-ratelimit-remaining": "1, 14990"}).used
        == 10
    )  # pyright: ignore[reportOptionalMemberAccess]
    for headers in (
        {},
        {"x-ratelimit-limit": "1, 15000"},
        {"x-ratelimit-limit": "1, 15000", "x-ratelimit-remaining": "banana"},
        {"x-ratelimit-limit": "1, 15000", "x-ratelimit-remaining": "1, 99999"},
        {"x-ratelimit-limit": "15000", "x-ratelimit-remaining": "100"},
    ):
        assert reported_from(headers) is None


async def test_the_search_wire_format_of_brave_is_unchanged() -> None:
    server = Server(
        respond(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://a.example.com",
                            "title": "T",
                            "description": "D",
                            "page_age": "2026-10-01",
                        }
                    ]
                }
            },
        )
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        found = await BraveSearch(BRAVE_KEY).search(client, "q", count=5, freshness_days=7)
    (request,) = server.requests
    assert dict(request.url.params) == {"q": "q", "count": "5", "freshness": "pw"}
    assert found == [
        {"url": "https://a.example.com", "title": "T", "snippet": "D", "published": "2026-10-01"}
    ]
