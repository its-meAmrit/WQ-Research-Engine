"""The Web Research API over ASGI, and SPC research using what it configures.

The real app, the real database and the real sealer. Tavily and Brave are an
``httpx2.MockTransport``: nothing leaves the process, so no test spends a credit, and the
transport records every host so a test can show SPC never talks to BRAIN.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from alpha_harness.config import Settings
from alpha_harness.integrations import legacy
from alpha_harness.integrations.service import WebResearchService
from alpha_harness.main import app
from alpha_harness.spc.execution import SpcRunner
from alpha_harness.spc.service import SpcService
from alpha_harness.state import AppState

from .conftest import GOOD, REF, FakeLLM
from .test_integrations import BRAVE_KEY, ENV_KEY, TAVILY_KEY, Server

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

PROMPT = "Pick equities. Output only the JSON array."
WEB = "/api/integrations/web"

TAVILY_RESULTS = {
    "results": [
        {
            "url": "https://ir.example.com/q3",
            "title": "Acme Q3",
            "content": "Revenue up.",
            "published_date": "2026-11-01",
        }
    ]
}


class Net:
    """Everything the application tried to send, and how to answer it."""

    def __init__(self) -> None:
        self.server = Server(self.answer)
        self.search_body: dict[str, Any] = TAVILY_RESULTS

    def answer(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/search":
            return httpx2.Response(200, json=self.search_body)
        return Server.ok(request)

    @property
    def hosts(self) -> set[str]:
        return {r.url.host for r in self.server.requests}


@pytest.fixture
def net() -> Net:
    return Net()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The legacy variables, empty unless a test sets them. Cleared from the real environment
    too, so a developer's own key can never leak into a test."""
    monkeypatch.delenv(legacy.ENV_PROVIDER, raising=False)
    monkeypatch.delenv(legacy.ENV_KEY, raising=False)
    return {}


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM, net: Net, env: dict[str, str]
) -> AsyncIterator[httpx2.AsyncClient]:
    monkeypatch.setattr("alpha_harness.state.Settings", lambda: Settings(data_dir=tmp_path))
    state = AppState()
    await state.db.create_all()
    transport = httpx2.MockTransport(net.server)
    state.web = WebResearchService(state.db, state.sealer, env=env, transport=transport)
    state.spc = SpcService(
        state.db,
        llm,  # pyright: ignore[reportArgumentType]
        tmp_path,
        None,
        SpcRunner(state.db, llm, tmp_path, None, web=state.web, transport=transport),  # pyright: ignore[reportArgumentType]
        web=state.web,
    )
    app.state.harness = state
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"x-harness-client": "1"},
    ) as http:
        yield http
    await state.db.dispose()
    await state.llm.aclose()


async def put_key(client: httpx2.AsyncClient, provider: str, key: str, **extra: Any) -> Any:
    return await client.put(f"{WEB}/providers/{provider}/key", json={"key": key, **extra})


async def provider(client: httpx2.AsyncClient, name: str) -> dict[str, Any]:
    body = (await client.get(WEB)).json()
    return next(p for p in body["providers"] if p["id"] == name)


async def wait(client: httpx2.AsyncClient, run_id: int) -> dict[str, Any]:
    async with asyncio.timeout(5):
        while True:
            body = (await client.get(f"/api/spc/runs/{run_id}")).json()
            if body["status"] in {"completed", "failed", "cancelled"}:
                return body
            await asyncio.sleep(0.01)


async def start(client: httpx2.AsyncClient, research: str = "auto") -> dict[str, Any]:
    made = await client.post(
        "/api/spc/prompts",
        json={"name": "Semis", "promptText": PROMPT, "weight": 0.3, "preferredModel": REF},
    )
    assert made.status_code == 201, made.text
    queued = await client.post(
        f"/api/spc/prompts/{made.json()['id']}/test", json={"research": research}
    )
    assert queued.status_code == 202, queued.text
    return await wait(client, queued.json()["id"])


# --- the API ----------------------------------------------------------------------------


async def test_a_fresh_install_lists_both_services_unconfigured(client: httpx2.AsyncClient) -> None:
    body = (await client.get(WEB)).json()
    assert [p["id"] for p in body["providers"]] == ["tavily", "brave"]
    tavily, brave = body["providers"]
    assert tavily["capabilities"] == ["search", "extract"]
    assert brave["capabilities"] == ["search"]
    assert not tavily["configured"]
    assert tavily["checkCost"]["mayConsumeCredit"] is True
    assert body["active"]["configured"] is False
    assert "API Integration" in body["active"]["help"]
    assert body["preferredProvider"] is None
    assert body["legacyEnvironment"]["present"] is False


async def test_adding_a_key_returns_a_hint_and_never_the_key(client: httpx2.AsyncClient) -> None:
    response = await put_key(client, "tavily", TAVILY_KEY, dailyRequestBudget=25)
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] and body["enabled"]
    assert body["hint"] == "tvly-…cdef"
    assert body["usage"]["dailyBudget"] == 25
    listing = (await client.get(WEB)).text
    for text in (response.text, listing):
        assert TAVILY_KEY not in text
        assert "SECRET" not in text


@pytest.mark.parametrize("bad", ["", "   ", "short", "two words here 12345", "Bearer abcdefghijkl"])
async def test_an_invalid_key_is_refused_with_a_reason(
    client: httpx2.AsyncClient, bad: str
) -> None:
    response = await put_key(client, "tavily", bad)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_key"
    assert not (await provider(client, "tavily"))["configured"]


async def test_an_unknown_provider_is_a_validation_error_everywhere(
    client: httpx2.AsyncClient,
) -> None:
    for call in (
        client.put(f"{WEB}/providers/bing/key", json={"key": TAVILY_KEY}),
        client.patch(f"{WEB}/providers/bing", json={"enabled": True}),
        client.delete(f"{WEB}/providers/bing/key"),
        client.post(f"{WEB}/providers/bing/check", json={"acknowledgeCost": True}),
        client.put(f"{WEB}/preferred", json={"provider": "bing"}),
    ):
        assert (await call).status_code == 422


async def test_a_duplicate_key_is_a_conflict(client: httpx2.AsyncClient) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    again = await put_key(client, "brave", TAVILY_KEY)
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "duplicate_key"
    assert TAVILY_KEY not in again.text


async def test_a_budget_out_of_range_is_refused(client: httpx2.AsyncClient) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    for bad in (0, -1, 10_000_000):
        patched = await client.patch(f"{WEB}/providers/tavily", json={"dailyRequestBudget": bad})
        assert patched.status_code == 422


async def test_enable_disable_and_remove(client: httpx2.AsyncClient) -> None:
    await put_key(client, "brave", BRAVE_KEY)
    off = await client.patch(f"{WEB}/providers/brave", json={"enabled": False})
    assert off.json()["enabled"] is False
    assert (await client.get(WEB)).json()["active"]["configured"] is False
    on = await client.patch(f"{WEB}/providers/brave", json={"enabled": True})
    assert on.json()["enabled"] is True
    assert (await client.get(WEB)).json()["active"]["provider"] == "brave"

    assert (await client.delete(f"{WEB}/providers/brave/key")).status_code == 204
    assert not (await provider(client, "brave"))["configured"]
    gone = await client.delete(f"{WEB}/providers/brave/key")
    assert gone.status_code == 404
    assert gone.json()["detail"]["code"] == "not_configured"
    missing = await client.patch(f"{WEB}/providers/brave", json={"enabled": True})
    assert missing.status_code == 404


async def test_the_default_can_be_chosen_and_cleared(client: httpx2.AsyncClient) -> None:
    refused = await client.put(f"{WEB}/preferred", json={"provider": "brave"})
    assert refused.status_code == 404  # no key to prefer
    await put_key(client, "tavily", TAVILY_KEY)
    await put_key(client, "brave", BRAVE_KEY)
    chosen = (await client.put(f"{WEB}/preferred", json={"provider": "brave"})).json()
    assert chosen["preferredProvider"] == "brave"
    assert chosen["active"]["provider"] == "brave"
    assert next(p for p in chosen["providers"] if p["id"] == "brave")["isPreferred"] is True
    cleared = (await client.put(f"{WEB}/preferred", json={"provider": None})).json()
    assert cleared["preferredProvider"] is None
    assert cleared["active"]["provider"] == "tavily"


async def test_a_test_that_may_cost_a_credit_needs_the_users_ok_first(
    client: httpx2.AsyncClient, net: Net
) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    refused = await client.post(f"{WEB}/providers/tavily/check", json={})
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "confirmation_required"
    assert "credit" in refused.json()["detail"]["message"]
    assert net.server.requests == []

    done = await client.post(f"{WEB}/providers/tavily/check", json={"acknowledgeCost": True})
    assert done.status_code == 200
    assert done.json()["ok"] is True
    assert done.json()["reported"]["remaining"] == 880
    assert len(net.server.requests) == 1
    shown = await provider(client, "tavily")
    assert shown["lastOkAt"] and shown["lastError"] is None
    assert shown["usage"]["requests"] == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "auth_failed"), (429, "rate_limited"), (432, "quota_exceeded"), (503, "unavailable")],
)
async def test_a_failed_check_is_a_normal_answer_that_says_what_to_do(
    client: httpx2.AsyncClient, net: Net, status: int, code: str
) -> None:
    net.server.handler = lambda _r: httpx2.Response(status)
    await put_key(client, "tavily", TAVILY_KEY)
    response = await client.post(f"{WEB}/providers/tavily/check", json={"acknowledgeCost": True})
    assert response.status_code == 200
    body = response.json()
    assert (body["ok"], body["code"]) == (False, code)
    assert TAVILY_KEY not in response.text
    assert (await provider(client, "tavily"))["lastErrorCode"] == code


async def test_the_daily_budget_stops_requests_with_a_429(
    client: httpx2.AsyncClient, net: Net
) -> None:
    await put_key(client, "tavily", TAVILY_KEY, dailyRequestBudget=1)
    ok = {"acknowledgeCost": True}
    assert (await client.post(f"{WEB}/providers/tavily/check", json=ok)).json()["ok"]
    spent = await client.post(f"{WEB}/providers/tavily/check", json=ok)
    assert spent.status_code == 429
    assert spent.json()["detail"]["code"] == "budget_exhausted"
    assert spent.json()["detail"]["retryAfter"] > 0
    assert len(net.server.requests) == 1


async def test_extraction_over_the_api(client: httpx2.AsyncClient, net: Net) -> None:
    net.server.handler = lambda _r: httpx2.Response(
        200, json={"results": [{"url": "https://a.example.com", "raw_content": "text"}]}
    )
    none = await client.post(f"{WEB}/extract", json={"urls": ["https://a.example.com"]})
    assert none.status_code == 404
    await put_key(client, "tavily", TAVILY_KEY)
    body = (await client.post(f"{WEB}/extract", json={"urls": ["https://a.example.com"]})).json()
    assert body["provider"] == "tavily"
    assert body["pages"][0]["content"] == "text"
    private = await client.post(f"{WEB}/extract", json={"urls": ["http://192.168.0.1/"]})
    assert private.status_code == 422
    assert (await client.post(f"{WEB}/extract", json={"urls": []})).status_code == 422


async def test_keys_survive_in_the_database_not_in_responses_or_logs(
    client: httpx2.AsyncClient, capfd: pytest.CaptureFixture[str]
) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    await client.post(f"{WEB}/providers/tavily/check", json={"acknowledgeCost": True})
    out, err = capfd.readouterr()
    assert TAVILY_KEY not in out + err


# --- SPC ---------------------------------------------------------------------------------


async def test_spc_reports_research_status_from_the_new_settings(
    client: httpx2.AsyncClient,
) -> None:
    before = (await client.get("/api/spc/options")).json()["research"]
    assert before["configured"] is False
    assert "API Integration › Web Research" in before["help"]
    await put_key(client, "tavily", TAVILY_KEY)
    after = (await client.get("/api/spc/options")).json()["research"]
    assert (after["configured"], after["provider"], after["source"]) == (True, "tavily", "store")


async def test_spc_uses_a_key_added_in_the_app(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    llm.replies = [GOOD]
    await put_key(client, "tavily", TAVILY_KEY, dailyRequestBudget=20)
    run = await start(client)

    assert run["status"] == "completed"
    assert run["research"]["status"] == "done"
    assert [e["title"] for e in run["evidence"]] == ["Acme Q3"]
    searches = [r for r in net.server.requests if r.url.path == "/search"]
    assert searches
    assert all(r.headers["authorization"] == f"Bearer {TAVILY_KEY}" for r in searches)
    assert TAVILY_KEY not in json.dumps(run)
    # What it used is counted against the user's budget, and the key is marked healthy.
    shown = await provider(client, "tavily")
    assert shown["usage"]["requests"] == len(searches)
    assert shown["lastOkAt"] is not None


async def test_spc_never_talks_to_brain_or_anyone_but_the_search_service(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    llm.replies = [GOOD]
    await put_key(client, "tavily", TAVILY_KEY)
    run = await start(client)
    assert run["status"] == "completed"
    assert net.hosts == {"api.tavily.com"}
    assert not any("worldquant" in h or "brain" in h for h in net.hosts)


async def test_spc_still_works_from_the_legacy_environment(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net, env: dict[str, str]
) -> None:
    env |= {legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY}
    options = (await client.get("/api/spc/options")).json()["research"]
    assert (options["configured"], options["source"]) == (True, "environment")
    assert "environment variable" in options["message"]

    llm.replies = [GOOD]
    run = await start(client)
    assert run["status"] == "completed"
    assert run["research"]["status"] == "done"
    searches = [r for r in net.server.requests if r.url.path == "/search"]
    assert all(r.headers["authorization"] == f"Bearer {ENV_KEY}" for r in searches)
    # Used, not saved.
    assert not (await provider(client, "tavily"))["configured"]
    assert ENV_KEY not in (await client.get(WEB)).text


async def test_a_stored_key_wins_over_the_environment_in_a_run(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net, env: dict[str, str]
) -> None:
    env |= {legacy.ENV_PROVIDER: "tavily", legacy.ENV_KEY: ENV_KEY}
    await put_key(client, "tavily", TAVILY_KEY)
    llm.replies = [GOOD]
    await start(client)
    keys = {r.headers["authorization"] for r in net.server.requests if r.url.path == "/search"}
    assert keys == {f"Bearer {TAVILY_KEY}"}


async def test_a_run_with_no_research_configured_says_so_and_still_completes_on_auto(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    llm.replies = [GOOD]
    run = await start(client, "auto")
    assert run["status"] == "completed"
    assert run["research"]["status"] == "not_configured"
    assert "API Integration › Web Research" in run["research"]["help"]
    assert net.server.requests == []


async def test_required_research_without_configuration_fails_with_the_new_setup_message(
    client: httpx2.AsyncClient, llm: FakeLLM
) -> None:
    run = await start(client, "required")
    assert run["status"] == "failed"
    assert run["errorCode"] == "research_not_configured"
    assert "API Integration › Web Research" in run["errorMessage"]
    assert llm.calls == []


async def test_a_disabled_key_is_not_used_by_a_run(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    await client.patch(f"{WEB}/providers/tavily", json={"enabled": False})
    llm.replies = [GOOD]
    run = await start(client)
    assert run["research"]["status"] == "not_configured"
    assert net.server.requests == []


async def test_a_refused_key_degrades_on_auto_and_marks_the_key(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    net.server.handler = lambda _r: httpx2.Response(401)
    await put_key(client, "tavily", TAVILY_KEY)
    llm.replies = [GOOD]
    run = await start(client)
    assert run["status"] == "completed"
    assert run["research"]["status"] == "failed"
    assert "refused the API key" in run["research"]["error"]
    assert TAVILY_KEY not in json.dumps(run)
    shown = await provider(client, "tavily")
    assert shown["lastOkAt"] is None
    assert shown["lastError"]


async def test_required_research_with_a_refused_key_fails_the_run(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    net.server.handler = lambda _r: httpx2.Response(429, headers={"Retry-After": "7"})
    await put_key(client, "tavily", TAVILY_KEY)
    run = await start(client, "required")
    assert run["status"] == "failed"
    assert llm.calls == []
    assert TAVILY_KEY not in json.dumps(run)


async def test_a_spent_budget_limits_the_searches_a_run_sends(
    client: httpx2.AsyncClient, llm: FakeLLM, net: Net
) -> None:
    await put_key(client, "tavily", TAVILY_KEY, dailyRequestBudget=1)
    llm.replies = [GOOD]
    run = await start(client)
    searches = [r for r in net.server.requests if r.url.path == "/search"]
    assert len(searches) == 1
    assert run["status"] == "completed"
    assert any("budget" in n for n in run["research"].get("notes", []))

    # Nothing left: the next run sends nothing, and says why.
    net.server.requests.clear()
    llm.replies = [GOOD]
    second = await start(client)
    assert [r for r in net.server.requests if r.url.path == "/search"] == []
    assert second["research"]["status"] in {"failed", "not_configured"}


async def test_saved_runs_and_their_evidence_survive_removing_the_key(
    client: httpx2.AsyncClient, llm: FakeLLM
) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    llm.replies = [GOOD]
    run = await start(client)
    await client.delete(f"{WEB}/providers/tavily/key")
    kept = (await client.get(f"/api/spc/runs/{run['id']}")).json()
    assert kept["evidence"] == run["evidence"]
    assert kept["sampleJson"] == run["sampleJson"]


# --- existing LLM setup stays as it was ---------------------------------------------------


async def test_llm_routes_still_work_and_know_nothing_of_search_services(
    client: httpx2.AsyncClient,
) -> None:
    await put_key(client, "tavily", TAVILY_KEY)
    providers = await client.get("/api/llm/providers")
    assert providers.status_code == 200
    names = {p["id"] for p in providers.json()["providers"]}
    assert not names & {"tavily", "brave"}
    assert (await client.get("/api/llm/keys")).json()["keys"] == []
    assert (await client.get("/api/llm/models")).status_code == 200
    assert (await client.get("/api/llm/prompts")).status_code == 200
    assert TAVILY_KEY not in providers.text
