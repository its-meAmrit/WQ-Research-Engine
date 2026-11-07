"""The real FastAPI app over ASGI: routes, the origin/CSRF middleware and the exception
handlers all run. Only the LLM is the test fake, and the data directory is a temp one."""

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from alpha_harness.config import Settings
from alpha_harness.main import app
from alpha_harness.spc.execution import SpcRunner
from alpha_harness.spc.service import SpcService
from alpha_harness.state import AppState

from .conftest import GOOD, REF, FakeLLM, position

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

PROMPT = "Pick equities. Output only the JSON array."


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM
) -> AsyncIterator[httpx2.AsyncClient]:
    monkeypatch.setattr("alpha_harness.state.Settings", lambda: Settings(data_dir=tmp_path))
    state = AppState()
    await state.db.create_all()
    state.spc = SpcService(
        state.db,
        llm,  # pyright: ignore[reportArgumentType]
        tmp_path,
        None,
        SpcRunner(state.db, llm, tmp_path, None, provider=lambda: None),  # pyright: ignore[reportArgumentType]
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


async def wait(client: httpx2.AsyncClient, run_id: int) -> dict[str, Any]:
    async with asyncio.timeout(5):
        while True:
            body = (await client.get(f"/api/spc/runs/{run_id}")).json()
            if body["status"] in {"completed", "failed", "cancelled"}:
                return body
            await asyncio.sleep(0.01)


async def make_prompt(client: httpx2.AsyncClient, **over: Any) -> dict[str, Any]:
    response = await client.post(
        "/api/spc/prompts",
        json={"name": "Semis", "promptText": PROMPT, "weight": 0.3, "preferredModel": REF} | over,
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_options_report_models_limits_and_unverified_schema(
    client: httpx2.AsyncClient,
) -> None:
    body = (await client.get("/api/spc/options")).json()
    assert body["models"][0]["ref"] == REF
    assert body["frequencies"] == ["daily", "weekly", "monthly", "quarterly"]
    assert body["limits"]["promptMaxChars"] == 10_000
    assert body["limits"]["maxActivePrompts"] == 50
    assert body["schemaProfile"]["verified"] is False
    assert body["research"]["configured"] in {True, False}


async def test_generate_is_free_and_stateless(client: httpx2.AsyncClient, llm: FakeLLM) -> None:
    response = await client.post(
        "/api/spc/generate",
        json={"thesis": "AI infrastructure winners", "approaches": ["hybrid", "valuation"]},
    )
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert [c["approach"] for c in candidates] == ["hybrid", "valuation"]
    assert all(c["characters"] <= 10_000 and "company_name" in c["promptText"] for c in candidates)
    assert llm.calls == []
    assert (await client.get("/api/spc/prompts")).json() == []
    bad = await client.post("/api/spc/generate", json={"thesis": "x", "approaches": ["nope"]})
    assert bad.status_code == 422


async def test_full_workflow_over_http(client: httpx2.AsyncClient, llm: FakeLLM) -> None:
    raw = json.dumps(GOOD)
    llm.replies = [raw]
    prompt = await make_prompt(client)
    pid = prompt["id"]

    queued = await client.post(f"/api/spc/prompts/{pid}/test", json={"research": "off"})
    assert queued.status_code == 202
    assert queued.json()["status"] in {"queued", "researching", "generating", "validating"}
    run = await wait(client, queued.json()["id"])
    assert run["status"] == "completed"
    assert run["sampleJson"] == raw
    assert len(run["portfolio"]) == 3

    listed = (await client.get("/api/spc/runs", params={"promptId": pid})).json()
    assert [r["id"] for r in listed] == [run["id"]]
    assert listed[0]["positions"] == 3

    analytics = (await client.get(f"/api/spc/runs/{run['id']}/analytics")).json()
    assert analytics["summary"]["positions"] == 3
    assert analytics["performance"]["available"] is False

    export = (await client.get(f"/api/spc/prompts/{pid}/export")).json()
    assert export["ready"] is True
    assert export["sampleJson"] == raw
    assert export["package"]["sampleOutput"] == raw
    assert export["schemaVerified"] is False

    edited = await client.patch(f"/api/spc/prompts/{pid}", json={"promptText": PROMPT + " More."})
    assert edited.json()["latestVersion"] == 2
    assert (await client.get(f"/api/spc/prompts/{pid}/export")).json()["ready"] is False
    cmp = (await client.get(f"/api/spc/prompts/{pid}/compare", params={"a": 1, "b": 2})).json()
    assert cmp["added"] + cmp["removed"] >= 1


async def test_invalid_output_is_visible_not_a_server_error(
    client: httpx2.AsyncClient, llm: FakeLLM
) -> None:
    llm.replies = [[position(isin="US0378331006")]]
    prompt = await make_prompt(client)
    queued = await client.post(f"/api/spc/prompts/{prompt['id']}/test", json={"research": "off"})
    run = await wait(client, queued.json()["id"])
    assert run["status"] == "failed"
    assert run["errorCode"] == "invalid_output"
    assert run["validation"]["issues"][0]["code"] == "invalid_isin"
    assert run["rawOutput"]


@pytest.mark.parametrize(
    ("method", "path", "body", "status", "code"),
    [
        ("GET", "/api/spc/prompts/999", None, 404, "spc_not_found"),
        ("GET", "/api/spc/runs/999", None, 404, "spc_not_found"),
        ("POST", "/api/spc/runs/999/cancel", {}, 404, "spc_not_found"),
        ("POST", "/api/spc/prompts", {"name": "x", "promptText": "a" * 10_001}, 422, "spc_invalid"),
        ("POST", "/api/spc/prompts", {"name": "x", "promptText": "p", "weight": 1.5}, 422, None),
        (
            "POST",
            "/api/spc/prompts",
            {"name": "x", "promptText": "p", "weight": 0.123},
            422,
            "spc_invalid",
        ),
        (
            "POST",
            "/api/spc/prompts",
            {"name": "x", "promptText": "p", "updateFrequency": "hourly"},
            422,
            None,
        ),
        ("POST", "/api/spc/prompts", {"name": "", "promptText": "p"}, 422, None),
    ],
)
async def test_errors_are_precise(
    client: httpx2.AsyncClient, method: str, path: str, body: Any, status: int, code: str | None
) -> None:
    response = await client.request(method, path, json=body)
    assert response.status_code == status, response.text
    if code:
        assert response.json()["detail"]["code"] == code
        assert response.json()["detail"]["message"]


async def test_no_api_key_is_a_clear_client_error(client: httpx2.AsyncClient, llm: FakeLLM) -> None:
    prompt = await make_prompt(client)
    llm.keyed = False
    response = await client.post(f"/api/spc/prompts/{prompt['id']}/test", json={"research": "off"})
    assert 400 <= response.status_code < 500
    assert "key" in response.text.lower()
    assert (await client.get("/api/spc/runs")).json() == []


async def test_double_click_is_a_conflict(client: httpx2.AsyncClient, llm: FakeLLM) -> None:
    llm.delay = 0.3
    llm.replies = [GOOD]
    prompt = await make_prompt(client)
    url = f"/api/spc/prompts/{prompt['id']}/test"
    first = await client.post(url, json={"research": "off"})
    second = await client.post(url, json={"research": "off"})
    assert (first.status_code, second.status_code) == (202, 409)
    await wait(client, first.json()["id"])


async def test_cancel_over_http(client: httpx2.AsyncClient, llm: FakeLLM) -> None:
    llm.delay = 5
    llm.replies = [GOOD]
    prompt = await make_prompt(client)
    run = (
        await client.post(f"/api/spc/prompts/{prompt['id']}/test", json={"research": "off"})
    ).json()
    await asyncio.sleep(0.05)
    cancelled = await client.post(f"/api/spc/runs/{run['id']}/cancel")
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.post(f"/api/spc/runs/{run['id']}/cancel")).status_code == 409


# -- existing behaviour is unchanged -----------------------------------------------------


async def test_writes_still_need_the_app_header(client: httpx2.AsyncClient) -> None:
    bare = await client.post(
        "/api/spc/generate", json={"thesis": "x"}, headers={"x-harness-client": ""}
    )
    assert bare.status_code == 403
    cross = await client.get("/api/spc/options", headers={"sec-fetch-site": "cross-site"})
    assert cross.status_code == 403


async def test_foreign_host_is_still_refused(client: httpx2.AsyncClient) -> None:
    assert (
        await client.get("/api/spc/options", headers={"host": "evil.example"})
    ).status_code == 400


async def test_existing_routes_and_openapi_still_work(client: httpx2.AsyncClient) -> None:
    assert (await client.get("/api/health")).status_code == 200
    paths = set((await client.get("/openapi.json")).json()["paths"])
    for existing in (
        "/api/alphas/summary",
        "/api/auth/login",
        "/api/catalog/datasets",
        "/api/chat/threads",
        "/api/competitions",
        "/api/lab-tasks",
        "/api/llm/keys",
        "/api/llm/models",
        "/api/health",
    ):
        assert existing in paths, existing
    # Every SPC route is under its own prefix, and none touches simulation or submission.
    spc = {p for p in paths if "spc" in p}
    assert spc
    assert all(p.startswith("/api/spc/") for p in spc)
    assert not any(word in p for p in spc for word in ("sims", "submit", "brain"))


async def test_existing_llm_chat_status_still_reports_models(client: httpx2.AsyncClient) -> None:
    # An existing LLM route, served by the real LLMService, is unaffected by the new
    # keyword on generate().
    assert (await client.get("/api/llm/models")).status_code == 200
