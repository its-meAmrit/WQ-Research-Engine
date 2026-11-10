"""The Discover API over ASGI: the acceptance workflow, end to end, with a restart in the middle.

Real app, real database, real routes. Tavily is a mock transport, pages come from ``FakeWeb``,
the model is the shared fake. Nothing reaches the network or BRAIN.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy import func, select

from alpha_harness.config import Settings
from alpha_harness.db.models import SimulationRecord
from alpha_harness.integrations.service import WebResearchService
from alpha_harness.main import app
from alpha_harness.research.discover.robots import RobotsPolicy
from alpha_harness.research.discover.service import DiscoveryService
from alpha_harness.research.scholarly import Scholarly
from alpha_harness.research.service import ResearchService
from alpha_harness.state import AppState

from . import discovery_fixtures as df
from . import paper_fixtures as fx
from .conftest import FakeLLM
from .test_discovery import markdown
from .test_paper_pipeline import operators

D = "/api/research/discover"
scope = {"instrumentType": "EQUITY", "region": "USA", "delay": 1, "universe": "TOP3000"}
Boot = Callable[[], "AsyncIterator[httpx2.AsyncClient]"]


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM) -> None:
        self.tmp_path, self.monkeypatch, self.llm = tmp_path, monkeypatch, llm
        self.net = df.Tavily()
        self.pages = fx.FakeWeb()
        self.states: list[AppState] = []

    async def boot(self) -> httpx2.AsyncClient:
        self.monkeypatch.setattr(
            "alpha_harness.state.Settings", lambda: Settings(data_dir=self.tmp_path)
        )
        state = AppState()
        await state.db.create_all()
        await state.registry.seed()
        state.llm = self.llm  # pyright: ignore[reportAttributeAccessIssue]
        state.web = WebResearchService(
            state.db, state.sealer, env={}, transport=httpx2.MockTransport(self.net)
        )
        registries = fx.Registries()
        state.research = ResearchService(
            state.db,
            self.llm,  # pyright: ignore[reportArgumentType]
            queries=fx.usa_catalogue(),
            operators=operators,
            scholarly=lambda: Scholarly(transport=registries.transport),
            fetch=self.pages,
        )
        state.discovery = DiscoveryService(
            state.db,
            state.research,
            state.web,
            state.registry,
            robots=RobotsPolicy(self.pages),
            fetch=self.pages,
        )
        app.state.harness = state
        self.states.append(state)
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"x-harness-client": "1"},
        )

    async def close(self) -> None:
        for state in self.states:
            await state.db.dispose()


@pytest.fixture
async def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM
) -> AsyncIterator[World]:
    w = World(tmp_path, monkeypatch, llm)
    yield w
    await w.close()


async def job_done(http: httpx2.AsyncClient, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(10):
        while True:
            body = (await http.get(f"/api/research/jobs/{job_id}")).json()
            if body["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return body
            await asyncio.sleep(0.02)


async def test_the_acceptance_workflow(world: World) -> None:
    http = await world.boot()
    async with http:
        # Tavily is set up under API Integration.
        stored = await http.put(
            "/api/integrations/web/providers/tavily/key", json={"key": df.TAVILY_KEY}
        )
        assert stored.status_code in {200, 201}, stored.text

        # 1-2. Open Discover and see what it offers.
        options = (await http.get(f"{D}/options")).json()
        assert "Earnings surprise" in options["topics"]
        assert {s["id"] for s in options["sources"]} >= {"web_research", "academic_indexes"}
        assert any(p["id"] == "tavily" and p["configured"] for p in options["providers"])
        assert options["limits"]["maxResults"] == 20

        # 3. Enter the topic; the queries are proposed and editable.
        proposals = (await http.post(f"{D}/queries", json={"topic": df.TOPIC})).json()
        assert 3 <= len(proposals) <= 5

        # The estimate says what a search costs before it is sent.
        body = {"topic": df.TOPIC, "sources": ["web_research"], "maxResults": 10}
        estimate = (await http.post(f"{D}/estimate", json=body)).json()
        assert estimate["requests"] == 1
        assert estimate["searchCredits"] == 1
        assert world.net.requests == []

        # 4. Discover with Tavily, at most ten results.
        started = await http.post(D, json=body | {"provider": "tavily"})
        assert started.status_code == 202, started.text
        job_id = started.json()["jobId"]
        job = await job_done(http, job_id)
        assert job["status"] == "completed"

        # 5. Review the sources, with attribution.
        hits = (await http.get(f"{D}/jobs/{job_id}/hits")).json()
        assert [h["url"] for h in hits] == [df.ARTICLE_URL, df.OTHER_URL]
        assert hits[0]["domain"] == "quant.example.com"
        assert hits[0]["retrievedAt"]
        assert world.net.sent("/search")[0]["max_results"] == 10
        assert world.llm.calls == []  # discovering reads results only

        # 6. Extract ideas from the first.
        world.net.pages[df.ARTICLE_URL] = markdown()
        world.llm.replies.append(df.two_web_ideas())
        extracting = await http.post(
            f"{D}/extract",
            json={
                "hitIds": [hits[0]["id"]],
                "scope": {
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "delay": 1,
                    "universe": "TOP3000",
                },
            },
        )
        assert extracting.status_code == 202, extracting.text
        done = await job_done(http, extracting.json()["jobId"])
        assert done["status"] == "completed", done
        idea_ids = done["result"]["ideaIds"]
        assert len(idea_ids) == 2

        # 7. The idea shows its feasibility, ranking and source.
        idea = (await http.get(f"/api/research/ideas/{idea_ids[0]}")).json()
        assert idea["brainAssessment"] is not None
        assert idea["ranking"]["brainFeasibility"]["level"] in {"high", "medium", "low", "unknown"}
        assert idea["source"]["canonicalUrl"] == df.ARTICLE_URL
        assert idea["status"] == "new"

        # 8. Save it, favourite it and mark it reviewed.
        saved = await http.patch(
            f"/api/research/ideas/{idea_ids[0]}",
            json={"status": "saved", "favorite": True, "reviewed": True},
        )
        assert saved.status_code == 200, saved.text

    # 9. Restart: a fresh application over the same data still has it.
    await world.states[0].db.dispose()
    reopened = await world.boot()
    async with reopened:
        again = (await reopened.get(f"/api/research/ideas/{idea_ids[0]}")).json()
        assert again["status"] == "saved"
        assert again["favorite"] is True
        assert again["reviewedAt"]
        listed = (
            await reopened.get(
                "/api/research/ideas", params={"favorite": "true", "domain": "quant.example.com"}
            )
        ).json()
        assert [i["id"] for i in listed] == [idea_ids[0]]
        hits_again = (await reopened.get(f"{D}/jobs/{job_id}/hits")).json()
        assert len(hits_again) == 2

        # 10. Send to the Alpha Builder: a draft with no expression, and no simulation.
        sent = await reopened.post(
            f"/api/research/ideas/{idea_ids[0]}/handoff",
            json={"scope": scope},
        )
        assert sent.status_code in {200, 201}, sent.text
        assert not sent.json().get("expression")
        async with world.states[-1].db.session() as session:
            assert await session.scalar(select(func.count()).select_from(SimulationRecord)) == 0
        assert world.net.hosts == {"api.tavily.com"}


async def test_a_missing_key_is_a_clear_422(world: World) -> None:
    http = await world.boot()
    async with http:
        refused = await http.post(D, json={"topic": df.TOPIC, "sources": ["web_research"]})
        assert refused.status_code == 422
        assert refused.json()["detail"]["code"] == "no_search_provider" or "no_search_provider" in (
            refused.text
        )


async def test_the_source_registry_can_be_edited(world: World) -> None:
    http = await world.boot()
    async with http:
        off = await http.patch(f"{D}/sources/quant_blogs", json={"enabled": False})
        assert off.status_code == 200
        assert off.json()["enabled"] is False
        bad = await http.post(
            f"{D}/sources", json={"label": "Local", "feedUrl": "http://127.0.0.1/feed.xml"}
        )
        assert bad.status_code == 422
        added = await http.post(
            f"{D}/sources", json={"label": "Mine", "feedUrl": "https://feeds.example.com/x.xml"}
        )
        assert added.status_code == 201
        gone = await http.delete(f"{D}/sources/{added.json()['id']}")
        assert gone.status_code == 204
        builtin = await http.delete(f"{D}/sources/web_research")
        assert builtin.status_code in {409, 422}
