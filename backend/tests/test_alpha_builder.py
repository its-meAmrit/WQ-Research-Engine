"""Alpha Builder over ASGI: idea to candidate to idle task, with BRAIN and the model mocked.

Real app, real database, real routes. The model is the shared fake; the field catalogue is a
fake with real field types; BRAIN's metadata (operators, settings schema) is fixed data. Nothing
reaches BRAIN, so no test can spend simulation quota. A task the builder makes is checked to be
*idle* with every trial parked: nothing here ever starts one.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy import func, select

from alpha_harness.alpha_builder.service import AlphaBuilderService
from alpha_harness.catalog.queries import FieldFilter, Tuple4
from alpha_harness.config import Settings
from alpha_harness.db.models import (
    AlphaCandidate,
    ResearchIdea,
    ResearchSource,
    SimulationRecord,
    Study,
    Trial,
)
from alpha_harness.labs.params import PASTE_ALPHAS
from alpha_harness.main import app
from alpha_harness.research.service import ResearchService
from alpha_harness.state import AppState

from . import paper_fixtures as fx
from .conftest import REF, FakeLLM

B = "/api/research/builder"
SCOPE = {"instrumentType": "EQUITY", "region": "USA", "delay": 1, "universe": "TOP3000"}
SETTINGS: dict[str, Any] = {
    **SCOPE,
    "language": "FASTEXPR",
    "neutralization": "SUBINDUSTRY",
    "decay": 4,
    "truncation": 0.08,
    "pasteurization": "ON",
    "nanHandling": "OFF",
    "maxTrade": "OFF",
    "maxPosition": "OFF",
}

OPERATORS = [
    *fx.OPERATORS,
    {
        "name": "ts_mean",
        "category": "Time Series",
        "scope": ["REGULAR"],
        "definition": "ts_mean(x, d)",
    },
    {
        "name": "zscore",
        "category": "Cross Sectional",
        "scope": ["REGULAR"],
        "definition": "zscore(x)",
    },
    {"name": "vec_avg", "category": "Vector", "scope": ["REGULAR"], "definition": "vec_avg(x)"},
    {
        "name": "group_neutralize",
        "category": "Group",
        "scope": ["REGULAR"],
        "definition": "group_neutralize(x, group)",
    },
    {
        "name": "ts_delay",
        "category": "Time Series",
        "scope": ["REGULAR"],
        "definition": "ts_delay(x, d)",
    },
]


def _choices(*values: Any) -> list[dict[str, Any]]:
    return [{"value": v, "label": str(v)} for v in values]


def _by_region(per_region: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "instrumentType": {"EQUITY": {"region": {r: _choices(*v) for r, v in per_region.items()}}}
    }


SCHEMA: dict[str, Any] = {
    "instrumentType": {"label": "Instrument Type", "choices": _choices("EQUITY")},
    "region": {
        "label": "Region",
        "choices": {"instrumentType": {"EQUITY": _choices("USA", "GLB", "EUR")}},
    },
    "delay": {"label": "Delay", "choices": _by_region({"USA": [0, 1], "GLB": [1], "EUR": [1]})},
    "universe": {
        "label": "Universe",
        "choices": _by_region(
            {"USA": ["TOP3000", "TOP1000"], "GLB": ["TOP3000"], "EUR": ["TOP1200"]}
        ),
    },
    "neutralization": {
        "label": "Neutralization",
        "choices": _by_region(
            {"USA": ["SUBINDUSTRY", "MARKET", "NONE"], "GLB": ["MARKET"], "EUR": ["MARKET"]}
        ),
    },
    "language": {"label": "Language", "choices": _choices("FASTEXPR")},
}


def rows(*extra: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "book value": [fx.row("bookvalue_ps", "Book value per share, as reported")],
        "market capitalization": [fx.row("cap", "Market capitalization", dataset_id="pv1")],
        "analyst estimates": [
            fx.row(
                "anl_vec",
                "Analyst estimates, several per day",
                field_type="VECTOR",
                dataset_id="anl",
            )
        ],
        "operating margin": [fx.row("gross_margin", "Gross margin", dataset_id="fundamental6")],
        "sentiment": [],
        **{r["field_id"]: [r] for r in extra},
    }


class Catalogue(fx.FakeQueries):
    """FakeQueries plus the by-id lookup the real catalogue has."""

    async def fields_by_id(
        self, scope: Tuple4, field_ids: list[str], *, whole_datasets: bool = False
    ) -> list[dict[str, Any]]:
        table = self.data.get((scope.region, scope.delay, scope.universe), {})
        every = {r["field_id"]: r for rs in table.values() for r in rs}
        return [
            every[f] | {"dataset_name": every[f]["dataset_id"]} for f in field_ids if f in every
        ]

    async def fields(
        self, scope: Tuple4, filters: FieldFilter, *, counted: bool = True
    ) -> dict[str, Any]:
        found = await super().fields(scope, filters, counted=counted)
        if found["results"]:
            return found
        table = self.data.get((scope.region, scope.delay, scope.universe), {})
        needle = str(filters.search or "").lower()
        hits = [
            r
            for rs in table.values()
            for r in rs
            if needle and (needle in r["field_id"].lower() or needle in r["description"].lower())
        ]
        return {"results": hits[: filters.limit], "total": len(hits)}


def catalogue() -> Catalogue:
    return Catalogue(
        {
            ("USA", 1, "TOP3000"): rows(),
            ("EUR", 1, "TOP1200"): {"book value": [fx.row("bookvalue_ps", "Book value per share")]},
        }
    )


DESIGN: dict[str, Any] = {
    "economic_hypothesis": "Cheap stocks relative to book value earn higher later returns.",
    "signal_construction": "Rank book value over market capitalization.",
    "variables": [
        {
            "name": "book value",
            "meaning": "Book value of equity per share",
            "kind": "equity_fundamental",
            "essential": True,
            "search_terms": ["book value"],
            "expected_type": "MATRIX",
            "source_backed": True,
        },
        {
            "name": "market capitalization",
            "meaning": "Market value of equity",
            "kind": "price_volume",
            "essential": True,
            "search_terms": ["market capitalization"],
            "expected_type": "MATRIX",
            "source_backed": True,
        },
    ],
    "transformations": [
        {"step": "cross-sectional rank", "operators": ["rank"], "source_backed": True}
    ],
    "expected_direction": "higher_better",
    "direction_basis": "The source's hypothesis: high book-to-market outperforms.",
    "lookback": "No look-back stated.",
    "data_timing_assumptions": "Book value as reported, with the usual reporting lag.",
    "relevant_operators": ["rank", "ts_rank", "not_an_operator"],
    "implementation_limitations": ["Per-share book value is not total equity."],
    "candidate_design": "rank(bookvalue_ps / cap)",
    "adaptations": [
        {"change": "Per-share book value", "why": "Total equity is not a BRAIN field."}
    ],
}


def variant(kind: str, expression: str, **over: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": f"{kind} variant",
        "expression": expression,
        "relation_to_source": "Implements the book-to-market hypothesis.",
        "implementation_explanation": "Ranks the ratio across the universe.",
        "source_backed_parts": ["Book-to-market ratio"],
        "adaptations": [],
        "changes_from_base": [] if kind == "replication" else ["Adds a smoothing window."],
        "limitations": ["Per-share value approximates total equity."],
    } | over


VARIANTS = {
    "variants": [
        variant("replication", "rank(bookvalue_ps / cap)"),
        variant("robust", "rank(ts_mean(bookvalue_ps / cap, 20))"),
    ]
}


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM) -> None:
        self.tmp_path, self.monkeypatch, self.llm = tmp_path, monkeypatch, llm
        self.states: list[AppState] = []
        self.operators: list[dict[str, Any]] | None = OPERATORS
        self.schema: dict[str, Any] | None = SCHEMA
        self.alphas: dict[str, dict[str, Any]] = {}
        self.queries = catalogue()

    async def boot(self) -> httpx2.AsyncClient:
        self.monkeypatch.setattr(
            "alpha_harness.state.Settings", lambda: Settings(data_dir=self.tmp_path)
        )
        state = AppState()
        await state.db.create_all()
        state.llm = self.llm  # pyright: ignore[reportAttributeAccessIssue]

        async def operators() -> list[dict[str, Any]] | None:
            return self.operators

        async def schema() -> dict[str, Any] | None:
            return self.schema

        async def by_ids(ids: list[str]) -> dict[str, dict[str, Any]]:
            return {i: self.alphas[i] for i in ids if i in self.alphas}

        state.research = ResearchService(
            state.db,
            self.llm,  # pyright: ignore[reportArgumentType]
            queries=self.queries,
            operators=operators,
        )
        state.builder = AlphaBuilderService(
            state.db,
            self.llm,
            state.research,
            self.queries,
            operators=operators,
            schema=schema,
            alphas=by_ids,
            host=state,
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


async def seed_idea(w: World, **over: Any) -> str:
    async with w.states[-1].db.session() as session:
        source = ResearchSource(title="Book-to-market revisited", import_method="upload")
        session.add(source)
        await session.flush()
        idea = ResearchIdea(
            source_id=source.id,
            title="Book-to-market",
            economic_hypothesis="Cheap stocks outperform.",
            required_variables=[],
            evidence=[{"quote": "Value stocks outperform growth stocks.", "page": 3}],
            lookahead_concerns=over.pop("lookahead_concerns", []),
            **over,
        )
        session.add(idea)
        await session.flush()
        return idea.id


async def done(http: httpx2.AsyncClient, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(10):
        while True:
            body = (await http.get(f"/api/research/jobs/{job_id}")).json()
            if body["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return body
            await asyncio.sleep(0.02)


async def design(
    http: httpx2.AsyncClient, w: World, idea_id: str, scope: dict[str, Any] = SCOPE
) -> dict[str, Any]:
    w.llm.replies.append(DESIGN)
    started = await http.post(f"{B}/ideas/{idea_id}/design", json={"scope": scope, "modelRef": REF})
    assert started.status_code == 202, started.text
    job = await done(http, started.json()["jobId"])
    assert job["status"] == "completed", job
    got = await http.get(f"{B}/designs/{job['result']['designId']}")
    return got.json()


async def generate(
    http: httpx2.AsyncClient,
    w: World,
    design_id: str,
    reply: Any = VARIANTS,
    settings: dict[str, Any] = SETTINGS,
    **over: Any,
) -> dict[str, Any]:
    w.llm.replies.append(reply)
    started = await http.post(
        f"{B}/designs/{design_id}/variants",
        json={"settings": settings, "settingsConfirmed": True, "count": 3, "modelRef": REF} | over,
    )
    assert started.status_code == 202, started.text
    return await done(http, started.json()["jobId"])


async def make(
    w: World, http: httpx2.AsyncClient, reply: Any = VARIANTS
) -> tuple[str, list[dict[str, Any]]]:
    idea_id = await seed_idea(w)
    d = await design(http, w, idea_id)
    job = await generate(http, w, d["id"], reply)
    assert job["status"] == "completed", job
    listed = (await http.get(f"{B}/ideas/{idea_id}/candidates")).json()
    return idea_id, listed


# --- idea to candidates ------------------------------------------------------------------------


async def test_idea_to_candidates_with_real_field_matching(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        d = await design(http, world, idea_id)
        names = {m["name"]: m for m in d["mapping"]}
        assert names["book value"]["status"] == "exact"
        assert names["book value"]["final"]["fieldId"] == "bookvalue_ps"
        assert names["market capitalization"]["final"]["fieldId"] == "cap"
        assert d["scope"] == SCOPE
        # The model's own list of operators is checked, not trusted.
        assert d["design"]["unknown_operators"] == ["not_an_operator"]
        job = await generate(http, world, d["id"])
        assert job["status"] == "completed"
        cands = (await http.get(f"{B}/ideas/{idea_id}/candidates")).json()
        assert [c["variantKind"] for c in cands] == ["replication", "robust"]
        base, robust = cands
        assert base["validationOutcome"] == "valid"
        assert base["status"] == "draft"
        assert base["fieldsUsed"] == ["bookvalue_ps", "cap"]
        assert base["operatorsUsed"] == ["rank"]
        assert robust["parentId"] == base["id"]
        assert robust["changes"]
        assert base["provenance"]["hypothesis"] == "Cheap stocks outperform."
        assert base["provenance"]["evidence"][0]["page"] == 3
        assert base["provenance"]["note"].startswith("Draft")
        assert "BRAIN's own simulation checks" in " ".join(base["validationDetail"]["notChecked"])
        # The model was shown real ids and told not to invent them.
        prompt = world.llm.calls[-1]["user"]
        assert "bookvalue_ps" in prompt and "Never invent either" in prompt


async def test_missing_field_blocks_generation_and_never_substitutes(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        reply = {
            **DESIGN,
            "variables": [
                {**DESIGN["variables"][0]},
                {**DESIGN["variables"][1], "name": "news sentiment", "search_terms": ["sentiment"]},
            ],
        }
        world.llm.replies.append(reply)
        started = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": SCOPE, "modelRef": REF}
        )
        job = await done(http, started.json()["jobId"])
        d = (await http.get(f"{B}/designs/{job['result']['designId']}")).json()
        sentiment = next(m for m in d["mapping"] if m["name"] == "news sentiment")
        assert sentiment["status"] == "missing" and sentiment["final"] is None
        assert [b["variable"] for b in d["blocked"]] == ["news sentiment"]
        before = len(world.llm.calls)
        refused = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={"settings": SETTINGS, "settingsConfirmed": True, "modelRef": REF},
        )
        assert refused.status_code == 409
        assert refused.json()["detail"]["code"] == "field_mapping_blocked"
        assert len(world.llm.calls) == before  # no model request was spent


async def test_proxy_needs_acceptance_and_substitution_must_exist_in_scope(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        reply = {
            **DESIGN,
            "variables": [
                {
                    **DESIGN["variables"][0],
                    "name": "operating profitability",
                    "search_terms": ["operating margin"],
                },
                DESIGN["variables"][1],
            ],
        }
        world.llm.replies.append(reply)
        started = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": SCOPE, "modelRef": REF}
        )
        job = await done(http, started.json()["jobId"])
        d = (await http.get(f"{B}/designs/{job['result']['designId']}")).json()
        proxy = next(m for m in d["mapping"] if m["name"] == "operating profitability")
        assert proxy["status"] == "proxy" and proxy["needsConfirmation"] and not proxy["usable"]
        assert "not a field described as it" in proxy["approximation"]
        # Not usable until accepted.
        assert (
            await http.post(
                f"{B}/designs/{d['id']}/variants",
                json={"settings": SETTINGS, "settingsConfirmed": True, "modelRef": REF},
            )
        ).status_code == 409
        accepted = await http.patch(
            f"{B}/designs/{d['id']}/mapping", json={"acceptedProxies": ["operating profitability"]}
        )
        assert accepted.status_code == 200
        item = next(m for m in accepted.json()["mapping"] if m["name"] == "operating profitability")
        assert item["usable"] and item["status"] == "proxy"
        # A substitution outside the scope's catalogue is refused, not stored.
        bad = await http.patch(
            f"{B}/designs/{d['id']}/mapping",
            json={"substitutions": {"market capitalization": "eur_only_field"}},
        )
        assert bad.status_code == 422 and bad.json()["detail"]["code"] == "field_not_in_scope"
        good = await http.patch(
            f"{B}/designs/{d['id']}/mapping",
            json={"substitutions": {"market capitalization": "cap"}},
        )
        entry = next(m for m in good.json()["mapping"] if m["name"] == "market capitalization")
        assert entry["status"] == "manual" and "your substitution" in entry["approximation"]


async def test_field_search_is_restricted_to_the_scope(world: World) -> None:
    http = await world.boot()
    async with http:
        hits = await http.post(f"{B}/fields/search", json={"scope": SCOPE, "text": "market"})
        assert [h["fieldId"] for h in hits.json()] == ["cap"]
        eur = {"instrumentType": "EQUITY", "region": "EUR", "delay": 1, "universe": "TOP1200"}
        assert [
            h["fieldId"]
            for h in (
                await http.post(f"{B}/fields/search", json={"scope": eur, "text": "book"})
            ).json()
        ] == ["bookvalue_ps"]


# --- validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "code", "text"),
    [
        ("rank(bogus(cap))", "rejected", "bogus is not an operator"),
        ("ts_rank(cap)", "rejected", "does not take 1 inputs"),
        ("rank(mystery_field)", "field_not_in_scope", "mystery_field is not a data field"),
        ("rank(anl_vec)", "vector_misuse", "VECTOR field"),
        ("rank(ts_delay(cap, -3))", "negative_lookback", "reads the future"),
        ("rank(cap", "syntax", "parse"),
        ("rank($x)", "placeholder", "template placeholder"),
    ],
)
async def test_blocked_expressions_are_named(
    world: World, expression: str, code: str, text: str
) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http, {"variants": [variant("replication", expression)]})
        cand = cands[0]
        assert cand["validationOutcome"] == "blocked"
        found = [c for c in cand["validationDetail"]["checks"] if c["code"] == code]
        assert found and text in found[0]["message"]
        # A blocked candidate cannot be saved.
        refused = await http.post(f"{B}/candidates/{cand['id']}/save")
        assert (
            refused.status_code == 409 and refused.json()["detail"]["code"] == "candidate_blocked"
        )


async def test_vector_field_through_a_vector_operator_is_fine(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(
            world, http, {"variants": [variant("replication", "rank(vec_avg(anl_vec))")]}
        )
        assert cands[0]["validationOutcome"] == "valid"


async def test_grouping_field_in_a_group_operator_is_fine(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(
            world,
            http,
            {"variants": [variant("replication", "group_neutralize(rank(cap), industry)")]},
        )
        detail = cands[0]["validationDetail"]
        assert cands[0]["validationOutcome"] == "needs_review"  # catalogue lists no industry field
        assert detail["groupingUsed"] == ["industry"]
        assert any(c["code"] == "grouping_not_in_catalogue" for c in detail["checks"])


async def test_needs_review_for_timing_and_complexity(world: World) -> None:
    http = await world.boot()
    async with http:
        world.queries.data[("USA", 1, "TOP3000")]["restated"] = [
            fx.row("eps_restated", "Latest restated earnings per share")
        ]
        long = "rank(" + " + ".join(["ts_mean(eps_restated, 5)"] * 14) + ")"
        idea_id = await seed_idea(world, lookahead_concerns=["Announcement lag is not stated."])
        d = await design(http, world, idea_id)
        world.llm.replies.append({"variants": [variant("replication", long)]})
        started = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={"settings": SETTINGS, "settingsConfirmed": True, "modelRef": REF},
        )
        await done(http, started.json()["jobId"])
        cand = (await http.get(f"{B}/ideas/{idea_id}/candidates")).json()[0]
        codes = {c["code"] for c in cand["validationDetail"]["checks"]}
        assert cand["validationOutcome"] == "needs_review"
        assert {"revised_data", "complexity", "idea_lookahead"} <= codes


async def test_unsupported_settings_are_blocked_and_nothing_is_defaulted(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        d = await design(http, world, idea_id)
        wrong = {**SETTINGS, "neutralization": "CROWDING"}
        refused = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={"settings": wrong, "settingsConfirmed": True, "modelRef": REF},
        )
        assert refused.status_code == 422 and "CROWDING" in refused.json()["detail"]["message"]
        no_neutralization = {k: v for k, v in SETTINGS.items() if k != "neutralization"}
        refused = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={"settings": no_neutralization, "settingsConfirmed": True, "modelRef": REF},
        )
        assert (
            refused.status_code == 422 and "never filled in" in refused.json()["detail"]["message"]
        )
        unconfirmed = await http.post(
            f"{B}/designs/{d['id']}/variants", json={"settings": SETTINGS, "modelRef": REF}
        )
        assert unconfirmed.json()["detail"]["code"] == "settings_unconfirmed"
        language = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={
                "settings": {**SETTINGS, "language": "PYTHON"},
                "settingsConfirmed": True,
                "modelRef": REF,
            },
        )
        assert language.status_code == 422
        both = await http.post(
            f"{B}/designs/{d['id']}/variants",
            json={
                "settings": {**SETTINGS, "maxTrade": "ON", "maxPosition": "ON"},
                "settingsConfirmed": True,
                "modelRef": REF,
            },
        )
        assert "cannot both be On" in both.json()["detail"]["message"]


# --- scope ---------------------------------------------------------------------------------------


async def test_usa_top3000_d1_is_kept_end_to_end_and_never_becomes_glb(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, cands = await make(world, http)
        for c in cands:
            assert c["scope"] == SCOPE
            assert {k: c["settings"][k] for k in SCOPE} == SCOPE
        saved = [(await http.post(f"{B}/candidates/{c['id']}/save")).json() for c in cands]
        prev = (
            await http.post(f"{B}/preview", json={"candidateIds": [c["id"] for c in saved]})
        ).json()
        assert {
            (i["settings"]["region"], i["settings"]["universe"], i["settings"]["delay"])
            for i in prev["items"]
        } == {("USA", "TOP3000", 1)}
        task = (
            await http.post(
                f"{B}/tasks",
                json={
                    "candidateIds": [c["id"] for c in saved],
                    "idempotencyKey": "key-scope-0001",
                    "confirm": True,
                },
            )
        ).json()
        async with world.states[-1].db.session() as session:
            study = await session.get(Study, task["taskId"])
            trials = (
                await session.scalars(select(Trial).where(Trial.study_id == task["taskId"]))
            ).all()
        assert study is not None and study.sampler_params["region"] == "USA"
        assert len(trials) == 2
        for t in trials:
            assert (
                t.settings["region"],
                t.settings["universe"],
                t.settings["delay"],
                t.settings["instrumentType"],
            ) == ("USA", "TOP3000", 1, "EQUITY")
        assert idea_id


async def test_unsynced_scope_is_blocked_with_a_reason_not_replaced(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        glb = {"instrumentType": "EQUITY", "region": "GLB", "delay": 1, "universe": "TOP3000"}
        refused = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": glb, "modelRef": REF}
        )
        assert refused.status_code == 409
        body = refused.json()["detail"]
        assert (
            body["code"] == "scope_not_synced"
            and "GLB" in body["message"]
            and "never switches" in body["message"]
        )
        assert not world.llm.calls  # no model request was spent
        # A scope the schema does not offer is refused for that reason.
        world.queries.data[("USA", 1, "TOP5000")] = rows()
        odd = {"instrumentType": "EQUITY", "region": "USA", "delay": 1, "universe": "TOP5000"}
        refused = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": odd, "modelRef": REF}
        )
        assert refused.json()["detail"]["code"] == "scope_unavailable"


async def test_settings_that_disagree_with_the_scope_are_blocked(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, cands = await make(world, http)
        edit = await http.post(
            f"{B}/candidates/{cands[0]['id']}/check",
            json={"settings": {**SETTINGS, "region": "GLB", "universe": "TOP3000"}},
        )
        codes = {c["code"] for c in edit.json()["checks"]}
        assert "scope_mismatch" in codes and edit.json()["outcome"] == "blocked"
        assert idea_id


async def test_scope_not_in_catalogue_blocks_validation(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        del world.queries.data[("USA", 1, "TOP3000")]
        again = await http.post(f"{B}/candidates/{cands[0]['id']}/revalidate")
        assert again.json()["validationOutcome"] == "blocked"
        assert any(
            c["code"] == "scope_not_synced" for c in again.json()["validationDetail"]["checks"]
        )


async def test_missing_metadata_blocks_instead_of_passing(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        world.operators, world.schema = None, None
        again = (await http.post(f"{B}/candidates/{cands[0]['id']}/revalidate")).json()
        codes = {c["code"] for c in again["validationDetail"]["checks"]}
        assert again["validationOutcome"] == "blocked"
        assert {"operators_unread", "no_settings_metadata"} <= codes


# --- the model ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "needle"),
    [
        ("this is not json", "did not return a JSON object"),
        ({"variants": []}, "does not match the required structure"),
        (
            {
                "variants": [
                    variant("replication", "rank(cap)", relation_to_source="A proven strategy.")
                ]
            },
            "claimed performance",
        ),
        ({"variants": [variant("direction", "rank(-cap)")]}, "without a justification"),
        ({"variants": [variant("replication", "rank(cap)", surprise="x")]}, "does not match"),
    ],
)
async def test_invalid_model_answers_save_nothing(world: World, reply: Any, needle: str) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        d = await design(http, world, idea_id)
        job = await generate(http, world, d["id"], reply)
        assert job["status"] == "failed" and job["errorCategory"] == "invalid_model_output"
        assert needle in job["errorMessage"]
        assert job["errorDetail"]
        assert (await http.get(f"{B}/ideas/{idea_id}/candidates")).json() == []


async def test_design_with_a_performance_claim_is_refused(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id = await seed_idea(world)
        world.llm.replies.append(
            {**DESIGN, "signal_construction": "Has a Sharpe of 2.1 in backtests."}
        )
        started = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": SCOPE, "modelRef": REF}
        )
        job = await done(http, started.json()["jobId"])
        assert job["status"] == "failed" and "claimed performance" in job["errorMessage"]
        assert (await http.get(f"{B}/ideas/{idea_id}/designs")).json() == []


async def test_direction_variant_with_justification_and_a_variant_cap(world: World) -> None:
    http = await world.boot()
    async with http:
        flip = variant(
            "direction",
            "rank(-bookvalue_ps / cap)",
            direction_justification="Distress risk compensation.",
        )
        _, cands = await make(
            world, http, {"variants": [variant("replication", "rank(bookvalue_ps / cap)"), flip]}
        )
        assert [c["variantKind"] for c in cands] == ["replication", "direction"]
        assert cands[1]["explanation"]["directionJustification"] == "Distress risk compensation."
        over = await http.post(
            f"{B}/designs/{cands[0]['designId']}/variants",
            json={"settings": SETTINGS, "settingsConfirmed": True, "count": 99},
        )
        assert over.status_code == 422  # the cap is enforced, not clamped silently


async def test_model_failure_is_reported_and_retry_works(world: World) -> None:
    http = await world.boot()
    async with http:
        from alpha_harness.llm.keys import LLMError

        idea_id = await seed_idea(world)
        world.llm.replies.append(LLMError("provider is down"))
        started = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": SCOPE, "modelRef": REF}
        )
        job = await done(http, started.json()["jobId"])
        assert job["status"] == "failed" and job["errorCategory"] == "llm_error"
        d = await design(http, world, idea_id)  # the retry
        assert d["id"]


async def test_a_running_job_can_be_cancelled(world: World) -> None:
    http = await world.boot()
    async with http:
        world.llm.delay = 5
        idea_id = await seed_idea(world)
        world.llm.replies.append(DESIGN)
        started = await http.post(
            f"{B}/ideas/{idea_id}/design", json={"scope": SCOPE, "modelRef": REF}
        )
        job_id = started.json()["jobId"]
        await asyncio.sleep(0.1)
        assert (await http.post(f"/api/research/jobs/{job_id}/cancel")).status_code in {200, 202}
        assert (await done(http, job_id))["status"] == "cancelled"
        assert (await http.get(f"{B}/ideas/{idea_id}/designs")).json() == []


# --- versions -------------------------------------------------------------------------------------


async def test_edits_make_new_versions_and_the_original_survives(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, cands = await make(world, http)
        base = cands[0]
        edited = await http.post(
            f"{B}/candidates/{base['id']}/revise", json={"expression": "rank(bookvalue_ps)"}
        )
        assert edited.status_code == 201
        new = edited.json()
        assert (
            new["version"] == 3
            and new["parentId"] == base["id"]
            and new["variantKind"] == "manual_edit"
        )
        assert new["changes"] == ["Expression edited by hand."]
        same = await http.get(f"{B}/candidates/{base['id']}")
        assert same.json()["expression"] == "rank(bookvalue_ps / cap)"
        # Spacing is not a change.
        dup = await http.post(
            f"{B}/candidates/{base['id']}/revise", json={"expression": "rank( bookvalue_ps/cap )"}
        )
        assert dup.status_code == 409 and dup.json()["detail"]["code"] == "unchanged"
        # An edit is checked: a bad one is stored as blocked, visibly.
        bad = await http.post(
            f"{B}/candidates/{base['id']}/revise", json={"expression": "rank(nope)"}
        )
        assert bad.json()["validationOutcome"] == "blocked"
        assert len((await http.get(f"{B}/ideas/{idea_id}/candidates")).json()) == 4


async def test_regenerating_keeps_the_earlier_candidates(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, first = await make(world, http)
        d = (await http.get(f"{B}/ideas/{idea_id}/designs")).json()[0]
        await generate(http, world, d["id"], {"variants": [variant("replication", "rank(cap)")]})
        after = (await http.get(f"{B}/ideas/{idea_id}/candidates")).json()
        assert [c["id"] for c in after[:2]] == [c["id"] for c in first]
        assert len(after) == 3 and [c["version"] for c in after] == [1, 2, 3]


# --- preview and task -----------------------------------------------------------------------------


async def sim_count(w: World) -> int:
    async with w.states[-1].db.session() as session:
        return (
            await session.execute(select(func.count()).select_from(SimulationRecord))
        ).scalar_one()


async def trial_count(w: World) -> int:
    async with w.states[-1].db.session() as session:
        return (await session.execute(select(func.count()).select_from(Trial))).scalar_one()


async def studies(w: World) -> int:
    async with w.states[-1].db.session() as session:
        return (await session.execute(select(func.count()).select_from(Study))).scalar_one()


async def test_every_step_before_the_task_creates_nothing_to_run(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        ids = [c["id"] for c in cands]
        for c in cands:
            await http.post(f"{B}/candidates/{c['id']}/revalidate")
            await http.post(f"{B}/candidates/{c['id']}/save")
        for _ in range(3):
            prev = await http.post(f"{B}/preview", json={"candidateIds": ids})
            assert prev.status_code == 200
        body = prev.json()
        assert body["simulations"] == 2 and body["starts"] is False and body["problems"] == []
        assert "start" in body["quota"]
        assert (
            await studies(world) == 0
            and await trial_count(world) == 0
            and await sim_count(world) == 0
        )


async def test_preview_and_task_need_saved_candidates_and_explicit_confirmation(
    world: World,
) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        ids = [cands[0]["id"]]
        draft = (await http.post(f"{B}/preview", json={"candidateIds": ids})).json()
        assert draft["simulations"] == 0 and "Save the candidate first" in draft["problems"][0]
        refused = await http.post(
            f"{B}/tasks",
            json={"candidateIds": ids, "idempotencyKey": "key-draft-001", "confirm": True},
        )
        assert refused.status_code == 409 and refused.json()["detail"]["code"] == "preview_blocked"
        await http.post(f"{B}/candidates/{ids[0]}/save")
        unconfirmed = await http.post(
            f"{B}/tasks", json={"candidateIds": ids, "idempotencyKey": "key-draft-002"}
        )
        assert (
            unconfirmed.status_code == 422
            and unconfirmed.json()["detail"]["code"] == "not_confirmed"
        )
        assert await studies(world) == 0


async def test_task_is_idle_parked_and_repeated_clicks_make_one(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        ids = [c["id"] for c in cands]
        for i in ids:
            await http.post(f"{B}/candidates/{i}/save")
        body = {"candidateIds": ids, "idempotencyKey": "click-0001-abc", "confirm": True}
        first, second = await asyncio.gather(
            http.post(f"{B}/tasks", json=body), http.post(f"{B}/tasks", json=body)
        )
        assert first.status_code == second.status_code == 201
        a, b = first.json(), second.json()
        assert a["taskId"] == b["taskId"] and sorted([a["existing"], b["existing"]]) == [
            False,
            True,
        ]
        assert a["started"] is False
        assert await studies(world) == 1
        async with world.states[-1].db.session() as session:
            study = await session.get(Study, a["taskId"])
            trials = (
                await session.scalars(select(Trial).where(Trial.study_id == a["taskId"]))
            ).all()
        assert study is not None and study.status == "IDLE"
        assert study.sampler == PASTE_ALPHAS and study.template_name.startswith("Alpha Builder")
        assert "Alpha Expression:" in study.template_source
        # Parked, unsent, linked to their candidates.
        assert {t.state for t in trials} == {"PRUNED"} and all(t.alpha_id is None for t in trials)
        assert {t.params["candidateId"] for t in trials} == set(ids)
        assert await sim_count(world) == 0
        # A different key for the same candidates is refused: no duplicate task.
        again = await http.post(f"{B}/tasks", json=body | {"idempotencyKey": "click-0002-abc"})
        assert again.status_code == 409 and "already made" in again.json()["detail"]["message"]
        assert await studies(world) == 1
        # The task appears in the existing Tasks list.
        listed = await http.get("/api/lab-tasks")
        shown = next(t for t in listed.json()["tasks"] if t["id"] == a["taskId"])
        # Waiting for the person's own Run: the existing lab-tasks run action is the only
        # thing that queues it (not exercised here: it would hand work to the engine).
        assert shown["status"] == "IDLE"


async def test_review_must_be_acknowledged(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(
            world,
            http,
            {"variants": [variant("replication", "group_neutralize(rank(cap), industry)")]},
        )
        assert cands[0]["validationOutcome"] == "needs_review"
        await http.post(f"{B}/candidates/{cands[0]['id']}/save")
        body = {
            "candidateIds": [cands[0]["id"]],
            "idempotencyKey": "review-key-01",
            "confirm": True,
        }
        refused = await http.post(f"{B}/tasks", json=body)
        assert (
            refused.status_code == 409
            and refused.json()["detail"]["code"] == "review_not_acknowledged"
        )
        assert (
            await http.post(f"{B}/tasks", json=body | {"acknowledgeReview": True})
        ).status_code == 201


async def test_blocked_candidate_in_a_set_blocks_the_task(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        await http.post(f"{B}/candidates/{cands[0]['id']}/save")
        # The catalogue changes under it: the task preview re-checks, not the stored reading.
        del world.queries.data[("USA", 1, "TOP3000")]["market capitalization"]
        world.queries.data[("USA", 1, "TOP3000")]["unrelated"] = [fx.row("nothing", "x")]
        prev = (await http.post(f"{B}/preview", json={"candidateIds": [cands[0]["id"]]})).json()
        assert prev["simulations"] == 0 and prev["problems"]
        await http.post(f"{B}/candidates/{cands[1]['id']}/save")


# --- results --------------------------------------------------------------------------------------


async def test_real_metrics_attach_to_the_right_candidate_and_survive_restart(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, cands = await make(world, http)
        ids = [c["id"] for c in cands]
        for i in ids:
            await http.post(f"{B}/candidates/{i}/save")
        before = (await http.get(f"{B}/candidates/{ids[0]}/results")).json()
        assert before["state"] == "no_task" and before["metrics"] is None
        task = (
            await http.post(
                f"{B}/tasks",
                json={"candidateIds": ids, "idempotencyKey": "results-key-1", "confirm": True},
            )
        ).json()
        waiting = (await http.get(f"{B}/candidates/{ids[0]}/results")).json()
        assert (
            waiting["state"] == "not_started"
            and waiting["metrics"] is None
            and waiting["diagnostics"] == []
        )
        # The engine would fill these in; here BRAIN's answers are simulated as stored rows.
        async with world.states[-1].db.session() as session:
            trials = (
                await session.scalars(select(Trial).where(Trial.study_id == task["taskId"]))
            ).all()
            by_candidate = {t.params["candidateId"]: t for t in trials}
            by_candidate[ids[0]].alpha_id = "ALPHA_A"
            by_candidate[ids[1]].alpha_id = "ALPHA_B"
            by_candidate[ids[0]].state = "COMPLETE"
            by_candidate[ids[1]].state = "COMPLETE"
        world.alphas["ALPHA_A"] = {
            "sharpe": 1.8,
            "fitness": 1.2,
            "turnover": 0.2,
            "returns": 0.1,
            "drawdown": 0.05,
            "margin": 0.003,
            "checks": '[{"name":"LOW_SHARPE","result":"PASS","limit":1.25,"value":1.8}]',
        }
        world.alphas["ALPHA_B"] = {
            "sharpe": 0.4,
            "fitness": 0.2,
            "turnover": 0.9,
            "test_sharpe": 0.1,
            "checks": '[{"name":"HIGH_TURNOVER","result":"FAIL","limit":0.7,"value":0.9}]',
        }
        a = (await http.get(f"{B}/candidates/{ids[0]}/results")).json()
        b = (await http.get(f"{B}/candidates/{ids[1]}/results")).json()
        assert a["state"] == b["state"] == "complete" and a["source"] == "BRAIN"
        assert a["metrics"]["sharpe"] == 1.8 and b["metrics"]["sharpe"] == 0.4
        assert (
            a["metrics"]["margin"] == 0.003 and b["metrics"]["margin"] is None
        )  # not reported: not invented
        assert a["settingsMismatch"] == [] and a["diagnostics"] == []
        sources = {d["source"] for d in b["diagnostics"]}
        assert {"BRAIN check", "Sharpe", "Turnover", "Out-of-sample"} <= sources
        assert a["settings"]["neutralization"] == "SUBINDUSTRY"
        lineage = (await http.get(f"{B}/candidates/{ids[0]}/lineage")).json()
        assert {t["candidateId"] for t in lineage["tested"]} == set(ids)
        assert "not evidence of improvement" in lineage["note"]
        await http.aclose()
    # A fresh process on the same files.
    http2 = await world.boot()
    async with http2:
        again = (await http2.get(f"{B}/candidates/{ids[1]}/results")).json()
        assert again["metrics"]["sharpe"] == 0.4 and again["taskId"] == task["taskId"]
        listed = (await http2.get(f"{B}/ideas/{idea_id}/candidates")).json()
        assert [c["status"] for c in listed] == ["tasked", "tasked"]
        assert (await http2.get(f"{B}/ideas/{idea_id}/designs")).json()[0]["scope"] == SCOPE


async def test_refinement_needs_real_results_and_makes_new_drafts(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        ids = [c["id"] for c in cands]
        refused = await http.post(f"{B}/candidates/{ids[0]}/refine", json={"modelRef": REF})
        assert refused.status_code == 409 and refused.json()["detail"]["code"] == "not_simulated"
        for i in ids:
            await http.post(f"{B}/candidates/{i}/save")
        task = (
            await http.post(
                f"{B}/tasks",
                json={"candidateIds": ids, "idempotencyKey": "refine-key-01", "confirm": True},
            )
        ).json()
        async with world.states[-1].db.session() as session:
            trials = (
                await session.scalars(select(Trial).where(Trial.study_id == task["taskId"]))
            ).all()
            for t in trials:
                t.alpha_id = f"A_{t.number}"
        for n in range(2):
            world.alphas[f"A_{n}"] = {"sharpe": 0.5, "turnover": 0.8}
        proposal = {
            "kind": "neutralization",
            "name": "Market neutral",
            "expression": "rank(bookvalue_ps / cap)",
            "settings_changes": {"neutralization": "MARKET"},
            "rationale": "Sector exposure may dominate.",
            "economic_justification": "Value differs across sectors for structural reasons.",
            "overfitting_risk": "Low: one setting changes.",
            "changes_from_parent": [],
        }
        world.llm.replies.append({"commentary": "Turnover is high.", "proposals": [proposal]})
        started = await http.post(f"{B}/candidates/{ids[0]}/refine", json={"modelRef": REF})
        job = await done(http, started.json()["jobId"])
        assert job["status"] == "completed" and job["result"]["commentarySource"] == "LLM"
        new = (await http.get(f"{B}/candidates/{job['result']['candidateIds'][0]}")).json()
        assert (
            new["status"] == "draft"
            and new["variantKind"] == "refinement"
            and new["parentId"] == ids[0]
        )
        assert new["settings"]["neutralization"] == "MARKET" and new["studyId"] is None
        assert any("optimistic" in x for x in new["limitations"])
        old = (await http.get(f"{B}/candidates/{ids[0]}")).json()
        assert old["settings"]["neutralization"] == "SUBINDUSTRY"  # the tested one is untouched
        prompt = world.llm.calls[-1]["user"]
        assert "0.5" in prompt and "null means BRAIN did not report it" in prompt


async def test_export_carries_provenance(world: World) -> None:
    http = await world.boot()
    async with http:
        _, cands = await make(world, http)
        out = (await http.get(f"{B}/candidates/{cands[0]['id']}/export")).json()
        assert out["candidate"]["expression"] == "rank(bookvalue_ps / cap)"
        assert out["idea"]["hypothesis"] == "Cheap stocks outperform."
        assert out["results"]["state"] == "no_task"


# --- isolation ------------------------------------------------------------------------------------


def test_alpha_builder_never_touches_spc() -> None:
    root = Path(__file__).parents[1] / "src" / "alpha_harness"
    for path in [*(root / "alpha_builder").glob("*.py"), root / "api" / "alpha_builder.py"]:
        text = path.read_text()
        assert "spc" not in text.lower().replace("specific", ""), path
        assert "SpcService" not in text, path


async def test_options_come_from_brain_metadata(world: World) -> None:
    http = await world.boot()
    async with http:
        got = (await http.get(f"{B}/options")).json()
        assert got["settingsSchema"]["neutralization"]
        assert {
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "instrumentType": "EQUITY",
        } in got["syncedScopes"]
        assert got["maxVariants"] == 7


async def test_candidates_survive_idea_listing_through_the_research_api(world: World) -> None:
    http = await world.boot()
    async with http:
        idea_id, cands = await make(world, http)
        detail = (await http.get(f"/api/research/ideas/{idea_id}")).json()
        assert {c["id"] for c in detail["candidates"]} == {c["id"] for c in cands}
        async with world.states[-1].db.session() as session:
            assert (
                await session.execute(select(func.count()).select_from(AlphaCandidate))
            ).scalar_one() == 2
