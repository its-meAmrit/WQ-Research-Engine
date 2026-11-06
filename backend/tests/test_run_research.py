from typing import TYPE_CHECKING, Any

import httpx2

from alpha_harness.db.sqlite import Database
from alpha_harness.spc.execution import SpcRunner
from alpha_harness.spc.research import ResearchFailedError
from alpha_harness.spc.service import SpcService
from alpha_harness.spc.views import PromptCreate, RunRequest

from .conftest import GOOD, REF, FakeLLM, settle

if TYPE_CHECKING:
    from pathlib import Path


class ScriptedSearch:
    name = "scripted"

    def __init__(self, results: list[dict[str, Any]] | Exception) -> None:
        self.results = results
        self.queries: list[str] = []

    async def search(
        self, client: httpx2.AsyncClient, query: str, *, count: int, freshness_days: int
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


def build(db: Database, llm: FakeLLM, tmp_path: Path, provider: ScriptedSearch) -> SpcService:
    runner = SpcRunner(db, llm, tmp_path, None, provider=lambda: provider)  # pyright: ignore[reportArgumentType]
    return SpcService(db, llm, tmp_path, None, runner)  # pyright: ignore[reportArgumentType]


async def prompt(service: SpcService) -> int:
    made = await service.create_prompt(
        PromptCreate(
            name="p",
            prompt_text="Pick equities.",
            preferred_model=REF,
            settings={"sector": "semiconductors", "universe": "Global", "thesis": "AI capex"},
        )
    )
    return made.id


async def test_evidence_reaches_the_model_and_is_saved_with_its_dates(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    search = ScriptedSearch(
        [
            {
                "url": "https://investor.acme.com/q3",
                "title": "Acme Q3",
                "snippet": "Beat.",
                "published": "2026-11-01T00:00:00Z",
            },
            {
                "url": "https://old.example.com/a",
                "title": "Old",
                "snippet": "x",
                "published": "2024-01-01T00:00:00Z",
            },
            {
                "url": "https://nodate.example.com/a",
                "title": "No date",
                "snippet": "x",
                "published": None,
            },
            {"url": "http://10.0.0.1/admin", "title": "Trap", "snippet": "x", "published": None},
        ]
    )
    service = build(db, llm, tmp_path, search)
    llm.replies = [GOOD]
    run = await settle(service, (await service.start_run(await prompt(service), RunRequest())).id)

    assert run.status == "completed"
    assert run.research and run.research["status"] == "done"
    assert (run.research["sources"], run.research["stale"], run.research["undated"]) == (3, 1, 1)
    by_title = {e.title: e for e in run.evidence}
    assert set(by_title) == {"Acme Q3", "Old", "No date"}
    assert by_title["Acme Q3"].published_at == "2026-11-01"
    assert by_title["Acme Q3"].source_type == "investor_relations"
    assert by_title["Old"].stale is True
    assert by_title["No date"].published_at is None
    assert all(e.retrieved_at for e in run.evidence)

    sent = llm.calls[0]["user"]
    assert sent.startswith("Pick equities.")
    assert "RESEARCH EVIDENCE" in sent
    assert "Acme Q3" in sent
    assert "undated" in sent
    assert "10.0.0.1" not in sent
    # The saved snapshot is the prompt alone, not the prompt plus today's evidence.
    assert run.prompt_snapshot == "Pick equities."
    assert any("semiconductors" in q for q in search.queries)


async def test_a_failed_search_degrades_honestly_unless_required(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    search = ScriptedSearch(ResearchFailedError("Scripted refused the search key."))
    service = build(db, llm, tmp_path, search)
    llm.replies = [GOOD]
    pid = await prompt(service)
    auto = await settle(service, (await service.start_run(pid, RunRequest(research="auto"))).id)
    assert auto.status == "completed"
    assert auto.research is not None
    assert auto.research["status"] == "failed"
    assert "refused the search key" in auto.research["error"]
    assert auto.evidence == []
    assert "RESEARCH EVIDENCE" not in llm.calls[0]["user"]

    strict = await settle(
        service, (await service.start_run(pid, RunRequest(research="required"))).id
    )
    assert strict.status == "failed"
    assert strict.error_code == "research_failed"
    assert len(llm.calls) == 1  # the model was never asked

    export = await service.export(pid, auto.id)
    assert next(c for c in export.checks if c.id == "research").status == "warn"


async def test_successful_research_passes_the_export_check(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    search = ScriptedSearch(
        [
            {
                "url": "https://a.example.com/x",
                "title": "A",
                "snippet": "s",
                "published": "2026-11-02",
            }
        ]
    )
    service = build(db, llm, tmp_path, search)
    llm.replies = [GOOD]
    pid = await prompt(service)
    await settle(service, (await service.start_run(pid, RunRequest())).id)
    check = next(c for c in (await service.export(pid)).checks if c.id == "research")
    assert check.status == "pass"
    assert "1 sources from scripted" in check.detail
