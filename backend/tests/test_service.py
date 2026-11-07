import asyncio
import json
from pathlib import Path

import pytest

from alpha_harness.db.models import SpcRun, SpcRunStatus, utcnow
from alpha_harness.db.sqlite import Database
from alpha_harness.llm.keys import BudgetExhaustedError, LLMUnavailable
from alpha_harness.spc.errors import SpcConflictError, SpcInvalidError, SpcNotFoundError
from alpha_harness.spc.execution import SpcRunner
from alpha_harness.spc.service import SpcService
from alpha_harness.spc.views import PromptCreate, PromptPatch, RunRequest

from .conftest import GOOD, REF, FakeLLM, position, settle


def _budget_error() -> BudgetExhaustedError:
    """The real exception type, without rebuilding the provider budget it normally carries."""
    exc = BudgetExhaustedError.__new__(BudgetExhaustedError)
    Exception.__init__(exc, "Every key has used its daily allowance of test-model.")
    return exc


PROMPT = "Pick global semiconductor equities. Output only the JSON array."


def make(**over: object) -> PromptCreate:
    return PromptCreate(name="Semis", prompt_text=PROMPT, weight=0.25, preferred_model=REF, **over)  # pyright: ignore[reportArgumentType]


# -- prompts and versions ------------------------------------------------------------------


async def test_create_get_list(service: SpcService) -> None:
    made = await service.create_prompt(make())
    assert made.latest_version == 1
    assert made.weight == 0.25
    got = await service.get_prompt(made.id)
    assert got.prompt_text == PROMPT
    assert [p.id for p in await service.list_prompts()] == [made.id]


async def test_edit_text_adds_a_version_and_leaves_the_old_one(service: SpcService) -> None:
    made = await service.create_prompt(make())
    edited = await service.update_prompt(made.id, PromptPatch(prompt_text=PROMPT + " More."))
    assert edited.latest_version == 2
    texts = {v.number: v.prompt_text for v in edited.versions}
    assert texts == {1: PROMPT, 2: PROMPT + " More."}


async def test_unchanged_text_does_not_add_a_version(service: SpcService) -> None:
    made = await service.create_prompt(make())
    again = await service.update_prompt(made.id, PromptPatch(prompt_text=PROMPT, name="Renamed"))
    assert again.latest_version == 1
    assert again.name == "Renamed"


@pytest.mark.parametrize("weight", [0.001, 0.125, 1.01])
async def test_weight_precision_is_enforced(service: SpcService, weight: float) -> None:
    made = await service.create_prompt(make())
    with pytest.raises((SpcInvalidError, ValueError)):
        await service.update_prompt(made.id, PromptPatch(weight=weight))


@pytest.mark.parametrize(("weight", "stored"), [(0, 0.0), (0.07, 0.07), (0.29, 0.29), (1, 1.0)])
async def test_weight_round_trips_without_float_drift(
    service: SpcService, weight: float, stored: float
) -> None:
    made = await service.create_prompt(make())
    assert (await service.update_prompt(made.id, PromptPatch(weight=weight))).weight == stored


async def test_prompt_length_limit(service: SpcService) -> None:
    with pytest.raises(SpcInvalidError, match="10,000"):
        await service.create_prompt(
            PromptCreate(name="x", prompt_text="a" * 10_001, preferred_model=REF)
        )
    ok = await service.create_prompt(PromptCreate(name="x", prompt_text="a" * 10_000))
    assert ok.characters == 10_000


async def test_model_version_limit(service: SpcService) -> None:
    with pytest.raises(SpcInvalidError, match="100"):
        await service.create_prompt(
            PromptCreate(name="x", prompt_text="p", model_version="v" * 101)
        )


async def test_unknown_model_is_refused(service: SpcService) -> None:
    with pytest.raises(Exception, match="not set up"):
        await service.create_prompt(
            PromptCreate(name="x", prompt_text="p", preferred_model="nope/x")
        )


async def test_duplicate_has_own_history_and_zero_weight(service: SpcService) -> None:
    made = await service.create_prompt(make())
    await service.update_prompt(made.id, PromptPatch(prompt_text=PROMPT + " v2"))
    copy = await service.duplicate_prompt(made.id)
    assert copy.id != made.id
    assert copy.weight == 0
    assert copy.latest_version == 1
    assert copy.prompt_text == PROMPT + " v2"
    await service.update_prompt(copy.id, PromptPatch(prompt_text="changed"))
    assert (await service.get_prompt(made.id)).latest_version == 2


async def test_restore_adds_a_version(service: SpcService) -> None:
    made = await service.create_prompt(make())
    await service.update_prompt(made.id, PromptPatch(prompt_text="second"))
    restored = await service.restore_version(made.id, 1)
    assert restored.latest_version == 3
    assert restored.prompt_text == PROMPT
    with pytest.raises(SpcNotFoundError):
        await service.restore_version(made.id, 9)


async def test_archive_and_prefer(service: SpcService) -> None:
    made = await service.create_prompt(make())
    v1 = made.versions[0].id
    got = await service.update_prompt(made.id, PromptPatch(preferred_version_id=v1, archived=True))
    assert got.archived
    assert got.preferred_version_id == v1
    assert await service.list_prompts() == []
    assert len(await service.list_prompts(archived=True)) == 1
    other = await service.create_prompt(make())
    with pytest.raises(SpcInvalidError):
        await service.update_prompt(other.id, PromptPatch(preferred_version_id=v1))


async def test_compare_versions(service: SpcService) -> None:
    made = await service.create_prompt(make())
    await service.update_prompt(made.id, PromptPatch(prompt_text="line one\nline two"))
    await service.update_prompt(made.id, PromptPatch(prompt_text="line one\nline 2\nline three"))
    cmp = await service.compare(made.id, 2, 3)
    assert cmp.added == 2
    assert cmp.removed == 1
    assert {ln.kind for ln in cmp.lines} == {"add", "del", "ctx"}


# -- runs ---------------------------------------------------------------------------------


async def test_run_completes_and_saves_exact_output(service: SpcService, llm: FakeLLM) -> None:
    raw = json.dumps(GOOD, indent=2)
    llm.replies = [raw]
    made = await service.create_prompt(make())
    started = await service.start_run(made.id, RunRequest(research="off"))
    assert started.status in {"queued", "researching", "generating"}
    done = await settle(service, started.id)
    assert done.status == "completed"
    assert done.valid is True
    assert done.raw_output == raw
    assert done.sample_json == raw
    assert [p.ticker for p in done.portfolio] == ["AAPL", "MSFT", "BAYN"]
    assert done.usage and done.usage["totalTokens"] == 30
    assert done.finished_at and done.started_at
    # The model was sent the saved prompt text and nothing was forced into a JSON object.
    assert llm.calls[0]["user"] == PROMPT
    assert llm.calls[0]["plain"] is True
    assert llm.calls[0]["response_schema"] is None


async def test_run_snapshot_survives_later_edits(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    done = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    await service.update_prompt(
        made.id, PromptPatch(prompt_text="rewritten", name="New name", weight=0.9)
    )
    again = await service.get_run(done.id)
    assert again.prompt_snapshot == PROMPT
    assert again.meta["name"] == "Semis"
    assert again.meta["weight_hundredths"] == 25


async def test_invalid_output_fails_with_errors_and_keeps_raw(
    service: SpcService, llm: FakeLLM
) -> None:
    bad = [position(isin="US0378331006"), position(isin="US5949181045", confidence_score=3)]
    llm.replies = [bad]
    made = await service.create_prompt(make())
    done = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    assert done.status == "failed"
    assert done.error_code == "invalid_output"
    assert done.valid is False
    assert done.raw_output == json.dumps(bad)
    assert done.portfolio == []
    assert done.validation and done.validation["errorCount"] == 2
    assert {i["code"] for i in done.validation["issues"]} == {"invalid_isin", "out_of_range"}


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ("", "empty_response"),
        ("   ", "empty_response"),
        (_budget_error(), "budget_exhausted"),
        (LLMUnavailable("The model did not answer in time."), "provider_timeout"),
        (LLMUnavailable("Provider is down."), "provider_unavailable"),
        (RuntimeError("boom"), "internal_error"),
    ],
)
async def test_failures_have_precise_codes(
    service: SpcService, llm: FakeLLM, reply: object, code: str
) -> None:
    llm.replies = [reply]
    made = await service.create_prompt(make())
    done = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    assert done.status == "failed"
    assert done.error_code == code
    assert done.error_message
    assert done.finished_at is not None


async def test_missing_key_is_refused_before_a_run_exists(
    service: SpcService, llm: FakeLLM
) -> None:
    llm.keyed = False
    made = await service.create_prompt(make())
    with pytest.raises(Exception, match=r"(?i)key"):
        await service.start_run(made.id, RunRequest(research="off"))
    assert await service.list_runs() == []


async def test_rapid_double_click_makes_one_run(service: SpcService, llm: FakeLLM) -> None:
    llm.delay = 0.2
    llm.replies = [GOOD, GOOD]
    made = await service.create_prompt(make())
    first = await service.start_run(made.id, RunRequest(research="off"))
    with pytest.raises(SpcConflictError, match="already in progress"):
        await service.start_run(made.id, RunRequest(research="off"))
    await settle(service, first.id)
    assert len(await service.list_runs()) == 1


async def test_cancel_leaves_a_clean_record(service: SpcService, llm: FakeLLM) -> None:
    llm.delay = 5
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    started = await service.start_run(made.id, RunRequest(research="off"))
    await asyncio.sleep(0.05)
    cancelled = await service.cancel_run(started.id)
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert cancelled.portfolio == []
    assert not service.runner.active(started.id)
    with pytest.raises(SpcConflictError):
        await service.cancel_run(started.id)


async def test_run_needs_research_when_required_but_unconfigured(service: SpcService) -> None:
    made = await service.create_prompt(make())
    done = await settle(
        service, (await service.start_run(made.id, RunRequest(research="required"))).id
    )
    assert done.status == "failed"
    assert done.error_code == "research_not_configured"
    assert "ALPHA_HARNESS_SPC_SEARCH" in (done.error_message or "")


async def test_auto_research_runs_without_it_and_says_so(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    done = await settle(service, (await service.start_run(made.id, RunRequest(research="auto"))).id)
    assert done.status == "completed"
    assert done.research and done.research["status"] == "not_configured"
    assert "RESEARCH EVIDENCE" not in llm.calls[0]["user"]


async def test_corrective_pass_keeps_the_original(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [[position(isin="US0378331006")], GOOD]
    made = await service.create_prompt(make())
    first = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    assert first.status == "failed"
    fixed = await settle(
        service, (await service.correct_run(first.id, RunRequest(research="off"))).id
    )
    assert fixed.status == "completed"
    assert fixed.corrects_run_id == first.id
    assert "CORRECTION" in llm.calls[1]["user"]
    assert "invalid_isin" not in llm.calls[1]["user"]
    # The earlier run is exactly as it was, and the saved prompt is not polluted.
    kept = await service.get_run(first.id)
    assert kept.status == "failed"
    assert kept.raw_output == json.dumps([position(isin="US0378331006")])
    assert fixed.prompt_snapshot == PROMPT
    with pytest.raises(SpcConflictError):
        await service.correct_run(fixed.id, RunRequest(research="off"))


# -- restart --------------------------------------------------------------------------------


async def test_restart_settles_interrupted_runs_and_keeps_finished_ones(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    llm.replies = [GOOD]
    runner = SpcRunner(db, llm, tmp_path, None, provider=lambda: None)  # pyright: ignore[reportArgumentType]
    first = SpcService(db, llm, tmp_path, None, runner)  # pyright: ignore[reportArgumentType]
    made = await first.create_prompt(make())
    finished = await settle(first, (await first.start_run(made.id, RunRequest(research="off"))).id)
    async with db.session() as session:
        for status in (SpcRunStatus.QUEUED, SpcRunStatus.RESEARCHING, SpcRunStatus.GENERATING):
            session.add(
                SpcRun(
                    prompt_id=made.id,
                    version_id=made.versions[0].id,
                    version_number=1,
                    prompt_snapshot=PROMPT,
                    meta_snapshot={},
                    model_ref=REF,
                    provider="fake",
                    model_id="m",
                    status=status.value,
                    created_at=utcnow(),
                )
            )
    # A new process: nothing is running these rows.
    second = SpcService(
        db,
        llm,
        tmp_path,
        None,
        SpcRunner(db, llm, tmp_path, None, provider=lambda: None),  # pyright: ignore[reportArgumentType]
    )
    assert await second.reconcile() == 3
    runs = await second.list_runs()
    assert sorted(r.status for r in runs) == ["completed", "failed", "failed", "failed"]
    assert {r.error_code for r in runs if r.status == "failed"} == {"interrupted"}
    kept = await second.get_run(finished.id)
    assert kept.status == "completed"
    assert len(kept.portfolio) == 3
    assert await second.reconcile() == 0


# -- analytics and export ------------------------------------------------------------------


async def test_analytics_counts_what_was_saved(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    done = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    a = await service.analytics(done.id)
    assert a.summary["positions"] == 3
    assert (a.summary["longs"], a.summary["shorts"]) == (2, 1)
    assert a.summary["grossConfidence"] == pytest.approx(1.8)
    assert a.summary["netConfidence"] == pytest.approx(1.0)
    assert a.summary["sector"] is None
    assert a.performance["available"] is False
    assert a.comparison is None


async def test_overlap_and_successive_comparison(service: SpcService, llm: FakeLLM) -> None:
    flipped = [
        position(confidence_score=-0.5),
        position(isin="FR0000120073", ticker="AI", mic="XPAR", company_name="Air Liquide"),
    ]
    llm.replies = [GOOD, flipped]
    made = await service.create_prompt(make())
    a = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    b = await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    assert b.status == "completed", b.error_message
    cmp = (await service.analytics(b.id)).comparison
    assert cmp is not None
    assert [f["ticker"] for f in cmp["flipped"]] == ["AAPL"]
    over = (await service.overlap([a.id, b.id])).overlap
    assert [c["ticker"] for c in over["conflicting"]] == ["AAPL"]
    with pytest.raises(SpcInvalidError):
        await service.overlap([a.id])


async def test_export_blocks_until_tested_then_gives_the_exact_sample(
    service: SpcService, llm: FakeLLM
) -> None:
    raw = json.dumps(GOOD)
    llm.replies = [raw]
    made = await service.create_prompt(make())
    before = await service.export(made.id)
    assert not before.ready
    assert before.package is None
    assert before.sample_json is None
    assert any(c.id == "tested" and c.status == "fail" for c in before.checks)
    await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    after = await service.export(made.id)
    assert after.ready
    assert after.sample_json == raw
    assert after.prompt_text == PROMPT
    assert after.package is not None
    assert after.package["sampleOutput"] == raw
    assert after.package["submission"]["weight"] == "0.25"
    assert "not been submitted" in after.package["notice"]
    status = {c.id: c.status for c in after.checks}
    assert status["same_text"] == "pass"
    assert status["schema"] == "warn"
    assert status["research"] == "warn"


async def test_editing_after_the_test_invalidates_readiness(
    service: SpcService, llm: FakeLLM
) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    await service.update_prompt(made.id, PromptPatch(prompt_text=PROMPT + " tweaked"))
    now = await service.export(made.id)
    assert not now.ready
    assert now.sample_json is None
    # The tested version is still exportable on its own, and says a newer one exists.
    old = await service.export(made.id, (await service.list_runs())[0].id)
    assert old.ready
    assert {c.id: c.status for c in old.checks}["latest"] == "warn"


async def test_metadata_change_after_the_test_warns(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(make())
    await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    await service.update_prompt(made.id, PromptPatch(weight=0.5))
    exp = await service.export(made.id)
    assert exp.ready
    meta = next(c for c in exp.checks if c.id == "metadata")
    assert meta.status == "warn"
    assert "weight" in meta.detail


async def test_zero_weight_is_a_warning_not_a_block(service: SpcService, llm: FakeLLM) -> None:
    llm.replies = [GOOD]
    made = await service.create_prompt(
        PromptCreate(name="x", prompt_text=PROMPT, preferred_model=REF)
    )
    await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    exp = await service.export(made.id)
    assert exp.ready
    assert next(c for c in exp.checks if c.id == "active").status == "warn"


async def test_local_runs_do_not_count_against_submission_limits(
    service: SpcService, llm: FakeLLM
) -> None:
    llm.replies = [GOOD] * 3
    made = await service.create_prompt(make())
    for _ in range(3):
        await settle(service, (await service.start_run(made.id, RunRequest(research="off"))).id)
    limits = (await service.options()).limits
    assert limits["activePrompts"] == 1
    assert limits["maxActivePrompts"] == 50
    assert limits["maxDailySubmissions"] == 50
    assert "do not count" in limits["note"]
