"""Fixtures. The fake LLM exists only here: the application never substitutes one."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from alpha_harness.db.sqlite import Database
from alpha_harness.llm.service import Answer
from alpha_harness.spc.execution import SpcRunner
from alpha_harness.spc.service import SpcService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

REF = "fake/test-model"


def position(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "isin": "US0378331005",
        "mic": "XNAS",
        "confidence_score": 0.6,
        "investment_horizon": 30,
        "return_prediction": 0.05,
    }
    return base | over


GOOD = [
    position(),
    position(
        company_name="Microsoft Corporation",
        ticker="MSFT",
        isin="US5949181045",
        confidence_score=-0.4,
    ),
    position(
        company_name="Bayer AG",
        ticker="BAYN",
        isin="DE000BAY0017",
        mic="XETR",
        confidence_score=0.8,
    ),
]


@dataclass
class Info:
    ref: str = REF
    id: str = "test-model"
    provider: str = "fake"


class FakeRegistry:
    def get(self, ref: str) -> Info | None:
        return Info() if ref == REF else None


@dataclass
class FakeLLM:
    """Answers with whatever the test queued, and records what it was asked."""

    replies: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    delay: float = 0.0
    keyed: bool = True
    registry: FakeRegistry = field(default_factory=FakeRegistry)

    async def runnable(self) -> list[dict[str, Any]]:
        return (
            [{"id": "test-model", "ref": REF, "provider": "fake", "remainingToday": 100}]
            if self.keyed
            else []
        )

    def model_for(self, ref: str | None) -> Info:
        return Info()

    async def generate(self, **kwargs: Any) -> Answer:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Answer(
            text=text,
            model="test-model",
            prompt_tokens=10,
            output_tokens=20,
            thinking_tokens=0,
            total_tokens=30,
        )


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "t.db")
    await database.create_all()
    yield database
    await database.dispose()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(db: Database, llm: FakeLLM, tmp_path: Path) -> SpcService:
    runner = SpcRunner(db, llm, tmp_path, None, provider=lambda: None)  # pyright: ignore[reportArgumentType]
    return SpcService(db, llm, tmp_path, None, runner)  # pyright: ignore[reportArgumentType]


async def settle(service: SpcService, run_id: int, *, timeout: float = 5.0) -> Any:
    """Wait for a run to reach a terminal status, as the screen does by polling."""
    async with asyncio.timeout(timeout):
        while True:
            run = await service.get_run(run_id)
            if run.status in {"completed", "failed", "cancelled"}:
                return run
            await asyncio.sleep(0.01)
