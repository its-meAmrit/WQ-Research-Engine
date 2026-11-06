"""The existing JSON modes must send exactly what they always did; ``plain`` is additive."""

from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from openai import Omit

from alpha_harness.llm.openai_compat import OpenAICompatible


class Capture:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content="[]")
        usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 1})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def client() -> tuple[OpenAICompatible, Capture]:
    compat = OpenAICompatible("https://example.invalid/v1", "k", httpx2.AsyncClient())
    capture = Capture()
    compat._sdk = SimpleNamespace(chat=SimpleNamespace(completions=capture))  # type: ignore[assignment]  # pyright: ignore
    return compat, capture


async def test_default_still_asks_for_a_json_object(
    client: tuple[OpenAICompatible, Capture],
) -> None:
    compat, capture = client
    await compat.generate(model="m", system="sys", user="user", temperature=0.5, schema=None)
    assert capture.kwargs["response_format"] == {"type": "json_object"}
    assert capture.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


async def test_schema_mode_unchanged(client: tuple[OpenAICompatible, Capture]) -> None:
    compat, capture = client
    schema = {"type": "object"}
    await compat.generate(model="m", system="sys", user="user", temperature=None, schema=schema)
    assert capture.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }


async def test_an_empty_system_message_is_still_sent_in_existing_modes(
    client: tuple[OpenAICompatible, Capture],
) -> None:
    compat, capture = client
    await compat.generate(model="m", system="", user="user", temperature=None, schema=None)
    assert capture.kwargs["messages"][0] == {"role": "system", "content": ""}


async def test_plain_sends_no_response_format_and_no_empty_system(
    client: tuple[OpenAICompatible, Capture],
) -> None:
    compat, capture = client
    text, usage = await compat.generate(
        model="m", system="", user="the prompt", temperature=0.3, schema=None, plain=True
    )
    assert isinstance(capture.kwargs["response_format"], Omit)
    assert capture.kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
    assert (text, usage) == ("[]", {"total_tokens": 1})


async def test_plain_ignores_a_schema(client: tuple[OpenAICompatible, Capture]) -> None:
    compat, capture = client
    await compat.generate(
        model="m", system="s", user="u", temperature=None, schema={"type": "object"}, plain=True
    )
    assert isinstance(capture.kwargs["response_format"], Omit)
    assert capture.kwargs["messages"][0] == {"role": "system", "content": "s"}
