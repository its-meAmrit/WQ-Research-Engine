"""An existing database, with data, gains the SPC tables and loses nothing."""

import sqlite3
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alpha_harness.db.models import Base, LLMModel, MetadataCache, Preference
from alpha_harness.db.sqlite import Database
from alpha_harness.spc.service import SpcService
from alpha_harness.spc.views import PromptCreate

if TYPE_CHECKING:
    from pathlib import Path

SPC_TABLES = {"spc_prompt", "spc_prompt_version", "spc_run", "spc_position", "spc_evidence"}
WEB_TABLES = {"integration_credential", "integration_usage"}
NEW_TABLES = SPC_TABLES | WEB_TABLES


def build_old_database(path: Path) -> None:
    """The schema as it was before SPC: every table but the new ones, filled with real rows."""
    engine = create_engine(f"sqlite:///{path}")
    old = [t for t in Base.metadata.sorted_tables if t.name not in NEW_TABLES]
    Base.metadata.create_all(engine, tables=old)
    engine.dispose()
    # Through the ORM, so every required column and default is filled the way the app does.
    with Session(create_engine(f"sqlite:///{path}")) as session:
        session.add_all(
            [
                Preference(key="theme", value="dark"),
                Preference(key="layout", value={"a": 1}),
                MetadataCache(key="k", value={"v": 1}),
                LLMModel(
                    provider="google",
                    model="gemini-x",
                    requests_per_minute=10,
                    requests_per_day=100,
                ),
            ]
        )
        session.commit()


def snapshot(path: Path, tables: list[str]) -> dict[str, tuple[list[str], list[tuple]]]:
    with sqlite3.connect(path) as db:
        out = {}
        for table in tables:
            columns = [r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]
            rows = db.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()
            out[table] = (columns, rows)
        return out


def table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as db:
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


async def test_old_database_gains_spc_tables_and_keeps_every_row(tmp_path: Path) -> None:
    path = tmp_path / "harness.db"
    build_old_database(path)
    assert not NEW_TABLES & table_names(path)
    existing = [t.name for t in Base.metadata.sorted_tables if t.name not in NEW_TABLES]
    before = snapshot(path, existing)
    assert len(before["preference"][1]) == 2

    db = Database(path)
    await db.create_all()
    try:
        assert table_names(path) >= NEW_TABLES
        # Existing tables: identical columns and identical rows.
        assert snapshot(path, existing) == before

        # And the new tables work on top of the migrated database.
        service = SpcService(db, None, tmp_path)  # pyright: ignore[reportArgumentType]
        made = await service.create_prompt(PromptCreate(name="p", prompt_text="text"))
        assert (await service.get_prompt(made.id)).latest_version == 1
    finally:
        await db.dispose()


async def test_migrating_twice_is_harmless_and_keeps_spc_data(tmp_path: Path) -> None:
    path = tmp_path / "harness.db"
    build_old_database(path)
    db = Database(path)
    await db.create_all()
    service = SpcService(db, None, tmp_path)  # pyright: ignore[reportArgumentType]
    made = await service.create_prompt(PromptCreate(name="kept", prompt_text="text", weight=0.4))
    await db.dispose()

    # A restart.
    again = Database(path)
    await again.create_all()
    try:
        service = SpcService(again, None, tmp_path)  # pyright: ignore[reportArgumentType]
        got = await service.get_prompt(made.id)
        assert (got.name, got.weight) == ("kept", 0.4)
        assert len(await service.list_prompts()) == 1
    finally:
        await again.dispose()


async def test_deleting_a_prompt_row_cascades_only_its_own_history(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    db = Database(path)
    await db.create_all()
    service = SpcService(db, None, tmp_path)  # pyright: ignore[reportArgumentType]
    a = await service.create_prompt(PromptCreate(name="a", prompt_text="one"))
    b = await service.create_prompt(PromptCreate(name="b", prompt_text="two"))
    await db.dispose()
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("DELETE FROM spc_prompt WHERE id = ?", (a.id,))
        left = raw.execute("SELECT prompt_id FROM spc_prompt_version").fetchall()
    assert left == [(b.id,)]


async def test_web_research_keys_survive_a_restart_on_a_migrated_database(tmp_path: Path) -> None:
    """The web service tables arrive on an old database, a key stored in them is still there
    after a restart, and the LLM key tables and rows are not touched."""
    from alpha_harness.integrations.service import WebResearchService
    from alpha_harness.sealing import Sealer

    path = tmp_path / "harness.db"
    build_old_database(path)
    llm_before = snapshot(path, ["llm_model", "preference"])
    sealer = Sealer(tmp_path / "key")

    db = Database(path)
    await db.create_all()
    await WebResearchService(db, sealer, env={}).store.put("tavily", "tvly-dev-0123456789abcdef")
    await db.dispose()

    again = Database(path)
    await again.create_all()
    try:
        store = WebResearchService(again, Sealer(tmp_path / "key"), env={}).store
        assert await store.secret("tavily") == "tvly-dev-0123456789abcdef"
    finally:
        await again.dispose()
    assert snapshot(path, ["llm_model"])["llm_model"] == llm_before["llm_model"]
    assert "tvly-dev-0123456789abcdef" not in "\n".join(sqlite3.connect(path).iterdump())
