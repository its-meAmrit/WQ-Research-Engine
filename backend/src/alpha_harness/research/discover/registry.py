"""Where research is looked for, kept in one table the person can read and switch.

Each entry is a *place* (a set of academic indexes, a web search limited to some sites, a
feed) with the adapter that reads it. Nothing here is a scraper of one named website: the
sites listed in the starting entries are search filters the person can edit, and an entry is
only ever read through an adapter that exists in this build (:data:`ADAPTERS`).

The quality note on an entry is written by whoever added it and says so. It is not computed
from how often an entry returned something, because that measures the search, not the source.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select, update

from ...db.models import ResearchRegistryEntry, utcnow
from ...db.sqlite import Database  # noqa: TC001 - used at runtime in annotations of the service

log = structlog.get_logger(__name__)

#: The adapters this build can read through. An entry naming another is refused.
ADAPTERS = ("scholarly", "tavily", "rss", "direct_url")
CATEGORIES = (
    "academic_repository",
    "research_blog",
    "financial_article",
    "strategy_repository",
    "user_url",
    "rss_feed",
)
FULL_TEXT = ("open_access_only", "provider_extract", "direct", "metadata_only")
QUALITY_TIERS = ("scholarly_index", "preprint_server", "practitioner", "unrated")
#: arXiv updates its feeds daily, and asks that its API terms be read before using them.
DAILY = 24 * 3600


@dataclass(frozen=True, slots=True)
class Definition:
    id: str
    label: str
    category: str
    adapter: str
    domain: str = "*"
    enabled: bool = True
    policy: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    min_interval: int = 0
    tier: str = "unrated"
    basis: str = ""


def _feed(category: str, label: str) -> Definition:
    return Definition(
        id=f"arxiv_{category.lower().replace('.', '_').replace('-', '_')}",
        label=f"arXiv {label} (new submissions)",
        category="rss_feed",
        adapter="rss",
        domain="rss.arxiv.org",
        policy={"robots": True, "fullText": "metadata_only", "maxPerJob": 25},
        config={"feedUrl": f"https://rss.arxiv.org/rss/{category}"},
        min_interval=DAILY,
        tier="preprint_server",
        basis="arXiv preprints are not peer reviewed. The feed lists new submissions daily.",
    )


#: Seeded at startup. The web entries' domains are starting filters, not endorsements: they
#: are sites that publish public quantitative research, and each page is still checked against
#: its own robots.txt before it is read.
BUILTIN: tuple[Definition, ...] = (
    Definition(
        id="academic_indexes",
        label="Academic indexes (OpenAlex, arXiv)",
        category="academic_repository",
        adapter="scholarly",
        domain="api.openalex.org",
        policy={"fullText": "open_access_only", "maxPerJob": 10},
        tier="scholarly_index",
        basis="Public metadata indexes. Full text is read only where an open copy is named.",
    ),
    Definition(
        id="web_research",
        label="Public finance research articles (web search)",
        category="financial_article",
        adapter="tavily",
        policy={"robots": True, "fullText": "provider_extract", "maxPerJob": 10},
        tier="unrated",
        basis="Open web search. Quality varies by site and is judged per article.",
    ),
    Definition(
        id="quant_blogs",
        label="Quantitative research blogs (web search, listed sites)",
        category="research_blog",
        adapter="tavily",
        policy={"robots": True, "fullText": "provider_extract", "maxPerJob": 10},
        config={"domains": ["alphaarchitect.com", "quantocracy.com", "robotwealth.com"]},
        tier="practitioner",
        basis="Practitioner writing. Methods are often described, rarely peer reviewed.",
    ),
    Definition(
        id="research_houses",
        label="Asset-manager research libraries (web search, listed sites)",
        category="financial_article",
        adapter="tavily",
        policy={"robots": True, "fullText": "provider_extract", "maxPerJob": 10},
        config={"domains": ["aqr.com"]},
        tier="practitioner",
        basis="Research published by an asset manager. Attributable; interests are the firm's.",
    ),
    Definition(
        id="strategy_repositories",
        label="Educational strategy repositories (web search, listed sites)",
        category="strategy_repository",
        adapter="tavily",
        policy={"robots": True, "fullText": "provider_extract", "maxPerJob": 10},
        config={"domains": ["github.com"]},
        tier="unrated",
        basis="Code and READMEs anyone can publish. A repository's claims are unverified.",
    ),
    Definition(
        id="user_urls",
        label="Public URLs you supply",
        category="user_url",
        adapter="direct_url",
        policy={"robots": True, "fullText": "direct", "maxPerJob": 10},
        tier="unrated",
        basis="Whatever you point at. Read only if robots.txt allows it.",
    ),
    _feed("q-fin.PM", "q-fin.PM Portfolio Management"),
    _feed("q-fin.ST", "q-fin.ST Statistical Finance"),
    _feed("q-fin.TR", "q-fin.TR Trading and Market Microstructure"),
)


def view(row: ResearchRegistryEntry) -> dict[str, Any]:
    def iso(value: datetime | None) -> str | None:
        return value.astimezone(UTC).isoformat() if value else None

    return {
        "id": row.id,
        "label": row.label,
        "domain": row.domain,
        "category": row.category,
        "adapter": row.adapter,
        "enabled": row.enabled,
        "retrievalPolicy": row.retrieval_policy,
        "config": row.config,
        "minIntervalSeconds": row.min_interval_seconds,
        "quality": row.quality,
        "builtin": row.builtin,
        "lastAttemptAt": iso(row.last_attempt_at),
        "lastSuccessAt": iso(row.last_success_at),
        "lastError": row.last_error,
        "lastErrorAt": iso(row.last_error_at),
    }


class RegistryError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class Registry:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def seed(self) -> int:
        """Add the built-in entries that are missing. An entry that exists is left as the person
        left it, except that its description and adapter settings follow the build."""
        added = 0
        async with self.db.session() as session:
            have = {
                r.id: r for r in (await session.execute(select(ResearchRegistryEntry))).scalars()
            }
            for d in BUILTIN:
                row = have.get(d.id)
                quality = {"tier": d.tier, "basis": d.basis, "setBy": "built-in"}
                if row is None:
                    session.add(
                        ResearchRegistryEntry(
                            id=d.id,
                            label=d.label,
                            domain=d.domain,
                            category=d.category,
                            adapter=d.adapter,
                            enabled=d.enabled,
                            retrieval_policy=d.policy,
                            config=d.config,
                            min_interval_seconds=d.min_interval,
                            quality=quality,
                            builtin=True,
                        )
                    )
                    added += 1
                elif row.builtin:
                    row.label, row.adapter, row.category = d.label, d.adapter, d.category
                    row.retrieval_policy = d.policy
                    if row.quality.get("setBy") == "built-in":
                        row.quality = quality
        return added

    async def entries(self, *, enabled: bool | None = None) -> list[ResearchRegistryEntry]:
        stmt = select(ResearchRegistryEntry).order_by(
            ResearchRegistryEntry.builtin.desc(), ResearchRegistryEntry.created_at
        )
        if enabled is not None:
            stmt = stmt.where(ResearchRegistryEntry.enabled == enabled)
        async with self.db.session() as session:
            return list((await session.execute(stmt)).scalars().all())

    async def get(self, entry_id: str) -> ResearchRegistryEntry:
        async with self.db.session() as session:
            row = await session.get(ResearchRegistryEntry, entry_id)
        if row is None:
            raise RegistryError("registry_not_found", "There is no such source.", 404)
        return row

    async def update(
        self,
        entry_id: str,
        *,
        enabled: bool | None = None,
        min_interval_seconds: int | None = None,
        domains: list[str] | None = None,
        quality_note: str | None = None,
    ) -> ResearchRegistryEntry:
        row = await self.get(entry_id)
        values: dict[str, Any] = {}
        if enabled is not None:
            values["enabled"] = enabled
        if min_interval_seconds is not None:
            values["min_interval_seconds"] = max(0, min(min_interval_seconds, 30 * DAILY))
        if domains is not None:
            if row.adapter != "tavily":
                raise RegistryError(
                    "no_domains", "Only web-search sources are limited to listed sites.", 422
                )
            values["config"] = row.config | {"domains": _domains(domains)}
        if quality_note is not None:
            values["quality"] = row.quality | {"basis": quality_note[:400], "setBy": "user"}
        if values:
            async with self.db.session() as session:
                await session.execute(
                    update(ResearchRegistryEntry)
                    .where(ResearchRegistryEntry.id == entry_id)
                    .values(**values)
                )
        return await self.get(entry_id)

    async def add_feed(self, label: str, feed_url: str, *, min_interval_seconds: int) -> Any:
        """A feed the person names. Public URL only; read through the same safe fetch."""
        from ...integrations.base import safe_public_url

        url = safe_public_url(feed_url)
        if url is None:
            raise RegistryError(
                "bad_feed_url", "A feed must be an http(s) address on the public web.", 422
            )
        entry_id = f"feed_{abs(hash(url)) % 10**10}"
        async with self.db.session() as session:
            existing = (
                (
                    await session.execute(
                        select(ResearchRegistryEntry).where(ResearchRegistryEntry.adapter == "rss")
                    )
                )
                .scalars()
                .all()
            )
            if any(r.config.get("feedUrl") == url for r in existing):
                raise RegistryError("duplicate_feed", "That feed is already listed.", 409)
            session.add(
                ResearchRegistryEntry(
                    id=entry_id,
                    label=label.strip()[:120] or url,
                    domain=url.split("/")[2].lower(),
                    category="rss_feed",
                    adapter="rss",
                    enabled=True,
                    retrieval_policy={"robots": True, "fullText": "metadata_only", "maxPerJob": 25},
                    config={"feedUrl": url},
                    min_interval_seconds=max(min_interval_seconds, 3600),
                    quality={"tier": "unrated", "basis": "Added by you.", "setBy": "user"},
                    builtin=False,
                )
            )
        return await self.get(entry_id)

    async def remove(self, entry_id: str) -> None:
        row = await self.get(entry_id)
        if row.builtin:
            raise RegistryError(
                "builtin", "Built-in sources cannot be removed; switch them off instead.", 409
            )
        async with self.db.session() as session:
            await session.delete(await session.get(ResearchRegistryEntry, entry_id))

    async def record(self, entry_id: str, *, error: str | None) -> None:
        """Note how the last read went, so the registry shows a source that has stopped working."""
        now = utcnow()
        values: dict[str, Any] = {"last_attempt_at": now}
        if error is None:
            values["last_success_at"] = now
            values["last_error"] = None
        else:
            values |= {"last_error": error[:500], "last_error_at": now}
        async with self.db.session() as session:
            await session.execute(
                update(ResearchRegistryEntry)
                .where(ResearchRegistryEntry.id == entry_id)
                .values(**values)
            )


def _domains(raw: list[str]) -> list[str]:
    """Host names only: no scheme, no path, lower case. Anything else is dropped."""
    out: list[str] = []
    for item in raw:
        host = item.strip().lower().removeprefix("https://").removeprefix("http://").split("/")[0]
        if host and "." in host and " " not in host and host not in out:
            out.append(host)
    return out[:20]
