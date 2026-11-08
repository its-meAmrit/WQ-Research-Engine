"""Web service keys, sealed at rest, with the user's own request budget.

Sealed with the application's AES-GCM sealer under a context of its own, bound to the provider:
a blob sealed for Tavily will not open as Brave's, nor as an LLM key or the BRAIN password.
Keys leave this module only as a masked hint, except to the adapter that makes a call.
"""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..db.models import IntegrationCredential, IntegrationUsage, Preference, utcnow
from ..sealing import SealError
from . import registry
from .base import SETTINGS_PATH
from .errors import (
    DuplicateKeyError,
    InvalidKeyError,
    NotConfiguredError,
    RequestBudgetExhaustedError,
    UnreadableKeyError,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..db.sqlite import Database
    from ..sealing import Sealer

log = structlog.get_logger(__name__)

CONTEXT = "web-integration"
#: A cap new keys start with. Low enough that a runaway loop is a nuisance, not a bill, and far
#: above what a Test Run spends (it sends at most four searches).
DEFAULT_DAILY_BUDGET = 100
MAX_DAILY_BUDGET = 100_000
PREFERENCE_KEY = "web_research_provider"

MIN_KEY_CHARS = 8
MAX_KEY_CHARS = 256


def seal_context(provider: str) -> str:
    return f"{CONTEXT}:{provider}"


def fingerprint(key: str) -> str:
    """Stable identity for a key, so the same one is not stored twice. A salted hash, not
    the key: the database should not hold a second recoverable copy of a secret."""
    return hashlib.sha256(f"{CONTEXT}\0{key}".encode()).hexdigest()


def hint(key: str) -> str:
    """Enough of a key to recognise it, not enough to use it."""
    if len(key) < 16:
        return "…" + key[-3:]
    return f"{key[:5]}…{key[-4:]}"


def clean_key(raw: str) -> str:
    """The key, trimmed, or :class:`InvalidKeyError` saying what is wrong with it.

    Structure only. Whether the provider accepts it is for the connectivity test to say:
    key formats change, and refusing a good key on a guess is worse than testing a bad one.
    """
    key = raw.strip()
    if not key:
        raise InvalidKeyError("That key is empty.")
    if not MIN_KEY_CHARS <= len(key) <= MAX_KEY_CHARS:
        raise InvalidKeyError(
            f"That does not look like an API key: keys here are {MIN_KEY_CHARS} to "
            f"{MAX_KEY_CHARS} characters, and this one is {len(key)}."
        )
    if not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        raise InvalidKeyError(
            "A key is one run of letters, digits and symbols. This one has spaces, line breaks "
            "or other characters in it; paste only the key itself."
        )
    if key.lower().startswith(("bearer ", "token ")) or "://" in key:
        raise InvalidKeyError("Paste the key alone, without 'Bearer', a URL or other wording.")
    return key


def check_budget(value: int) -> int:
    if not 1 <= value <= MAX_DAILY_BUDGET:
        raise InvalidKeyError(
            f"A daily request budget has to be between 1 and {MAX_DAILY_BUDGET:,} requests."
        )
    return value


def today() -> str:
    return utcnow().date().isoformat()


def seconds_to_reset(now: datetime | None = None) -> float:
    """Until 00:00 UTC, when the daily count starts again."""
    now = now or utcnow()
    tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    return (tomorrow - now).total_seconds()


class CredentialStore:
    """Keys, their health, the daily request count and the default service."""

    def __init__(self, db: Database, sealer: Sealer) -> None:
        self.db = db
        self.sealer = sealer
        # Reading the count and taking a slot must be one step, or two callers reading between
        # them would both take the last one. Held only for local reads.
        self._reserving = asyncio.Lock()

    # -- keys ------------------------------------------------------------

    async def all_rows(self) -> list[IntegrationCredential]:
        async with self.db.session() as session:
            rows = (await session.scalars(select(IntegrationCredential))).all()
        # A row a newer build wrote for a service this one does not know is left alone.
        return [r for r in rows if registry.known(r.provider)]

    async def get(self, provider: str) -> IntegrationCredential | None:
        registry.spec_for(provider)
        async with self.db.session() as session:
            return await session.scalar(
                select(IntegrationCredential).where(IntegrationCredential.provider == provider)
            )

    async def require(self, provider: str) -> IntegrationCredential:
        row = await self.get(provider)
        if row is None:
            label = registry.spec_for(provider).label
            raise NotConfiguredError(
                f"No {label} key has been added yet. Add one under {SETTINGS_PATH}."
            )
        return row

    async def put(
        self, provider: str, key: str, daily_budget: int | None = None
    ) -> tuple[IntegrationCredential, bool]:
        """Store a key for ``provider``, or replace the one it has. ``(row, replaced)``.

        One key per service: a search API's quota is per account, so a second key from the same
        account adds nothing, and picking between accounts is not something this app does.
        Replacing keeps the user's budget and on/off choice but drops the old key's check
        history, which says nothing about the new one.
        """
        spec = registry.spec_for(provider)
        clean = clean_key(key)
        if daily_budget is not None:
            check_budget(daily_budget)
        print_ = fingerprint(clean)
        replaced = False
        try:
            async with self.db.session() as session:
                clash = await session.scalar(
                    select(IntegrationCredential).where(IntegrationCredential.fingerprint == print_)
                )
                if clash is not None:
                    where = (
                        f"already stored for {spec.label}"
                        if clash.provider == provider
                        else f"already stored as a {registry.spec_for(clash.provider).label} key"
                    )
                    raise DuplicateKeyError(f"That key is {where}.")
                row = await session.scalar(
                    select(IntegrationCredential).where(IntegrationCredential.provider == provider)
                )
                if row is None:
                    row = IntegrationCredential(
                        provider=provider,
                        key_sealed=self.sealer.seal(clean, context=seal_context(provider)),
                        hint=hint(clean),
                        fingerprint=print_,
                        daily_request_budget=daily_budget or DEFAULT_DAILY_BUDGET,
                    )
                    session.add(row)
                else:
                    replaced = True
                    row.key_sealed = self.sealer.seal(clean, context=seal_context(provider))
                    row.hint = hint(clean)
                    row.fingerprint = print_
                    row.last_checked_at = row.last_ok_at = None
                    row.last_error = row.last_error_code = None
                    row.reported_usage = None
                    if daily_budget is not None:
                        row.daily_request_budget = daily_budget
                await session.flush()
                await session.refresh(row)
        except IntegrityError:
            # Two requests raced past the checks above; the database is the final word.
            raise DuplicateKeyError(f"{spec.label} already has a key stored.") from None
        log.info("web.key.stored", provider=provider, hint=row.hint, replaced=replaced)
        return row, replaced

    async def update(
        self, provider: str, *, enabled: bool | None = None, daily_budget: int | None = None
    ) -> IntegrationCredential:
        if daily_budget is not None:
            check_budget(daily_budget)
        async with self.db.session() as session:
            row = await self._row(session, provider)
            if enabled is not None:
                row.enabled = enabled
            if daily_budget is not None:
                row.daily_request_budget = daily_budget
            await session.flush()
            await session.refresh(row)
        log.info("web.key.updated", provider=provider, enabled=row.enabled)
        return row

    async def remove(self, provider: str) -> None:
        """Delete a key, its usage count, and the default if it was this one."""
        async with self.db.session() as session:
            row = await self._row(session, provider)
            await session.delete(row)
            chosen = await session.get(Preference, PREFERENCE_KEY)
            if chosen is not None and chosen.value == provider:
                await session.delete(chosen)
        log.info("web.key.removed", provider=provider)

    @staticmethod
    async def _row(session: AsyncSession, provider: str) -> IntegrationCredential:
        """The key's row, or the error that says what to do about there not being one."""
        spec = registry.spec_for(provider)
        row = await session.scalar(
            select(IntegrationCredential).where(IntegrationCredential.provider == provider)
        )
        if row is None:
            raise NotConfiguredError(
                f"No {spec.label} key has been added yet. Add one under {SETTINGS_PATH}."
            )
        return row

    async def secret(self, provider: str) -> str:
        """The key, for the adapter that is about to use it. Nothing else should call this."""
        row = await self.require(provider)
        try:
            return self.sealer.open(row.key_sealed, context=seal_context(provider))
        except SealError:
            raise UnreadableKeyError(
                f"The stored {registry.spec_for(provider).label} key could not be read. This "
                f"happens if the application's encryption key was replaced. Add the key again "
                f"under {SETTINGS_PATH}."
            ) from None

    async def mark(
        self,
        provider: str,
        *,
        ok: bool,
        code: str | None = None,
        message: str | None = None,
        reported: dict[str, Any] | None = None,
        checked: bool = False,
    ) -> None:
        """Record how a call went, so a dead key is visible rather than mysterious.

        ``checked`` is set by the connectivity test; a real use records the outcome without
        claiming a check happened.
        """
        async with self.db.session() as session:
            row = await session.scalar(
                select(IntegrationCredential).where(IntegrationCredential.provider == provider)
            )
            if row is None:
                return
            now = utcnow()
            if checked:
                row.last_checked_at = now
            if ok:
                row.last_ok_at = now
                row.last_error = row.last_error_code = None
            else:
                row.last_error = (message or "")[:500] or None
                row.last_error_code = code
            if reported is not None:
                row.reported_usage = reported

    # -- default service -------------------------------------------------

    async def preferred(self) -> str | None:
        async with self.db.session() as session:
            row = await session.get(Preference, PREFERENCE_KEY)
        return row.value if row is not None and registry.known(str(row.value)) else None

    async def set_preferred(self, provider: str | None) -> None:
        if provider is not None:
            await self.require(provider)  # a default must be one that has a key
        async with self.db.session() as session:
            row = await session.get(Preference, PREFERENCE_KEY)
            if provider is None:
                if row is not None:
                    await session.delete(row)
            elif row is None:
                session.add(Preference(key=PREFERENCE_KEY, value=provider))
            else:
                row.value = provider

    # -- budget ----------------------------------------------------------

    async def usage_today(self) -> dict[int, int]:
        """Requests sent today, by credential id."""
        async with self.db.session() as session:
            rows = await session.scalars(
                select(IntegrationUsage).where(IntegrationUsage.day == today())
            )
            return {r.credential_id: r.requests for r in rows}

    async def reserve(self, provider: str, wanted: int, *, all_or_nothing: bool = False) -> int:
        """Take up to ``wanted`` requests from today's budget and return how many were granted.

        Counted before they are sent, so a crash mid-batch errs towards the user's cap rather
        than past it. Raises when none are left, or, with ``all_or_nothing``, when fewer than
        ``wanted`` are.
        """
        async with self._reserving, self.db.session() as session:
            row = await self._row(session, provider)
            usage = await session.scalar(
                select(IntegrationUsage).where(
                    IntegrationUsage.credential_id == row.id, IntegrationUsage.day == today()
                )
            )
            used = usage.requests if usage else 0
            left = max(0, row.daily_request_budget - used)
            if left == 0 or (all_or_nothing and left < wanted):
                label = registry.spec_for(provider).label
                raise RequestBudgetExhaustedError(
                    f"Today's {label} budget is used up: {used} of {row.daily_request_budget} "
                    f"requests sent. It starts again at 00:00 UTC, or raise it under "
                    f"{SETTINGS_PATH} if that is what you want.",
                    retry_after=seconds_to_reset(),
                )
            granted = min(wanted, left)
            if usage is None:
                usage = IntegrationUsage(credential_id=row.id, day=today(), requests=0)
                session.add(usage)
            usage.requests += granted
            usage.last_request_at = utcnow()
            return granted
