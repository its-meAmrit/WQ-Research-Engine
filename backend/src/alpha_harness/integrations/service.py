"""What the rest of the application calls to use a web research service.

**Which key a search uses**, in order:

1. An *enabled* key stored here, the user's default service first, then the rest in registry
   order.
2. The legacy environment variables, read at the moment of use and never copied into the
   database.
3. Nothing: research is reported as not configured, with the way to set it up.

Every request sent with a stored key is counted against the user's own daily budget first. A
key that comes from the environment has no row to count against, so it is not budgeted, and the
status says so.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import httpx2
import structlog

from . import legacy, registry
from .base import (
    SETTINGS_PATH,
    Capability,
    ExtractResult,
    ReportedUsage,
    SearchProvider,
    Source,
    WebProvider,
)
from .credentials import CredentialStore, seconds_to_reset, today
from .errors import (
    ConfirmationRequiredError,
    IntegrationError,
    NotConfiguredError,
    UnreadableKeyError,
    UnsupportedCapabilityError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..db.models import IntegrationCredential
    from ..db.sqlite import Database
    from ..sealing import Sealer

log = structlog.get_logger(__name__)

#: Seconds a request to a service may take.
TIMEOUT_SECONDS = 15.0


@dataclass(slots=True)
class Resolved:
    """The adapter a call will use, and where its key came from."""

    provider: SearchProvider
    source: Source
    #: The registry id, or ``None`` for a caller-supplied provider that is not one of them.
    provider_id: str | None = None
    #: ``None`` when nothing is stored for it, so there is no budget to count against.
    credential_id: int | None = None


@dataclass(slots=True)
class CheckOutcome:
    provider: str
    ok: bool
    detail: str | None = None
    code: str | None = None
    message: str | None = None
    retry_after: float | None = None
    reported: ReportedUsage | None = None


def _order(rows: list[IntegrationCredential], preferred: str | None) -> list[IntegrationCredential]:
    """Rows in the order to try them: the default first, then registry order."""
    ranking = {name: n for n, name in enumerate(registry.REGISTRY)}
    return sorted(rows, key=lambda r: (r.provider != preferred, ranking[r.provider]))


class WebResearchService:
    def __init__(
        self,
        db: Database,
        sealer: Sealer,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.store = CredentialStore(db, sealer)
        self._env = env
        self._transport = transport

    def _client(self) -> httpx2.AsyncClient:
        # No redirects: a key sent to the provider is not sent to wherever it points next.
        return httpx2.AsyncClient(
            timeout=httpx2.Timeout(TIMEOUT_SECONDS),
            follow_redirects=False,
            transport=self._transport,
        )

    def http_client(self) -> httpx2.AsyncClient:
        """A client under this service's timeout and no-redirect policy, for adapters."""
        return self._client()

    # -- which key -------------------------------------------------------

    async def _candidates(self, capability: Capability) -> list[IntegrationCredential]:
        rows = [
            r
            for r in await self.store.all_rows()
            if r.enabled and capability in registry.spec_for(r.provider).capabilities
        ]
        return _order(rows, await self.store.preferred())

    async def resolve(self, capability: Capability = "search") -> Resolved | None:
        """The adapter to use for ``capability``, by the precedence above, or ``None``."""
        for row in await self._candidates(capability):
            try:
                key = await self.store.secret(row.provider)
            except UnreadableKeyError as exc:
                # Say so on the key, then try the next one: one dead row should not stop research.
                await self.store.mark(row.provider, ok=False, code=exc.code, message=str(exc))
                continue
            return Resolved(registry.build(row.provider, key), "store", row.provider, row.id)
        config = legacy.read(self._env)
        if config is not None and config.usable and config.provider is not None:
            spec = registry.spec_for(config.provider)
            if capability in spec.capabilities:
                return Resolved(
                    registry.build(config.provider, config.key),
                    "environment",
                    config.provider,
                    None,
                )
        return None

    async def resolve_search(self) -> Resolved | None:
        return await self.resolve("search")

    # -- what a call costs -----------------------------------------------

    async def reserve(self, resolved: Resolved, wanted: int) -> int:
        """Count up to ``wanted`` requests against the user's budget, returning how many were
        granted. Unbudgeted sources (the environment, a caller's own) get all of them."""
        if resolved.credential_id is None or resolved.provider_id is None:
            return wanted
        return await self.store.reserve(resolved.provider_id, wanted)

    async def record(self, resolved: Resolved, *, error: IntegrationError | str | None) -> None:
        """Note how a real use went, on the key, so a dead one shows in the UI."""
        if resolved.credential_id is None or resolved.provider_id is None:
            return
        if error is None:
            await self.store.mark(resolved.provider_id, ok=True)
        else:
            code = error.code if isinstance(error, IntegrationError) else "failed"
            await self.store.mark(resolved.provider_id, ok=False, code=code, message=str(error))

    # -- connectivity ----------------------------------------------------

    async def check(self, provider: str, *, acknowledged: bool) -> CheckOutcome:
        """Test a stored key. May cost a credit, so it does nothing until told that is fine.

        Provider failures come back as an outcome, not an exception: a key that is refused is
        the *answer* to a test, not a fault in it.
        """
        spec = registry.spec_for(provider)
        await self.store.require(provider)
        if spec.check_cost.may_consume_credit and not acknowledged:
            raise ConfirmationRequiredError(spec.check_cost.description)
        await self.store.reserve(provider, 1, all_or_nothing=True)
        adapter = registry.build(provider, await self.store.secret(provider))
        try:
            async with self._client() as client:
                report = await adapter.check(client)
        except IntegrationError as exc:
            await self.store.mark(provider, ok=False, code=exc.code, message=str(exc), checked=True)
            return CheckOutcome(
                provider, False, code=exc.code, message=str(exc), retry_after=exc.retry_after
            )
        await self.store.mark(
            provider,
            ok=True,
            checked=True,
            reported=report.reported.to_dict() if report.reported else None,
        )
        return CheckOutcome(provider, True, detail=report.detail, reported=report.reported)

    # -- extraction ------------------------------------------------------

    async def extract(
        self, urls: list[str], *, depth: Literal["basic", "advanced"] = "basic"
    ) -> tuple[str, ExtractResult]:
        """Page text for ``urls`` from the first service that can extract: (provider, result)."""
        resolved = await self.resolve("extract")
        if resolved is None or resolved.provider_id is None:
            raise NotConfiguredError(
                "No enabled service can extract page content. Tavily can: add its key and "
                f"enable it under {SETTINGS_PATH}."
            )
        adapter = resolved.provider
        if not isinstance(adapter, WebProvider) or "extract" not in adapter.spec.capabilities:
            raise UnsupportedCapabilityError("That service cannot extract pages.")
        await self.reserve(resolved, 1)
        try:
            async with self._client() as client:
                result = await adapter.extract(client, urls, depth=depth)
        except IntegrationError as exc:
            await self.record(resolved, error=exc)
            raise
        await self.record(resolved, error=None)
        return resolved.provider_id, result

    # -- what the UI shows -----------------------------------------------

    async def active(self) -> dict[str, Any]:
        """Which service a search would use right now, and why, without opening a key."""
        candidates = await self._candidates("search")
        preferred = await self.store.preferred()
        if candidates:
            row = candidates[0]
            spec = registry.spec_for(row.provider)
            message = None
            if preferred and preferred != row.provider:
                chosen = registry.spec_for(preferred).label
                message = f"Your default, {chosen}, has no enabled key, so {spec.label} is used."
            return {
                "configured": True,
                "provider": row.provider,
                "label": spec.label,
                "source": "store",
                "message": message,
                "help": None,
            }
        config = legacy.read(self._env)
        if config is not None and config.usable and config.provider is not None:
            spec = registry.spec_for(config.provider)
            return {
                "configured": True,
                "provider": config.provider,
                "label": spec.label,
                "source": "environment",
                "message": (
                    f"Using the {legacy.ENV_KEY} environment variable ({spec.label}). That is "
                    f"the older way to set this up: add the key under {SETTINGS_PATH} to see "
                    "it here, test it and cap how much it is used. Environment keys are not "
                    "counted against a budget."
                ),
                "help": None,
            }
        problem = config.problem if config is not None else None
        return {
            "configured": False,
            "provider": None,
            "label": None,
            "source": None,
            "message": None,
            "help": f"{problem} {legacy.SETUP_HELP}" if problem else legacy.SETUP_HELP,
        }

    async def research_status(self) -> dict[str, Any]:
        """What the SPC screens show about research."""
        return await self.active()

    async def provider_status(self, provider: str) -> dict[str, Any]:
        registry.spec_for(provider)
        return next(p for p in (await self.status())["providers"] if p["id"] == provider)

    async def status(self) -> dict[str, Any]:
        rows = {r.provider: r for r in await self.store.all_rows()}
        used = await self.store.usage_today()
        preferred = await self.store.preferred()
        active = await self.active()
        providers: list[dict[str, Any]] = []
        for reg in registry.REGISTRY.values():
            spec, row = reg.spec, rows.get(reg.spec.id)
            providers.append(
                {
                    "id": spec.id,
                    "label": spec.label,
                    "description": spec.description,
                    "capabilities": list(spec.capabilities),
                    "key_url": spec.key_url,
                    "docs_url": spec.docs_url,
                    "check_cost": {
                        "may_consume_credit": spec.check_cost.may_consume_credit,
                        "description": spec.check_cost.description,
                    },
                    "configured": row is not None,
                    "enabled": row.enabled if row else False,
                    "hint": row.hint if row else None,
                    "is_preferred": preferred == spec.id,
                    "is_active": active["provider"] == spec.id and active["source"] == "store",
                    "last_checked_at": _iso(row.last_checked_at) if row else None,
                    "last_ok_at": _iso(row.last_ok_at) if row else None,
                    "last_error": row.last_error if row else None,
                    "last_error_code": row.last_error_code if row else None,
                    "created_at": _iso(row.created_at) if row else None,
                    "usage": _usage(row, used.get(row.id, 0)) if row else None,
                    "reported_usage": row.reported_usage if row else None,
                }
            )
        config = legacy.read(self._env)
        return {
            "providers": providers,
            "preferred_provider": preferred,
            "active": active,
            "legacy_environment": {
                "present": config is not None,
                "provider": config.provider if config else None,
                "usable": bool(config and config.usable),
                "in_use": active["source"] == "environment",
                "note": (
                    config.problem
                    if config is not None and not config.usable
                    else (
                        f"{legacy.ENV_PROVIDER} and {legacy.ENV_KEY} are set. They are used "
                        "only while no enabled key is stored here."
                    )
                    if config is not None
                    else None
                ),
            },
        }


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _usage(row: IntegrationCredential, requests: int) -> dict[str, Any]:
    return {
        "day": today(),
        "requests": requests,
        "daily_budget": row.daily_request_budget,
        "remaining_today": max(0, row.daily_request_budget - requests),
        "resets_in_seconds": round(seconds_to_reset()),
        "note": (
            "Counted here: requests this application sent with the key, per UTC day. Not the "
            "provider's balance; other tools using the same key are not included."
        ),
    }
