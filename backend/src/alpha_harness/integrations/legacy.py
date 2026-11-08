"""The environment variables SPC research was configured with before there was a settings UI.

``ALPHA_HARNESS_SPC_SEARCH`` names the provider and ``ALPHA_HARNESS_SPC_SEARCH_KEY`` holds its
key. They keep working, as a fallback behind anything configured in the application. They are
read at the moment of use and never copied into the database: a key the user put in the
environment stays where they put it.
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import registry

if TYPE_CHECKING:
    from collections.abc import Mapping

ENV_PROVIDER = "ALPHA_HARNESS_SPC_SEARCH"
ENV_KEY = "ALPHA_HARNESS_SPC_SEARCH_KEY"

SETUP_HELP = (
    "No web research service is set up. Add a Tavily or Brave Search key under API "
    "Integration › Web Research. (The older "
    f"{ENV_PROVIDER} and {ENV_KEY} environment variables still work as a fallback.) Without "
    "research a run can still go ahead, but only on the model's own knowledge, which is not "
    "current."
)


@dataclass(frozen=True, slots=True)
class LegacyConfig:
    #: The provider name as written, so a typo can be quoted back.
    named: str
    #: The key. Excluded from ``repr`` so a stray log line or traceback cannot print it.
    key: str = field(repr=False)

    @property
    def provider(self) -> str | None:
        """The registry id, or ``None`` when the name is not a service here."""
        name = self.named.strip().lower()
        return name if registry.known(name) else None

    @property
    def usable(self) -> bool:
        return self.provider is not None and bool(self.key)

    @property
    def problem(self) -> str | None:
        if not self.named:
            return f"{ENV_PROVIDER} is not set."
        if self.provider is None:
            return f"{self.named!r} is not a search provider here; use 'brave' or 'tavily'."
        if not self.key:
            return f"{ENV_KEY} is not set."
        return None


def read(env: Mapping[str, str] | None = None) -> LegacyConfig | None:
    """The legacy configuration, or ``None`` when neither variable is set."""
    env = os.environ if env is None else env
    named = (env.get(ENV_PROVIDER) or "").strip()
    key = (env.get(ENV_KEY) or "").strip()
    if not named and not key:
        return None
    return LegacyConfig(named=named, key=key)
