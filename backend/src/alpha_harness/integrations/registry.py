"""The services this build knows, and how to build an adapter for one.

Adding a service is a module with a ``SPEC`` and an adapter class, one line in
:data:`REGISTRY`, and its id in :data:`~.base.ProviderId`. Nothing else in the application
names a provider: the store, the API and the UI all read this.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .brave import SPEC as BRAVE
from .brave import BraveSearch
from .errors import UnknownProviderError
from .tavily import SPEC as TAVILY
from .tavily import TavilySearch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import ProviderSpec, WebProvider


@dataclass(frozen=True, slots=True)
class Registration:
    spec: ProviderSpec
    build: Callable[[str], WebProvider]


#: In display order, which is also the order used to break a tie when no default is chosen.
REGISTRY: dict[str, Registration] = {
    "tavily": Registration(TAVILY, TavilySearch),
    "brave": Registration(BRAVE, BraveSearch),
}


def known(provider: str) -> bool:
    return provider in REGISTRY


def spec_for(provider: str) -> ProviderSpec:
    """The spec for ``provider``, or :class:`UnknownProviderError` listing what exists."""
    found = REGISTRY.get(provider)
    if found is None:
        raise UnknownProviderError(
            f"{provider[:40]!r} is not a web research service here. "
            f"Choose one of: {', '.join(REGISTRY)}."
        )
    return found.spec


def build(provider: str, key: str) -> WebProvider:
    spec_for(provider)  # raises for an unknown one
    return REGISTRY[provider].build(key)
