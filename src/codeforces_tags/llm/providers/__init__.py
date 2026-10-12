"""The LLM providers: one adapter module per provider, each on one line of ``PROVIDERS``.

An adapter module, ``llm/providers/<provider>.py``, exposes ``SDK_MODULE`` (the import name of
the provider's SDK), ``request_body(request)`` (the body of a request, as the provider's API
takes it) and ``ask(settings)`` (the ``Ask`` that sends requests; it imports the SDK, which no
module imports at module level). A new provider adds its module and one line here, and edits no
other module of the toolkit.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# The adapter module of each provider, by the provider's name in LlmSettings.provider.
PROVIDERS: dict[str, str] = {"openai": "codeforces_tags.llm.providers.openai"}


def provider_module(provider: str) -> ModuleType:
    """Import a provider's adapter module.

    Args:
        provider: A key of ``PROVIDERS``.

    Returns:
        The module.

    Raises:
        ValueError: If ``provider`` is not a key of ``PROVIDERS``.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"{provider!r} is not a provider: expected one of {sorted(PROVIDERS)}.")
    return importlib.import_module(PROVIDERS[provider])


def sdk_modules() -> tuple[str, ...]:
    """Return the import name of every provider's SDK, without importing any SDK.

    A tagging method's prediction path imports none of them (``conformance``).

    Returns:
        The ``SDK_MODULE`` of each adapter, in ``PROVIDERS`` order.
    """
    return tuple(provider_module(provider).SDK_MODULE for provider in PROVIDERS)
