"""Web research services: the keys for them, the adapters that call them, and the budget.

These are the non-LLM services (Tavily, Brave Search). They are deliberately separate from
:mod:`..llm`, which is built around generative models: a search API has no model, no
per-model limits and no key rotation, and forcing it through those would mean inventing a
model to hang a key on.

* :mod:`.base`: the contract every service meets, shared HTTP and error handling.
* :mod:`.registry`: the services that exist. The one place a new service is added.
* :mod:`.credentials`: sealed keys, the user's request budget, the default service.
* :mod:`.service`: what the rest of the application calls.
* :mod:`.legacy`: the pre-UI environment variables, kept as a fallback.
"""
