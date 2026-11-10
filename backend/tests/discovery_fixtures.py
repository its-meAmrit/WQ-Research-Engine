"""Fixtures for Discover: a Tavily stand-in, article pages, and model answers.

Nothing here reaches a network, a real model or BRAIN. Tavily is an ``httpx2.MockTransport``
that records each request; pages are served by the paper tests' ``FakeWeb``.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx2

from . import paper_fixtures as fx

TAVILY_KEY = "tvly-dev-SECRET-0123456789abcdef"
ARTICLE_URL = "https://quant.example.com/research/pead"
OTHER_URL = "https://blog.example.org/earnings-drift"
THIRD_URL = "https://papers.example.net/revisions"

SURPRISE_QUOTE = (
    "Stocks with large positive earnings surprises keep outperforming for about sixty trading "
    "days after the announcement."
)
REVISION_QUOTE = (
    "Companies whose analysts revise forecasts upward after earnings tend to beat the market "
    "over the next quarter."
)
TOPIC = "earnings surprise cross-sectional equity anomaly"


def article(
    *, extra: str = "", title: str = "Post-Earnings Announcement Drift in Practice"
) -> bytes:
    filler = (
        "We rank every stock in the universe each month by its standardised earnings surprise "
        "and hold the top decile for the following quarter, rebalancing monthly. "
        "Earnings come from the quarterly filings and analyst consensus data. "
    ) * 3
    return f"""<html><head><title>{title}</title>
<meta name="author" content="Jane Quant">
<meta property="article:published_time" content="2026-03-02">
<script>alert('ignore me')</script></head>
<body><nav>Home | About | Subscribe</nav><article>
<h1>{title}</h1>
<h2>The idea</h2>
<p>{SURPRISE_QUOTE} {filler}</p>
<h2>A second effect</h2>
<p>{REVISION_QUOTE} {filler}</p>
<p>{extra}</p>
</article><footer>Copyright</footer></body></html>""".encode()


def paywalled() -> bytes:
    return (
        b"<html><head><title>Premium</title></head><body><article><p>Subscribe to continue "
        b"reading. This article is for subscribers only. Sign in to read the full text."
        b"</p></article></body></html>"
    )


def web_idea(title: str, quote: str, **over: Any) -> dict[str, Any]:
    base = fx.idea(title, quote) | {
        "original_hypothesis": "Stocks react slowly to earnings news.",
        "idea_kind": "documented_factor",
        "expected_relationship": "Higher surprise leads to higher returns.",
        "rebalancing": "Monthly.",
        "brain_implementation": "Rank a standardised earnings surprise field, long high.",
        "key_assumptions": ["Announcement dates are known."],
        "reported_performance": ["The author reports a Sharpe of 1.2."],
        "methodology_documented": True,
        "evidence": [
            {"section_id": None, "page": 1, "quote": quote, "supports": "the effect itself"}
        ],
    }
    return base | over


def web_extraction(*ideas: dict[str, Any], **over: Any) -> dict[str, Any]:
    return {
        "source_class": "market_hypothesis",
        "classification_reason": "A practitioner write-up with a method.",
        "summary": "Earnings surprises predict returns.",
        "ideas": list(ideas),
        "no_ideas_reason": None if ideas else "No testable idea.",
    } | over


def two_web_ideas() -> dict[str, Any]:
    return web_extraction(
        web_idea("Post-earnings drift", SURPRISE_QUOTE),
        web_idea(
            "Analyst revision momentum",
            REVISION_QUOTE,
            topic="analyst",
            signal_definition="Rank by the change in consensus after earnings.",
        ),
    )


def search_body(*urls: str, spend: float | None = 1) -> dict[str, Any]:
    return {
        "results": [
            {
                "url": url,
                "title": f"Result {n}",
                "content": "Snippet about earnings drift.",
                "published_date": "2026-03-02",
                "score": 0.9 - n / 10,
            }
            for n, url in enumerate(urls)
        ],
        "usage": {"credits": spend},
    }


@dataclass
class Tavily:
    """Answers Tavily's endpoints from a queue; records every request."""

    results: dict[str, Any] = field(default_factory=lambda: search_body(ARTICLE_URL, OTHER_URL))
    #: Statuses to answer the next searches with, before ``results`` is served.
    failures: list[httpx2.Response] = field(default_factory=list)
    pages: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    requests: list[httpx2.Request] = field(default_factory=list)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/search":
            if self.failures:
                return self.failures.pop(0)
            return httpx2.Response(200, json=self.results)
        if path == "/extract":
            urls = json.loads(request.content)["urls"]
            return httpx2.Response(
                200,
                json={
                    "results": [
                        {"url": u, "raw_content": self.pages[u]} for u in urls if u in self.pages
                    ],
                    "failed_results": [
                        {"url": u, "error": self.failed.get(u, "unreachable")}
                        for u in urls
                        if u not in self.pages
                    ],
                },
            )
        return httpx2.Response(404)

    def sent(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    @property
    def hosts(self) -> set[str]:
        return {r.url.host for r in self.requests}
