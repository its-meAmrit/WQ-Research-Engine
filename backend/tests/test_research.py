from datetime import UTC, datetime, timedelta

import httpx2
import pytest

from alpha_harness.spc import research
from alpha_harness.spc.research import (
    BraveSearch,
    ResearchFailedError,
    Source,
    build_queries,
    classify,
    configured_provider,
    evidence_block,
    gather,
    parse_published,
    safe_public_url,
    status,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/x",
        "http://127.0.0.1:8000/api",
        "http://0.0.0.0/",
        "https://10.1.2.3/",
        "https://192.168.0.1/",
        "https://172.16.5.5/",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/",
        "https://[fe80::1]/",
        "file:///etc/passwd",
        "ftp://example.com/f",
        "javascript:alert(1)",
        "https://user:pw@example.com/",
        "https://intranet/page",
        "https://printer.local/",
        "https://svc.internal/",
        "",
        "not a url",
    ],
)
def test_unsafe_urls_are_refused(url: str) -> None:
    assert safe_public_url(url) is None


def test_public_urls_pass() -> None:
    assert safe_public_url("https://www.sec.gov/cgi-bin/x") == "https://www.sec.gov/cgi-bin/x"
    assert safe_public_url("http://8.8.8.8/") == "http://8.8.8.8/"


def test_classification_is_a_label_from_the_domain() -> None:
    assert classify("https://investor.nvidia.com/news") == "investor_relations"
    assert classify("https://www.sec.gov/edgar/x") == "regulatory_filing"
    assert classify("https://www.reuters.com/a") == "financial_news"
    assert classify("https://random-blog.example/a") == "web"


def test_published_dates_are_never_invented() -> None:
    assert parse_published(None, now=NOW) is None
    assert parse_published("", now=NOW) is None
    assert parse_published("last tuesday-ish", now=NOW) is None
    assert parse_published("2026-10-01", now=NOW) == datetime(2026, 10, 1, tzinfo=UTC)
    assert parse_published("2026-10-01T10:00:00Z", now=NOW) == datetime(2026, 10, 1, 10, tzinfo=UTC)
    assert parse_published("3 days ago", now=NOW) == NOW - timedelta(days=3)


def test_configuration_and_status() -> None:
    assert configured_provider({}) is None
    assert configured_provider({research.ENV_PROVIDER: "brave"}) is None
    assert configured_provider({research.ENV_PROVIDER: "bing", research.ENV_KEY: "k"}) is None
    assert (
        configured_provider({research.ENV_PROVIDER: "Tavily", research.ENV_KEY: "k"}).name
        == "tavily"
    )  # pyright: ignore[reportOptionalMemberAccess]
    off = status({})
    assert off["configured"] is False
    assert "ALPHA_HARNESS_SPC_SEARCH" in off["help"]
    assert status({research.ENV_PROVIDER: "bing", research.ENV_KEY: "k"})["help"].startswith(
        "'bing'"
    )
    on = status({research.ENV_PROVIDER: "brave", research.ENV_KEY: "secret"})
    assert on == {"configured": True, "provider": "brave", "help": None}
    assert "secret" not in str(on)


def test_queries_come_from_options_or_the_prompt() -> None:
    q = build_queries(
        {"sector": "semiconductors", "universe": "Global", "thesis": "AI capex"}, "", now=NOW
    )
    assert q
    assert all("November 2026" in x or "valuation" in x for x in q)
    assert len(q) <= 4
    by_hand = build_queries(
        None, "You are an analyst picking European banks with strong capital.", now=NOW
    )
    assert by_hand
    assert build_queries(None, "", now=NOW) == []


def _transport(handler):  # noqa: ANN001, ANN202
    return httpx2.MockTransport(handler)


async def test_gather_normalises_dates_flags_stale_and_drops_unsafe() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["X-Subscription-Token"] == "k"
        return httpx2.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://investor.acme.com/q3",
                            "title": "Acme Q3",
                            "description": "Beat.",
                            "page_age": "2026-11-01T00:00:00",
                        },
                        {
                            "url": "https://old.example.com/a",
                            "title": "Old",
                            "description": "x",
                            "page_age": "2025-01-01T00:00:00",
                        },
                        {
                            "url": "https://nodate.example.com/a",
                            "title": "No date",
                            "description": "x",
                        },
                        {
                            "url": "http://169.254.169.254/latest",
                            "title": "Trap",
                            "description": "x",
                        },
                        {"url": "https://investor.acme.com/q3", "title": "dup", "description": "x"},
                    ]
                }
            },
        )

    result = await gather(BraveSearch("k"), ["q1"], now=NOW, transport=_transport(handler))
    by_title = {s.title: s for s in result.sources}
    assert set(by_title) == {"Acme Q3", "Old", "No date"}
    assert by_title["Acme Q3"].published_at == "2026-11-01"
    assert by_title["Acme Q3"].stale is False
    assert by_title["Acme Q3"].source_type == "investor_relations"
    assert by_title["Old"].stale is True
    # Unknown is its own state: neither fresh nor stale, and no date is made up.
    assert by_title["No date"].published_at is None
    assert by_title["No date"].stale is False
    assert any("unsafe" in n for n in result.notes)
    summary = result.summary()
    assert (summary["sources"], summary["stale"], summary["undated"]) == (3, 1, 1)


async def test_one_failed_query_is_noted_and_the_rest_carry_on() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if "bad" in request.url.params["q"]:
            return httpx2.Response(429)
        return httpx2.Response(
            200,
            json={
                "web": {
                    "results": [{"url": "https://a.example.com/x", "title": "A", "description": ""}]
                }
            },
        )

    result = await gather(BraveSearch("k"), ["good", "bad"], now=NOW, transport=_transport(handler))
    assert len(result.sources) == 1
    assert any("rate limiting" in n for n in result.notes)


async def test_every_query_failing_is_an_error_never_an_empty_success() -> None:
    with pytest.raises(ResearchFailedError, match="refused the API key"):
        await gather(
            BraveSearch("k"),
            ["a", "b"],
            now=NOW,
            transport=_transport(lambda _r: httpx2.Response(401)),
        )


async def test_network_failure_is_reported_without_leaking_the_key() -> None:
    def handler(_r: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    with pytest.raises(ResearchFailedError) as err:
        await gather(BraveSearch("SECRETKEY"), ["a"], now=NOW, transport=_transport(handler))
    assert "SECRETKEY" not in str(err.value)
    assert "could not be reached" in str(err.value)


def test_evidence_block_names_gaps_instead_of_hiding_them() -> None:
    sources = [
        Source("https://a.example.com", "A", "text", "q", "web", None, NOW),
        Source("https://b.example.com", "B", "text", "q", "web", "2025-01-01", NOW, stale=True),
    ]
    block = evidence_block(sources, retrieved=NOW)
    assert "retrieved 2026-11-06" in block
    assert "undated" in block
    assert "stale" in block
