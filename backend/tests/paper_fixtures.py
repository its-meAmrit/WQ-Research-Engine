"""Local fixtures for the paper pipeline: PDFs built here, registries and a field catalogue
faked. Nothing reaches the network, a real model or BRAIN."""

import json
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import httpx2
import pymupdf

from alpha_harness.catalog.queries import FieldFilter, Tuple4
from alpha_harness.research import netfetch

VALUE_QUOTE = (
    "Value strategies buy cheap assets and sell expensive ones, measured by book-to-market."
)
MOMENTUM_QUOTE = (
    "Momentum strategies buy past winners over the prior 12 months, skipping the most recent month."
)
FORMULA = "r_t = alpha + beta * BM_t-1 + e_t        (1)"
DOI = "10.1111/jofi.12021"
ABSTRACT = (
    "We find consistent value and momentum return premia across eight diverse markets and "
    "asset classes. Value and momentum are negatively correlated."
)

USA = Tuple4(region="USA", delay=1, universe="TOP3000")
EUR = Tuple4(region="EUR", delay=1, universe="TOP1200")


def _box(page: Any, y: float, text: str, size: float = 10) -> None:
    page.insert_textbox(pymupdf.Rect(72, y, 540, y + 300), text, fontsize=size, fontname="helv")


@cache
def paper_pdf() -> bytes:
    """A four-page paper with two separable ideas (value, momentum), an equation and a table."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 80), "Value and Momentum Everywhere", fontsize=20, fontname="hebo")
    p1.insert_text((72, 110), "Cliff Asness, Tobias Moskowitz", fontsize=11)
    p1.insert_text((72, 140), f"doi:{DOI}", fontsize=9)
    p1.insert_text((72, 170), "Abstract", fontsize=12, fontname="hebo")
    _box(p1, 180, ABSTRACT)
    p1.insert_text((72, 260), "1. Introduction", fontsize=13, fontname="hebo")
    _box(p1, 270, f"{VALUE_QUOTE} {MOMENTUM_QUOTE}")
    p2 = doc.new_page()
    p2.insert_text((72, 80), "2. Data and Methodology", fontsize=13, fontname="hebo")
    _box(p2, 90, "We use monthly returns and book value of equity from Compustat.")
    p2.insert_text((72, 180), FORMULA, fontsize=11)
    p2.insert_text((72, 230), "Table 1: Summary statistics", fontsize=10)
    for r in range(4):
        for c in range(3):
            x, y = 72 + c * 90, 245 + r * 20
            p2.draw_rect(pymupdf.Rect(x, y, x + 90, y + 20))
            p2.insert_text(
                (x + 4, y + 14), ["Mkt", "Mean", "Sharpe"][c] if r == 0 else f"{r}.{c}", fontsize=9
            )
    p3 = doc.new_page()
    p3.insert_text((72, 80), "3. Conclusion", fontsize=13, fontname="hebo")
    _box(p3, 90, "Value and momentum are pervasive. A combined strategy improves Sharpe ratios.")
    p4 = doc.new_page()
    p4.insert_text((72, 80), "References", fontsize=13, fontname="hebo")
    _box(
        p4, 90, "Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers."
    )
    return doc.tobytes()


@cache
def other_paper_pdf() -> bytes:
    """A different, shorter paper: a different hash and no DOI."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Low Volatility Anomaly", fontsize=20, fontname="hebo")
    page.insert_text((72, 170), "Abstract", fontsize=12, fontname="hebo")
    _box(
        page,
        180,
        "Low volatility stocks earn higher risk adjusted returns than high volatility stocks.",
    )
    return doc.tobytes()


@cache
def scanned_pdf() -> bytes:
    """Pages that are pictures: no text layer."""
    source = pymupdf.open()
    sp = source.new_page()
    sp.insert_text((72, 100), "Scanned words", fontsize=30)
    pix = sp.get_pixmap(dpi=50)
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


CORRUPT = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\ntrailer garbage"


def variable(name: str, terms: list[str], **over: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} as the paper defines it",
        "kind": "fundamental",
        "essential": True,
        "search_terms": terms,
        "point_in_time_risk": "none",
    } | over


def idea(title: str, quote: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": title,
        "topic": "value",
        "main_finding": "Cheap assets outperform expensive ones.",
        "economic_explanation": "Investors overpay for glamour.",
        "signal_definition": "Rank stocks by book-to-market.",
        "signal_direction": "long_high",
        "variables": [
            variable("book-to-market", ["book value", "market capitalization"]),
        ],
        "operators_needed": ["rank"],
        "reported_formula": None,
        "formula_page": None,
        "lookback_period": "Not stated in the text.",
        "holding_period": "One month.",
        "universe": "NYSE, AMEX and Nasdaq stocks.",
        "markets_studied": ["United States"],
        "point_in_time_requirements": "Fundamentals lagged six months.",
        "methodology": "Monthly sorts into portfolios.",
        "statistical_limitations": ["Sample ends in 2011."],
        "implementation_variants": ["Industry-neutral ranking."],
        "lookahead_concerns": [],
        "missing_information": [],
        "author_claims": ["Value earns a premium."],
        "model_interpretations": ["Likely stronger among small caps."],
        "proposed_hypotheses": ["Combine with sector neutralisation on BRAIN."],
        "evidence": [
            {"section_id": "s3", "page": 1, "quote": quote, "supports": "definition of the signal"}
        ],
    }
    return base | over


def extraction(*ideas: dict[str, Any], **over: Any) -> dict[str, Any]:
    return {
        "paper_summary": "Value and momentum premia exist across markets.",
        "methodology_summary": "Monthly portfolio sorts on Compustat and CRSP data.",
        "ideas": list(ideas),
        "no_ideas_reason": None,
    } | over


def two_ideas() -> dict[str, Any]:
    return extraction(
        idea("Value premium from book-to-market", VALUE_QUOTE),
        idea(
            "Momentum with a skipped month",
            MOMENTUM_QUOTE,
            topic="momentum",
            variables=[variable("12-month return", ["close price", "returns"], kind="price")],
            operators_needed=["ts_delta", "rank"],
            signal_definition="Rank by return over the prior 12 months, skipping the last month.",
            author_claims=["Momentum earns a premium."],
        ),
    )


# --- registries --------------------------------------------------------------------------


def crossref_body(doi: str = DOI) -> dict[str, Any]:
    return {
        "message": {
            "DOI": doi.upper(),
            "title": ["Value and Momentum Everywhere"],
            "author": [
                {"given": "Clifford S.", "family": "Asness"},
                {"given": "Tobias J.", "family": "Moskowitz"},
            ],
            "issued": {"date-parts": [[2013, 6]]},
            "container-title": ["The Journal of Finance"],
            "URL": f"https://doi.org/{doi}",
            "subject": ["Finance"],
            "is-referenced-by-count": 3000,
        }
    }


def openalex_body(*, open_pdf: str | None = None, doi: str = DOI) -> dict[str, Any]:
    words = ABSTRACT.split()
    index: dict[str, list[int]] = {}
    for position, word in enumerate(words):
        index.setdefault(word, []).append(position)
    return {
        "title": "Value and Momentum Everywhere",
        "doi": f"https://doi.org/{doi}",
        "publication_date": "2013-06-01",
        "abstract_inverted_index": index,
        "authorships": [{"author": {"display_name": "Clifford S. Asness"}}],
        "open_access": {"is_oa": open_pdf is not None, "oa_status": "green"},
        "best_oa_location": {
            "pdf_url": open_pdf,
            "license": "cc-by",
            "landing_page_url": "https://x.org/p",
        },
        "primary_location": {"source": {"display_name": "The Journal of Finance"}},
        "topics": [{"display_name": "Asset pricing"}],
        "cited_by_count": 3000,
    }


ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <title>Cross-sectional   Momentum in Equities</title>
    <summary>We study momentum across 40 countries.</summary>
    <published>2023-01-29T00:00:00Z</published>
    <author><name>A. Researcher</name></author>
    <category term="q-fin.PM"/>
  </entry>
</feed>"""


@dataclass
class Registries:
    """Answers the registries' URLs from a table; records what was asked."""

    open_pdf: str | None = None
    down: bool = False
    calls: list[str] = field(default_factory=list)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        self.calls.append(url)
        if self.down:
            raise httpx2.ConnectError("offline", request=request)
        if "api.crossref.org/works/" in url:
            if "10.9999" in url:
                return httpx2.Response(404)
            return httpx2.Response(200, json=crossref_body())
        if "api.openalex.org/works/doi:" in url:
            if "10.9999" in url:
                return httpx2.Response(404)
            return httpx2.Response(200, json=openalex_body(open_pdf=self.open_pdf))
        if "api.openalex.org/works" in url:
            return httpx2.Response(200, json={"results": [openalex_body(open_pdf=self.open_pdf)]})
        if "export.arxiv.org" in url:
            return httpx2.Response(200, text=ARXIV_ATOM)
        return httpx2.Response(404)

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handler)


# --- the field catalogue -----------------------------------------------------------------


def row(field_id: str, description: str = "", **over: Any) -> dict[str, Any]:
    return {
        "field_id": field_id,
        "description": description,
        "field_type": "MATRIX",
        "dataset_id": "fundamental6",
        "coverage": 0.9,
        "alpha_count": 100,
    } | over


@dataclass
class FakeQueries:
    """A catalogue as ``{(region, delay, universe): {search term: [rows]}}``.

    Matches :class:`alpha_harness.research.feasibility.Queries`; the real
    ``CatalogQueries`` is exercised by its own tests against DuckDB.
    """

    data: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]]
    asked: list[tuple[str, str]] = field(default_factory=list)

    async def fields(
        self, scope: Tuple4, filters: FieldFilter, *, counted: bool = True
    ) -> dict[str, Any]:
        key = (scope.region, scope.delay, scope.universe)
        self.asked.append((scope.region, str(filters.search)))
        rows = self.data.get(key, {}).get(str(filters.search), [])
        return {"results": rows[: filters.limit], "total": len(rows)}

    async def fields_anywhere(self, search: str, limit: int) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for tables in self.data.values():
            for r in tables.get(search, []):
                found.setdefault(r["field_id"], r)
        return list(found.values())[:limit]

    async def field_availability(self, field_id: str) -> list[dict[str, Any]]:
        return [
            {"instrument_type": "EQUITY", "region": k[0], "delay": k[1], "universe": k[2]}
            for k, tables in self.data.items()
            if any(r["field_id"] == field_id for rows in tables.values() for r in rows)
        ]

    async def synced_tuples(self) -> list[dict[str, Any]]:
        return [
            {
                "instrument_type": "EQUITY",
                "region": k[0],
                "delay": k[1],
                "universe": k[2],
                "fields": 9,
            }
            for k in self.data
        ]


def usa_catalogue() -> FakeQueries:
    return FakeQueries(
        {
            ("USA", 1, "TOP3000"): {
                "book value": [row("bookvalue_ps", "Book value per share, as reported")],
                "market capitalization": [row("cap", "Market capitalization", dataset_id="pv1")],
                "close price": [row("close", "Daily close price", dataset_id="pv1")],
                "returns": [row("returns", "Daily returns", dataset_id="pv1")],
            },
            ("EUR", 1, "TOP1200"): {
                "book value": [row("bookvalue_ps", "Book value per share, as reported")],
                "earnings surprise": [row("eur_only_surprise", "Surprise vs consensus")],
            },
        }
    )


OPERATORS = [
    {
        "name": "rank",
        "category": "Cross Sectional",
        "scope": ["REGULAR"],
        "definition": "rank(x, rate=2)",
    },
    {
        "name": "ts_delta",
        "category": "Time Series",
        "scope": ["REGULAR"],
        "definition": "ts_delta(x, d)",
    },
    {
        "name": "ts_rank",
        "category": "Time Series",
        "scope": ["REGULAR"],
        "definition": "ts_rank(x, d, constant = 0)",
    },
]


# --- fetching ----------------------------------------------------------------------------


@dataclass
class FakeWeb:
    """What :func:`alpha_harness.research.netfetch.fetch` is replaced by: URL → bytes."""

    pages: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def __call__(self, url: str, **_: Any) -> netfetch.Fetched:
        self.calls.append(url)
        if url not in self.pages:
            raise netfetch.FetchError("not_found", "There is nothing at that address.")
        kind, data = self.pages[url]
        if kind == "denied":
            raise netfetch.FetchError("access_denied", "The server requires a login.")
        return netfetch.Fetched(url, kind, data, 200)


def landing_page(*, pdf_url: str | None = None, doi: str | None = DOI) -> bytes:
    tags = [
        '<meta name="citation_title" content="Value and Momentum Everywhere">',
        '<meta name="citation_author" content="Clifford Asness">',
        f'<meta name="citation_abstract" content="{ABSTRACT}">',
    ]
    if doi:
        tags.append(f'<meta name="citation_doi" content="{doi}">')
    if pdf_url:
        tags.append(f'<meta name="citation_pdf_url" content="{pdf_url}">')
    return (
        f"<html><head><title>x</title>{''.join(tags)}</head><body>Buy now!</body></html>".encode()
    )


def dumps(value: Any) -> str:
    return json.dumps(value)
