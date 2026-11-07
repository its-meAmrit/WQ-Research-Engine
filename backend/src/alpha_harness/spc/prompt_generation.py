"""Turning an investment idea and a few choices into prompts ready to paste into SPC.

Pure text assembly: no model call, no network, no hidden state. A generated prompt stands
alone, because WorldQuant runs it without any of this application around it. Each approach
gives a different emphasis to the same skeleton, and every prompt is held to the platform's
character limit, dropping optional detail before it would cut the output instructions.
"""

from typing import Literal

from pydantic import Field

from ..schemas import Out
from .schema import DEFAULT_TARGET_EQUITIES, FREQUENCIES, PROFILE, PROMPT_MAX_CHARS

Direction = Literal["long_short", "long_only", "short_only"]
ReturnUnit = Literal["fraction", "percent"]

APPROACHES: dict[str, tuple[str, str]] = {
    "fundamental_quality": (
        "Fundamental quality",
        "Favour companies with durable profitability, strong balance sheets, high returns on "
        "invested capital, stable or growing free cash flow and conservative leverage. Short "
        "companies with deteriorating margins, weak cash conversion, heavy refinancing risk or "
        "aggressive accounting.",
    ),
    "valuation": (
        "Valuation",
        "Compare forward earnings, cash-flow and enterprise-value multiples with each "
        "company's own history, its peers and its expected growth. Go long where expectations "
        "look too low for the business quality, and short where the price assumes growth or "
        "margins the evidence does not support.",
    ),
    "sector_specific": (
        "Sector-specific",
        "Use the metrics that decide results in this sector (for example capacity, pricing, "
        "backlog, unit economics or regulatory position). Rank companies against sector peers "
        "on those drivers, not on generic screens.",
    ),
    "macro_sensitive": (
        "Macro-sensitive",
        "Identify the interest-rate, inflation, currency, commodity and growth exposures that "
        "matter to this theme. Go long companies whose earnings benefit from the current "
        "macro backdrop and short those most exposed to its reversal.",
    ),
    "event_driven": (
        "Event-driven",
        "Focus on dated catalysts: earnings, guidance changes, product launches, regulatory "
        "decisions, index changes, restructurings and capital actions. Size each position to "
        "the catalyst's likely effect and set the horizon to when it resolves.",
    ),
    "news_driven": (
        "News-driven",
        "Weigh the most recent company-specific and sector news and what it changes in "
        "expectations. Separate lasting information from noise, and discount news already "
        "reflected in the share price.",
    ),
    "hybrid": (
        "Hybrid",
        "Combine business quality, valuation and near-term evidence. A position needs support "
        "from more than one of these, and conviction should rise only when they agree.",
    ),
}


class GenerationOptions(Out):
    """What Prompt Studio asks for. Every field has a default, so a thesis alone is enough."""

    thesis: str = Field(min_length=1, max_length=4000)
    universe: str = Field(default="Global listed equities", max_length=200)
    sector: str = Field(default="", max_length=200)
    direction: Direction = "long_short"
    target_equities: int = Field(default=DEFAULT_TARGET_EQUITIES, ge=5, le=200)
    #: The days a typical position is expected to take to realise its predicted return.
    horizon_days: int = Field(default=30, ge=1, le=730)
    update_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "weekly"
    #: Newest evidence the prompt asks the model to prefer, in days.
    freshness_days: int = Field(default=90, ge=1, le=1095)
    max_sector_percent: int = Field(default=30, ge=5, le=100)
    max_country_percent: int = Field(default=40, ge=5, le=100)
    #: Extra constraints in the consultant's own words.
    constraints: str = Field(default="", max_length=1500)
    #: The platform does not say in any text read here whether a return is a fraction or a
    #: percentage, so the prompt states one and the consultant can change it.
    return_unit: ReturnUnit = "fraction"
    approaches: list[str] = Field(default_factory=lambda: ["hybrid"], min_length=1, max_length=7)


class PromptCandidate(Out):
    approach: str
    label: str
    prompt_text: str
    characters: int
    warnings: list[str]


def _direction_rules(direction: Direction) -> str:
    if direction == "long_only":
        return (
            "Take long positions only. Every confidence_score must be greater than 0. "
            "Do not include short positions."
        )
    if direction == "short_only":
        return (
            "Take short positions only. Every confidence_score must be less than 0. "
            "Do not include long positions."
        )
    return (
        "Take both long and short positions. A positive confidence_score means long and a "
        "negative one means short; the larger its absolute value, the stronger your "
        "conviction. Use the whole -1 to +1 range, and never output a score of 0."
    )


def _output_block(options: GenerationOptions) -> str:
    p = PROFILE
    unit = (
        "a decimal fraction (0.05 means 5%)"
        if options.return_unit == "fraction"
        else "a percentage (5 means 5%)"
    )
    return (
        "OUTPUT FORMAT\n"
        "Return only a JSON array: the first character of your answer is [ and the last is ]. "
        "Write no text, explanation, headings or markdown code fences before or after it. "
        "Each element is one object with exactly these keys and no others:\n"
        "[\n"
        "  {\n"
        f'    "{p.company_name}": "<registered company name>",\n'
        f'    "{p.ticker}": "<ticker symbol on the listing exchange>",\n'
        f'    "{p.isin}": "<12-character ISIN>",\n'
        f'    "{p.mic}": "<4-character ISO 10383 market identifier code of the listing>",\n'
        f'    "{p.confidence_score}": <number from -1 to 1>,\n'
        f'    "{p.investment_horizon}": <positive number of days until the return is expected>,\n'
        f'    "{p.return_prediction}": <positive number, the size of the expected return as '
        f"{unit}>\n"
        "  }\n"
        "]\n"
        f"The sign of {p.confidence_score} gives the direction; {p.return_prediction} is "
        "always positive. List each security once."
    )


def _research_block(options: GenerationOptions) -> str:
    return (
        "RESEARCH STANDARDS\n"
        f"- Prefer evidence from the last {options.freshness_days} days. Where you can search "
        "the web, use primary sources first: company filings and investor-relations releases, "
        "stock-exchange disclosures, audited financial statements, government statistics, "
        "then credible financial news.\n"
        "- Verify every claim you rely on. If you cannot verify a figure, do not use it.\n"
        "- Never invent a ticker, ISIN, MIC, price, financial figure or source. If you are "
        "not certain of a company's ISIN and MIC, leave that company out.\n"
        "- It is better to return fewer positions than to include one whose identifiers or "
        "facts you cannot confirm."
    )


def _build(options: GenerationOptions, approach: str, *, compact: bool) -> str:
    _, emphasis = APPROACHES[approach]
    scope = options.universe + (f", focused on {options.sector}" if options.sector else "")
    parts = [
        "ROLE\nYou are a senior global equity research analyst and portfolio manager who "
        "builds disciplined, evidence-based long/short portfolios.",
        f"OBJECTIVE\n{options.thesis.strip()}",
        f"UNIVERSE\nSelect only listed equities within: {scope}. Choose around "
        f"{options.target_equities} securities. Exclude companies you cannot identify "
        "precisely, and any security that trades too thinly to hold.",
        f"SELECTION APPROACH\n{emphasis}",
        "LONG AND SHORT LOGIC\n" + _direction_rules(options.direction),
        _research_block(options),
    ]
    if not compact:
        constraints = [
            f"No single sector above {options.max_sector_percent}% of the portfolio's total "
            "absolute confidence.",
            f"No single country above {options.max_country_percent}% of it.",
            "Spread conviction: avoid one position dominating the total.",
        ]
        if options.constraints.strip():
            constraints.append(options.constraints.strip())
        parts.append("PORTFOLIO CONSTRAINTS\n" + "\n".join(f"- {c}" for c in constraints))
        parts.append(
            "HORIZON AND UPDATES\n"
            f"Set investment_horizon to the days you expect each prediction to take to play "
            f"out; around {options.horizon_days} days is typical for this thesis. This "
            f"portfolio is refreshed {options.update_frequency}, so choose positions that "
            "make sense between refreshes."
        )
    parts.append(_output_block(options))
    return "\n\n".join(parts)


def generate(options: GenerationOptions) -> list[PromptCandidate]:
    """One candidate per requested approach, each within the platform's character limit."""
    out: list[PromptCandidate] = []
    for approach in dict.fromkeys(options.approaches):
        if approach not in APPROACHES:
            raise ValueError(f"{approach!r} is not an approach: {', '.join(APPROACHES)}.")
        warnings: list[str] = []
        text = _build(options, approach, compact=False)
        if len(text) > PROMPT_MAX_CHARS:
            text = _build(options, approach, compact=True)
            warnings.append(
                "Too long for the platform's limit, so the portfolio-constraints and "
                "horizon sections were dropped. Shorten the thesis or extra constraints."
            )
        if len(text) > PROMPT_MAX_CHARS:
            warnings.append(
                f"Still {len(text):,} characters; the platform accepts {PROMPT_MAX_CHARS:,}. "
                "Shorten the thesis."
            )
        if options.update_frequency not in FREQUENCIES:
            warnings.append("Unknown update frequency.")
        out.append(
            PromptCandidate(
                approach=approach,
                label=APPROACHES[approach][0],
                prompt_text=text,
                characters=len(text),
                warnings=warnings,
            )
        )
    return out
