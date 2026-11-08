"""The shape of an SPC submission, kept in one place.

**What is confirmed and what is not.** WorldQuant's documentation gives the limits below and
names the data a position carries (company name, ticker, ISIN, MIC, confidence score,
investment horizon, return prediction). It does not, in any text this project could read, give
the machine-readable key names: the support pages were not reachable when this was written
and the example code block in the PDF is clipped. The keys below are therefore **candidates**,
and :data:`PROFILE` says so (``verified=False``) everywhere it is shown. Compare them with the
full official example before submitting, and change them here if they differ: the validator,
the prompt generator and the export all read this one object.
"""

from dataclasses import dataclass

#: Characters a prompt may hold.
PROMPT_MAX_CHARS = 10_000
#: Characters the optional model version may hold.
MODEL_VERSION_MAX_CHARS = 100
#: Prompts with a non-zero weight a consultant may have active at once.
MAX_ACTIVE_PROMPTS = 50
#: Prompts a consultant may submit in one day, zero-weight ones included. A WorldQuant
#: limit on submissions: local drafts and test runs are not counted against it.
MAX_DAILY_SUBMISSIONS = 50
#: Suggested equities per prompt. A guideline, not a schema rule.
DEFAULT_TARGET_EQUITIES = 50

FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")

#: Model families the documentation names. Informational: the model is whichever the user set
#: up under API Integration, and this list is not used to restrict that choice.
DOCUMENTED_MODEL_FAMILIES = (
    "Claude",
    "GPT",
    "Gemini",
    "DeepSeek",
    "Kimi",
    "Qwen3",
    "GLM",
    "Llama",
    "Minimax",
    "Mistral",
)


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    """The keys of one position object, and whether they were checked against the source."""

    company_name: str
    ticker: str
    isin: str
    mic: str
    confidence_score: str
    investment_horizon: str
    return_prediction: str
    verified: bool
    note: str

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            self.company_name,
            self.ticker,
            self.isin,
            self.mic,
            self.confidence_score,
            self.investment_horizon,
            self.return_prediction,
        )

    @property
    def text_keys(self) -> tuple[str, ...]:
        return (self.company_name, self.ticker, self.isin, self.mic)

    @property
    def number_keys(self) -> tuple[str, ...]:
        return (self.confidence_score, self.investment_horizon, self.return_prediction)


PROFILE = SchemaProfile(
    company_name="company_name",
    ticker="ticker",
    isin="isin",
    mic="mic",
    confidence_score="confidence_score",
    investment_horizon="investment_horizon",
    return_prediction="return_prediction",
    verified=False,
    note=(
        "These key names are candidates, not confirmed WorldQuant requirements: the official "
        "example could not be read in full. Compare them with the complete example in "
        "WorldQuant's 'What do I need to submit' article before you submit."
    ),
)
