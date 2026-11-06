"""Checking a model's answer against the SPC portfolio schema.

Valid JSON is the first check of many. The answer must be a bare top-level array, every
position must carry exactly the schema's keys with the right types and ranges, identifiers
must be well formed, and no security may appear twice. Nothing is repaired: a value that fails
is reported with where it is and what to do, and the original text is kept as it came.
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .identifiers import ReferenceData, check_isin, check_mapping, check_mic
from .schema import PROFILE, SchemaProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

ERROR = "error"
WARNING = "warning"

#: Names a model writes when it has nothing real to put there.
_PLACEHOLDERS = frozenset(
    {"", "n/a", "na", "none", "null", "tbd", "unknown", "example", "placeholder", "xxx", "-"}
)
_TICKER = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-_/ ]{0,19}")
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Issue:
    """One thing wrong, or worth knowing, and where it is."""

    severity: str
    code: str
    message: str
    index: int | None = None
    field: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "index": self.index,
            "field": self.field,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPosition:
    """A position that passed every blocking check, and how far its identifiers were checked."""

    index: int
    company_name: str
    ticker: str
    isin: str
    mic: str
    confidence_score: float
    investment_horizon: float
    return_prediction: float
    isin_status: str
    mic_status: str
    #: ``verified`` / ``mismatch`` when a user-supplied security master covers the ISIN, else
    #: ``unverified``. A good checksum alone is never "verified".
    mapping_status: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "companyName": self.company_name,
            "ticker": self.ticker,
            "isin": self.isin,
            "mic": self.mic,
            "confidenceScore": self.confidence_score,
            "investmentHorizon": self.investment_horizon,
            "returnPrediction": self.return_prediction,
            "isinStatus": self.isin_status,
            "micStatus": self.mic_status,
            "mappingStatus": self.mapping_status,
        }


@dataclass(slots=True)
class ValidationReport:
    """Everything the validator found."""

    #: No blocking error: the answer can be exported as a tested sample.
    valid: bool
    issues: list[Issue] = field(default_factory=list)
    positions: list[ValidatedPosition] = field(default_factory=list)
    #: The JSON array text taken out of the answer: a substring of it, never reformatted.
    sample_json: str | None = None
    item_count: int = 0
    schema_verified: bool = False

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "itemCount": self.item_count,
            "schemaVerified": self.schema_verified,
            "errorCount": len(self.errors),
            "warningCount": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


def _error(code: str, message: str, **kw: Any) -> Issue:
    return Issue(ERROR, code, message, **kw)


def _warning(code: str, message: str, **kw: Any) -> Issue:
    return Issue(WARNING, code, message, **kw)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON; numbers must be finite")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def extract_array(raw: str) -> tuple[str | None, list[Issue]]:
    """The JSON array text inside an answer, or why there is none.

    Only two shapes are taken: the answer itself, or the answer wrapped whole in a code fence
    (a warning, since the platform wants bare JSON). Prose around an array is an error, not
    something quietly cut away, because the platform would see the prose too.
    """
    text = raw.strip()
    if not text:
        return None, [
            _error(
                "empty_output",
                "The model returned nothing.",
                suggestion="Run it again, or choose a model with a larger output limit.",
            )
        ]
    issues: list[Issue] = []
    fenced = _FENCE.fullmatch(text)
    if fenced:
        inner = fenced.group(1).strip()
        issues.append(
            _warning(
                "wrapped_in_code_fence",
                "The answer was wrapped in a markdown code fence. The JSON inside is used as "
                "the sample, but the platform expects bare JSON.",
                suggestion="Add 'no markdown, no code fences' to the prompt.",
            )
        )
        text = inner
    if text.startswith("[") and text.endswith("]"):
        return text, issues
    if text.startswith("{"):
        issues.append(
            _error(
                "top_level_not_array",
                "The answer is a JSON object. SPC needs a top-level array of positions.",
                suggestion="Tell the prompt to output a bare array, not an object wrapping one.",
            )
        )
        return None, issues
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        issues.append(
            _error(
                "prose_outside_json",
                "There is text before or after the JSON array. The platform reads the whole "
                "answer, so anything outside the array breaks it.",
                suggestion="Add 'output only the JSON array, with no other text' to the prompt.",
            )
        )
    else:
        issues.append(
            _error(
                "no_json_found",
                "No JSON array was found in the answer.",
                suggestion="Check the prompt asks for the array, and see the raw output.",
            )
        )
    return None, issues


def _text(value: object, key: str, index: int, issues: list[Issue]) -> str | None:
    if not isinstance(value, str):
        issues.append(
            _error(
                "wrong_type",
                f"{key} must be a string.",
                index=index,
                field=key,
                suggestion="Wrap the value in quotes.",
            )
        )
        return None
    stripped = value.strip()
    if stripped != value:
        issues.append(
            _warning(
                "whitespace_trimmed",
                f"{key} has spaces at its ends.",
                index=index,
                field=key,
            )
        )
    if stripped.casefold() in _PLACEHOLDERS:
        issues.append(
            _error(
                "placeholder_value",
                f"{key} is empty or a placeholder.",
                index=index,
                field=key,
                suggestion="Every position needs a real value; drop it if one is not known.",
            )
        )
        return None
    return stripped


def _number(value: object, key: str, index: int, issues: list[Issue]) -> float | None:
    if not _is_number(value):
        issues.append(
            _error(
                "wrong_type",
                f"{key} must be a number.",
                index=index,
                field=key,
                suggestion="Use a bare number, not a string.",
            )
        )
        return None
    number = float(value)  # pyright: ignore[reportArgumentType]
    if not math.isfinite(number):
        issues.append(_error("non_finite_number", f"{key} is not finite.", index=index, field=key))
        return None
    return number


def _position(
    item: object,
    index: int,
    profile: SchemaProfile,
    ref: ReferenceData | None,
    issues: list[Issue],
) -> ValidatedPosition | None:
    if not isinstance(item, dict):
        issues.append(
            _error(
                "item_not_object",
                "Every array entry must be a JSON object describing one position.",
                index=index,
            )
        )
        return None
    before = len([i for i in issues if i.severity == ERROR])
    missing = [k for k in profile.keys if k not in item]
    extra = [k for k in item if k not in profile.keys]
    issues.extend(
        _error(
            "missing_field",
            f"{key} is missing.",
            index=index,
            field=key,
            suggestion=f"Add {key} to every position.",
        )
        for key in missing
    )
    issues.extend(
        _error(
            "unexpected_field",
            f"{key} is not part of the submission schema.",
            index=index,
            field=str(key),
            suggestion="Remove it; the platform accepts only the documented fields.",
        )
        for key in extra
    )
    texts = {k: _text(item[k], k, index, issues) for k in profile.text_keys if k in item}
    nums = {k: _number(item[k], k, index, issues) for k in profile.number_keys if k in item}

    confidence = nums.get(profile.confidence_score)
    horizon = nums.get(profile.investment_horizon)
    prediction = nums.get(profile.return_prediction)
    if confidence is not None:
        if not -1.0 <= confidence <= 1.0:
            issues.append(
                _error(
                    "out_of_range",
                    f"{profile.confidence_score} is {confidence:g}; it must be between -1 and 1.",
                    index=index,
                    field=profile.confidence_score,
                    suggestion="Positive is long, negative is short.",
                )
            )
        elif confidence == 0:
            issues.append(
                _warning(
                    "zero_confidence",
                    "A confidence score of 0 means no position.",
                    index=index,
                    field=profile.confidence_score,
                    suggestion="Remove the entry rather than hold a position of zero.",
                )
            )
    if horizon is not None and horizon <= 0:
        issues.append(
            _error(
                "out_of_range",
                f"{profile.investment_horizon} is {horizon:g}; it must be above 0 days.",
                index=index,
                field=profile.investment_horizon,
            )
        )
    if prediction is not None and prediction <= 0:
        issues.append(
            _error(
                "out_of_range",
                f"{profile.return_prediction} is {prediction:g}; it must be above 0. "
                "Direction comes from the sign of the confidence score.",
                index=index,
                field=profile.return_prediction,
            )
        )

    isin, mic = texts.get(profile.isin), texts.get(profile.mic)
    ticker = texts.get(profile.ticker)
    name = texts.get(profile.company_name)
    isin_check = check_isin(isin) if isin is not None else None
    mic_check = check_mic(mic, ref) if mic is not None else None
    if isin_check is not None and not isin_check.ok:
        issues.append(
            _error(
                "invalid_isin" if isin_check.status != "placeholder" else "placeholder_isin",
                f"{isin}: {isin_check.detail}",
                index=index,
                field=profile.isin,
                suggestion="Look the ISIN up from the issuer or exchange; never guess one.",
            )
        )
    if mic_check is not None and mic_check.status in {"invalid_format", "unrecognized", "inactive"}:
        issues.append(
            _error(
                "invalid_mic",
                f"{mic}: {mic_check.detail}",
                index=index,
                field=profile.mic,
                suggestion="Use the ISO 10383 operating MIC of the listing venue.",
            )
        )
    if ticker is not None and not _TICKER.fullmatch(ticker):
        issues.append(
            _warning(
                "unusual_ticker",
                f"{ticker!r} does not look like a ticker.",
                index=index,
                field=profile.ticker,
            )
        )
    mapping = None
    if isin_check and isin_check.ok and isin and mic and ticker and name:
        mapping = check_mapping(isin, mic, ticker, name, ref)
        if mapping is not None and mapping.status == "mismatch":
            issues.append(
                _error(
                    "identifier_mismatch",
                    mapping.detail,
                    index=index,
                    field=profile.isin,
                    suggestion="The identifiers disagree; fix whichever one is wrong.",
                )
            )
    if len([i for i in issues if i.severity == ERROR]) > before:
        return None
    assert None not in (name, ticker, isin, mic, confidence, horizon, prediction)  # noqa: S101
    assert isin_check is not None and mic_check is not None  # noqa: S101
    return ValidatedPosition(
        index=index,
        company_name=name or "",
        ticker=ticker or "",
        isin=isin or "",
        mic=mic or "",
        confidence_score=confidence or 0.0,
        investment_horizon=horizon or 0.0,
        return_prediction=prediction or 0.0,
        isin_status=isin_check.status,
        mic_status=mic_check.status,
        mapping_status=mapping.status if mapping else "unverified",
    )


def _duplicates(rows: Sequence[ValidatedPosition], issues: list[Issue]) -> bool:
    """Flag a security listed twice. Returns whether any duplicate was blocking."""
    blocked = False
    seen: dict[str, ValidatedPosition] = {}
    listings: dict[tuple[str, str], ValidatedPosition] = {}
    for row in rows:
        first = seen.get(row.isin)
        if first is not None:
            opposite = (first.confidence_score > 0) != (row.confidence_score > 0)
            issues.append(
                _error(
                    "conflicting_positions" if opposite else "duplicate_security",
                    f"{row.isin} already appears at entry {first.index}"
                    + (" with the opposite direction." if opposite else "."),
                    index=row.index,
                    field=PROFILE.isin,
                    suggestion="List each security once, with a single net confidence score.",
                )
            )
            blocked = True
        else:
            seen[row.isin] = row
        key = (row.ticker.casefold(), row.mic)
        earlier = listings.get(key)
        if earlier is not None and earlier.isin != row.isin:
            issues.append(
                _warning(
                    "duplicate_listing",
                    f"{row.ticker} on {row.mic} is also entry {earlier.index} under a different "
                    "ISIN.",
                    index=row.index,
                    field=PROFILE.ticker,
                )
            )
        else:
            listings.setdefault(key, row)
    return blocked


def validate_output(
    raw: str, *, profile: SchemaProfile = PROFILE, reference: ReferenceData | None = None
) -> ValidationReport:
    """Validate one model answer. Never raises on bad input: every problem is an issue."""
    sample, issues = extract_array(raw)
    report = ValidationReport(valid=False, issues=issues, schema_verified=profile.verified)
    if sample is None:
        return report
    try:
        parsed = json.loads(sample, parse_constant=_reject_constant)
    except ValueError as exc:
        where = ""
        if isinstance(exc, json.JSONDecodeError):
            where = f" (line {exc.lineno}, column {exc.colno})"
        issues.append(
            _error(
                "invalid_json",
                f"The answer is not valid JSON: {exc.args[0] if exc.args else exc}{where}.",
                suggestion="Look for a trailing comma, a missing quote, or a truncated answer.",
            )
        )
        return report
    if not isinstance(parsed, list):
        issues.append(_error("top_level_not_array", "The JSON is not a top-level array."))
        return report
    report.item_count = len(parsed)
    if not parsed:
        issues.append(
            _error(
                "empty_array",
                "The array holds no positions.",
                suggestion="The prompt must lead to a portfolio of securities.",
            )
        )
        return report

    rows: list[ValidatedPosition] = []
    for index, item in enumerate(parsed):
        found = _position(item, index, profile, reference, issues)
        if found is not None:
            rows.append(found)
    blocked = _duplicates(rows, issues)
    clean = not any(i.severity == ERROR for i in issues)
    report.valid = clean and not blocked
    report.sample_json = sample
    if report.valid:
        report.positions = rows
    return report
