import json
from typing import Any

import pytest

from alpha_harness.spc.identifiers import ReferenceData, SecurityRecord
from alpha_harness.spc.validation import validate_output

from .conftest import GOOD, position


def codes(raw: str, **kw: Any) -> list[str]:
    return [i.code for i in validate_output(raw, **kw).issues]


def run(items: Any) -> Any:
    return validate_output(json.dumps(items))


def test_valid_portfolio() -> None:
    report = run(GOOD)
    assert report.valid
    assert report.item_count == 3
    assert [p.ticker for p in report.positions] == ["AAPL", "MSFT", "BAYN"]
    assert report.positions[0].isin_status == "checksum_valid"
    assert report.positions[0].mapping_status == "unverified"
    assert report.schema_verified is False


def test_sample_is_the_models_text_not_a_reformat() -> None:
    raw = (
        '[ {"company_name":"Apple Inc.",  "ticker":"AAPL","isin":"US0378331005",'
        '"mic":"XNAS","confidence_score":0.5,"investment_horizon":10,'
        '"return_prediction":0.02} ]'
    )
    report = validate_output(raw)
    assert report.valid
    assert report.sample_json == raw


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("", "empty_output"),
        ("   \n", "empty_output"),
        ('{"positions": []}', "top_level_not_array"),
        ("Here is the portfolio: [1]", "prose_outside_json"),
        ("no json at all", "no_json_found"),
        ("[", "no_json_found"),
        ("[{]", "invalid_json"),
        ("[NaN]", "invalid_json"),
        ("[Infinity]", "invalid_json"),
        ("[]", "empty_array"),
        ("[1, 2]", "item_not_object"),
    ],
)
def test_structural_failures(raw: str, code: str) -> None:
    report = validate_output(raw)
    assert not report.valid
    assert code in [i.code for i in report.issues]


def test_trailing_prose_is_an_error_not_trimmed() -> None:
    report = validate_output(json.dumps(GOOD) + "\nHope this helps!")
    assert not report.valid
    assert report.sample_json is None


def test_code_fence_is_a_warning_and_still_validates() -> None:
    report = validate_output("```json\n" + json.dumps(GOOD) + "\n```")
    assert report.valid
    assert "wrapped_in_code_fence" in [i.code for i in report.warnings]
    assert report.sample_json is not None
    assert report.sample_json.startswith("[")


@pytest.mark.parametrize(
    ("change", "code", "field"),
    [
        ({"confidence_score": 1.01}, "out_of_range", "confidence_score"),
        ({"confidence_score": -1.5}, "out_of_range", "confidence_score"),
        ({"investment_horizon": 0}, "out_of_range", "investment_horizon"),
        ({"investment_horizon": -5}, "out_of_range", "investment_horizon"),
        ({"return_prediction": 0}, "out_of_range", "return_prediction"),
        ({"return_prediction": -0.03}, "out_of_range", "return_prediction"),
        ({"confidence_score": "0.5"}, "wrong_type", "confidence_score"),
        ({"confidence_score": True}, "wrong_type", "confidence_score"),
        ({"investment_horizon": None}, "wrong_type", "investment_horizon"),
        ({"ticker": 123}, "wrong_type", "ticker"),
        ({"ticker": ""}, "placeholder_value", "ticker"),
        ({"company_name": "N/A"}, "placeholder_value", "company_name"),
        ({"isin": "US0378331006"}, "invalid_isin", "isin"),
        ({"isin": "US123456789" + "0"}, "invalid_isin", "isin"),
        ({"mic": "xnas"}, "invalid_mic", "mic"),
        ({"mic": "XNASD"}, "invalid_mic", "mic"),
    ],
)
def test_field_failures_say_where(change: dict[str, Any], code: str, field: str) -> None:
    report = run([position(), position(**({"isin": "US5949181045"} | change))])
    assert not report.valid
    hit = [i for i in report.errors if i.index == 1 and i.field == field]
    assert hit, report.issues
    assert any(i.code in {code, "placeholder_isin"} for i in hit)
    assert all(i.message for i in hit)


def test_boundary_values_are_allowed() -> None:
    items = [
        position(confidence_score=1, investment_horizon=0.5),
        position(isin="US5949181045", ticker="MSFT", confidence_score=-1),
    ]
    assert run(items).valid


def test_zero_confidence_warns_but_passes() -> None:
    report = run([position(confidence_score=0)])
    assert report.valid
    assert "zero_confidence" in [i.code for i in report.warnings]


def test_missing_and_unexpected_fields() -> None:
    item = position()
    del item["mic"]
    item["notes"] = "extra"
    report = run([item])
    assert not report.valid
    found = {(i.code, i.field) for i in report.errors}
    assert ("missing_field", "mic") in found
    assert ("unexpected_field", "notes") in found


def test_duplicate_and_conflicting_securities() -> None:
    dup = run([position(), position(confidence_score=0.2)])
    assert not dup.valid
    assert "duplicate_security" in [i.code for i in dup.errors]
    conflict = run([position(), position(confidence_score=-0.2)])
    assert "conflicting_positions" in [i.code for i in conflict.errors]


def test_same_ticker_under_another_isin_is_only_a_warning() -> None:
    report = run([position(), position(isin="US5949181045")])
    assert report.valid
    assert "duplicate_listing" in [i.code for i in report.warnings]


def test_unknown_mic_blocks_only_when_a_reference_is_loaded() -> None:
    items = [position(mic="ZZZZ")]
    assert run(items).valid
    ref = ReferenceData(mics={"XNAS": "ACTIVE"})
    report = validate_output(json.dumps(items), reference=ref)
    assert not report.valid
    assert "invalid_mic" in [i.code for i in report.errors]


def test_security_master_mismatch_blocks_and_match_verifies() -> None:
    master = ReferenceData(
        securities={"US0378331005": SecurityRecord("US0378331005", "XNAS", "AAPL", "Apple Inc.")}
    )
    ok = validate_output(json.dumps([position()]), reference=master)
    assert ok.valid
    assert ok.positions[0].mapping_status == "verified"
    bad = validate_output(json.dumps([position(ticker="MSFT")]), reference=master)
    assert not bad.valid
    assert "identifier_mismatch" in [i.code for i in bad.errors]


def test_invalid_entries_do_not_leak_into_positions() -> None:
    report = run([position(), position(isin="bad")])
    assert not report.valid
    assert report.positions == []


def test_report_serialises() -> None:
    d = run([position(confidence_score=9)]).to_dict()
    assert d["valid"] is False
    assert d["errorCount"] == 1
    assert {"severity", "code", "message", "index", "field", "suggestion"} <= d["issues"][0].keys()
    json.dumps(d)


def test_never_raises_on_hostile_input() -> None:
    for raw in ["[" * 5000 + "]" * 5000, "[\x00]", "[" + "9" * 400 + "]", '[{"a":' * 50]:
        validate_output(raw)
