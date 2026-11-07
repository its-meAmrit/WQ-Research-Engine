import pytest

from alpha_harness.spc.identifiers import (
    ReferenceData,
    SecurityRecord,
    check_isin,
    check_mapping,
    check_mic,
    isin_check_digit,
    load_reference,
)


@pytest.mark.parametrize(
    "isin", ["US0378331005", "US5949181045", "GB0002634946", "DE000BAY0017", "JP3633400001"]
)
def test_known_real_isins_pass_checksum(isin: str) -> None:
    assert check_isin(isin).status == "checksum_valid"


def test_wrong_check_digit_names_the_right_one() -> None:
    result = check_isin("US0378331006")
    assert result.status == "invalid_checksum"
    assert "should be 5" in result.detail
    assert not result.ok


@pytest.mark.parametrize(
    "bad", ["", "US037833100", "us0378331005", "US03783310055", "1S0378331005"]
)
def test_malformed_isin(bad: str) -> None:
    assert check_isin(bad).status == "invalid_format"


def test_unknown_country_prefix() -> None:
    assert check_isin("QQ0000000009").status == "invalid_format"


def test_placeholder_isin_with_valid_checksum_is_still_rejected() -> None:
    body = "US123456789"
    isin = body + str(isin_check_digit(body))
    assert check_isin(isin).status == "placeholder"


def test_checksum_valid_is_not_called_verified() -> None:
    assert check_isin("US0378331005").status != "verified"


def test_mic_structure_without_reference_is_not_recognised() -> None:
    assert check_mic("XNAS").status == "structure_valid"
    assert check_mic("xnas").status == "invalid_format"
    assert check_mic("XNA").status == "invalid_format"


def test_mic_against_reference() -> None:
    ref = ReferenceData(mics={"XNAS": "ACTIVE", "XOLD": "EXPIRED"})
    assert check_mic("XNAS", ref).status == "recognized"
    assert check_mic("XOLD", ref).status == "inactive"
    assert check_mic("ZZZZ", ref).status == "unrecognized"


def test_mapping_only_verified_by_a_master() -> None:
    assert check_mapping("US0378331005", "XNAS", "AAPL", "Apple Inc.", None) is None
    ref = ReferenceData(
        securities={"US0378331005": SecurityRecord("US0378331005", "XNAS", "AAPL", "Apple Inc.")}
    )
    assert check_mapping("US0378331005", "XNAS", "aapl", "apple inc.", ref).status == "verified"  # pyright: ignore[reportOptionalMemberAccess]
    wrong = check_mapping("US0378331005", "XNYS", "AAPL", "Apple Inc.", ref)
    assert wrong is not None
    assert wrong.status == "mismatch"
    assert "XNAS" in wrong.detail
    assert check_mapping("US5949181045", "XNAS", "MSFT", "Microsoft", ref) is None


def test_reference_files_load_and_bad_ones_are_skipped(tmp_path) -> None:  # noqa: ANN001
    assert load_reference(tmp_path).mics is None
    (tmp_path / "spc_mic_reference.csv").write_text("MIC,STATUS\nXNAS,ACTIVE\nXOLD,DELETED\n")
    (tmp_path / "spc_security_master.csv").write_bytes(b"\xff\xfe\x00bad")
    ref = load_reference(tmp_path)
    assert ref.mics == {"XNAS": "ACTIVE", "XOLD": "DELETED"}
    assert ref.securities is None
