"""One spelling per paper: DOIs, arXiv ids and links reduce to a canonical form."""

import pytest

from alpha_harness.research.identifiers import (
    IdentifierError,
    canonical_url,
    classify,
    normalise_arxiv,
    normalise_doi,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1086/260062", "10.1086/260062"),
        ("doi:10.1086/260062", "10.1086/260062"),
        ("DOI: 10.1086/260062", "10.1086/260062"),
        ("https://doi.org/10.1086/260062", "10.1086/260062"),
        ("https://dx.doi.org/10.1086/260062.", "10.1086/260062"),
        (
            "HTTPS://DOI.ORG/10.1111/J.1540-6261.1992.TB04398.X",
            "10.1111/j.1540-6261.1992.tb04398.x",
        ),
        (
            "https://onlinelibrary.wiley.com/doi/10.1111%2Fj.1540-6261.1992.tb04398.x?utm_source=a",
            "10.1111/j.1540-6261.1992.tb04398.x",
        ),
        ("see (10.1016/j.jfineco.2012.05.001).", "10.1016/j.jfineco.2012.05.001"),
    ],
)
def test_doi_is_one_lowercase_spelling(raw: str, expected: str) -> None:
    assert normalise_doi(raw) == expected


@pytest.mark.parametrize("raw", ["", "hello", "10.1086", "11.1086/x", "doi:"])
def test_not_a_doi(raw: str) -> None:
    assert normalise_doi(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v5", "1706.03762"),
        ("arXiv:1706.03762v2", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v2", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v2.pdf", "1706.03762"),
        ("q-fin/0409001v1", "q-fin/0409001"),
        ("2301.1234", "2301.1234"),
    ],
)
def test_arxiv_id_drops_the_version(raw: str, expected: str) -> None:
    assert normalise_arxiv(raw) == expected


def test_arxiv_rejects_look_alikes() -> None:
    assert normalise_arxiv("2019.1234567") is None
    assert normalise_arxiv("10.1086/260062") is None


def test_url_drops_tracking_fragment_and_default_port() -> None:
    assert (
        canonical_url("HTTPS://Example.com:443/a/b/?utm_campaign=z&id=3#frag")
        == "https://example.com/a/b?id=3"
    )
    assert canonical_url("ftp://x.org/a") is None
    assert canonical_url("not a url") is None


def test_classify_prefers_the_identifier_over_the_link() -> None:
    assert classify("https://doi.org/10.1086/260062").kind == "doi"
    assert classify("https://arxiv.org/abs/1706.03762v3").value == "1706.03762"
    assert (
        classify("https://example.com/paper.pdf?utm_source=x").value
        == "https://example.com/paper.pdf"
    )
    with pytest.raises(IdentifierError):
        classify("   ")
    with pytest.raises(IdentifierError):
        classify("just words")
