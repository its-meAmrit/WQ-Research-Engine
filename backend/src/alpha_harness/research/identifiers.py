"""Naming a paper the same way every time, so the same paper is never stored twice.

A DOI arrives as ``10.1086/260062``, ``doi:10.1086/260062``, ``https://doi.org/10.1086/260062``
or inside a publisher URL, in any case; an arXiv paper as ``1706.03762v5``,
``arXiv:1706.03762`` or an ``arxiv.org/abs`` link; a page as a URL with tracking parameters.
Each is reduced here to one canonical spelling, and :func:`classify` says which it is.
"""

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

#: Crossref's own guidance for matching modern DOIs, loosened to the whole suffix.
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_DOI_PREFIXES = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|urn:doi:)", re.IGNORECASE)
#: Characters a DOI never ends with but a sentence around it does.
_TRAILING = ".,;:)]}>'\""

_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
_ARXIV_OLD = re.compile(
    r"(?<![\w-])([a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?!\d)", re.IGNORECASE
)
_ARXIV_PREFIX = re.compile(r"^(?:arxiv\s*:\s*)", re.IGNORECASE)

#: Query parameters that name where a visitor came from, never which page it is.
_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|igshid$|_hs)", re.IGNORECASE)

MAX_INPUT_CHARS = 2_048


class IdentifierError(ValueError):
    """Text that is not a DOI, an arXiv id or a URL."""


def normalise_doi(raw: str) -> str | None:
    """The DOI in ``raw``, lower-cased and bare, or ``None`` when there is none.

    DOIs are case-insensitive, so lower case is the one stored spelling. Percent-escapes in a
    URL are undone first: ``10.1002%2F...`` is the same DOI.
    """
    text = unquote(raw.strip())[:MAX_INPUT_CHARS]
    text = _DOI_PREFIXES.sub("", text)
    match = _DOI.search(text)
    if match is None:
        return None
    doi = match.group(0).rstrip(_TRAILING)
    # A query string or a fragment belongs to the page, not the DOI.
    doi = re.split(r"[?#]", doi, maxsplit=1)[0]
    return doi.lower() if "/" in doi else None


def normalise_arxiv(raw: str) -> str | None:
    """The arXiv id in ``raw`` without its version, or ``None``.

    The version is dropped on purpose: ``v1`` and ``v5`` are one paper, and importing both
    would store it twice. The id keeps its old-style category (``q-fin/0409001``) when it has
    one.
    """
    text = raw.strip()[:MAX_INPUT_CHARS]
    if "arxiv.org" in text.lower():
        path = urlsplit(text if "//" in text else f"//{text}").path
        text = re.sub(r"^/(?:abs|pdf|html)/", "", path).removesuffix(".pdf").strip("/")
    text = _ARXIV_PREFIX.sub("", text)
    if (new := _ARXIV_NEW.fullmatch(text)) is not None:
        return new.group(1)
    if (old := _ARXIV_OLD.fullmatch(text)) is not None:
        return old.group(1).lower()
    return None


def canonical_url(raw: str) -> str | None:
    """A URL in one spelling: lower-case scheme and host, no fragment, no tracking parameters,
    no default port, no trailing slash. ``None`` when it is not an http(s) URL."""
    try:
        parts = urlsplit(raw.strip()[:MAX_INPUT_CHARS])
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not host:
        return None
    scheme = parts.scheme.lower()
    netloc = host if port in (None, 80 if scheme == "http" else 443) else f"{host}:{port}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)))
    path = parts.path.rstrip("/") or ""
    return urlunsplit((scheme, netloc, path, query, ""))


@dataclass(frozen=True, slots=True)
class Reference:
    """What the person typed, understood. ``kind`` is ``doi``, ``arxiv`` or ``url``."""

    kind: str
    value: str


def classify(raw: str) -> Reference:
    """Whether ``raw`` is a DOI, an arXiv id or a URL, with its canonical spelling.

    A link into doi.org or arxiv.org is the DOI or the id, not a URL to fetch: both have an
    API that says more than the landing page.
    """
    text = raw.strip()
    if not text:
        raise IdentifierError("Enter a DOI, an arXiv id or a link.")
    if len(text) > MAX_INPUT_CHARS:
        raise IdentifierError("That is too long to be a DOI, an arXiv id or a link.")
    if (arxiv := normalise_arxiv(text)) is not None:
        return Reference("arxiv", arxiv)
    if text.lower().startswith(("http://", "https://")):
        if "doi.org/" in text.lower() and (doi := normalise_doi(text)) is not None:
            return Reference("doi", doi)
        url = canonical_url(text)
        if url is None:
            raise IdentifierError("That link is not a web address this application can read.")
        return Reference("url", url)
    if (doi := normalise_doi(text)) is not None:
        return Reference("doi", doi)
    raise IdentifierError(
        "That is not a DOI (10.xxxx/...), an arXiv id (1706.03762) or an http(s) link."
    )
