"""Checking the identifiers a position names.

A check digit proves an identifier is well formed, not that the security exists, and this
module never says more than it knows. Each identifier lands on one rung of a ladder:

ISIN: ``invalid_format`` < ``invalid_checksum`` < ``placeholder`` < ``checksum_valid`` <
``verified`` (only when a security master the user supplied agrees) or ``mismatch`` (when it
disagrees). MIC: ``invalid_format`` < ``structure_valid`` < ``recognized`` (found in a MIC
reference the user supplied) or ``unrecognized`` / ``inactive``.

No reference data is bundled: an exchange list transcribed into code is wrong by the time
anyone reads it. Drop ``spc_mic_reference.csv`` (ISO 10383, a ``MIC`` column and an optional
``STATUS`` column) and ``spc_security_master.csv`` (``isin``, ``mic``, ``ticker``, ``name``)
in the data directory and they are used.
"""

import csv
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)

#: ISO 3166-1 alpha-2, plus the two prefixes ISINs use for supranational issues.
# A string split in one place, not a 250-line list; SIM905 would turn it back into one.
ISIN_PREFIXES = frozenset(
    """
AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ
BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS
BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN
CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE
EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF
GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM
HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM
JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC
LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK
ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA
NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG
PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW
SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS
ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO
TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI
VN VU WF WS YE YT ZA ZM ZW XS EU
""".split()  # noqa: SIM905
)

_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_MIC = re.compile(r"[A-Z0-9]{4}")


def isin_check_digit(body: str) -> int:
    """The ISO 6166 check digit for an 11-character ISIN body: letters become two digits
    (A=10 … Z=35), then Luhn over the whole digit string."""
    digits = "".join(str(int(c, 36)) for c in body)
    total = 0
    # Doubling starts at the digit next to the check digit, the rightmost of the body.
    for position, char in enumerate(reversed(digits)):
        n = int(char)
        if position % 2 == 0:
            n *= 2
            n = n - 9 if n > 9 else n
        total += n
    return (10 - total % 10) % 10


def _looks_placeholder(isin: str) -> bool:
    nsin = isin[2:11]
    return (
        len(set(nsin)) == 1
        or nsin in {"123456789", "987654321", "012345678", "000000001"}
        or nsin.count("0") >= 8
    )


@dataclass(frozen=True, slots=True)
class IdentifierCheck:
    """Where one identifier stands, and the sentence that says why."""

    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Well formed and not a placeholder: good enough to keep, not proof it exists."""
        return self.status not in {"invalid_format", "invalid_checksum", "placeholder"}


def check_isin(value: str) -> IdentifierCheck:
    if not _ISIN.fullmatch(value):
        return IdentifierCheck(
            "invalid_format",
            "An ISIN is 12 characters: a two-letter country code, nine letters or digits, "
            "and one check digit.",
        )
    if value[:2] not in ISIN_PREFIXES:
        return IdentifierCheck(
            "invalid_format", f"{value[:2]} is not a country code an ISIN can start with."
        )
    expected = isin_check_digit(value[:11])
    if expected != int(value[11]):
        return IdentifierCheck(
            "invalid_checksum",
            f"The check digit should be {expected}, not {value[11]}. This ISIN is mistyped "
            "or made up.",
        )
    if _looks_placeholder(value):
        return IdentifierCheck(
            "placeholder", "The body of this ISIN is a pattern, not a real issue's number."
        )
    return IdentifierCheck(
        "checksum_valid",
        "The check digit is right. That does not show the security exists or matches the "
        "company named.",
    )


@dataclass(slots=True)
class SecurityRecord:
    isin: str
    mic: str
    ticker: str
    name: str


@dataclass(slots=True)
class ReferenceData:
    """Optional lists the user supplied. Empty means "cannot verify", never "all fine"."""

    mics: dict[str, str] | None = None
    securities: dict[str, SecurityRecord] | None = field(default=None)

    @property
    def has_mics(self) -> bool:
        return self.mics is not None

    @property
    def has_securities(self) -> bool:
        return self.securities is not None


def load_reference(data_dir: Path) -> ReferenceData:
    """Read whichever reference files exist. A file that cannot be read is skipped, loudly."""
    ref = ReferenceData()
    mic_file = data_dir / "spc_mic_reference.csv"
    if mic_file.is_file():
        try:
            with mic_file.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            ref.mics = {
                (r.get("MIC") or r.get("mic") or "").strip().upper(): (
                    (r.get("STATUS") or r.get("status") or "ACTIVE").strip().upper()
                )
                for r in rows
                if (r.get("MIC") or r.get("mic") or "").strip()
            }
        except OSError, csv.Error, UnicodeDecodeError:
            log.warning("spc.mic_reference_unreadable", path=str(mic_file), exc_info=True)
    master = data_dir / "spc_security_master.csv"
    if master.is_file():
        try:
            with master.open(newline="", encoding="utf-8-sig") as handle:
                records = [
                    SecurityRecord(
                        isin=(r.get("isin") or r.get("ISIN") or "").strip().upper(),
                        mic=(r.get("mic") or r.get("MIC") or "").strip().upper(),
                        ticker=(r.get("ticker") or r.get("TICKER") or "").strip(),
                        name=(r.get("name") or r.get("NAME") or "").strip(),
                    )
                    for r in csv.DictReader(handle)
                ]
            ref.securities = {r.isin: r for r in records if r.isin}
        except OSError, csv.Error, UnicodeDecodeError:
            log.warning("spc.security_master_unreadable", path=str(master), exc_info=True)
    return ref


def check_mic(value: str, ref: ReferenceData | None = None) -> IdentifierCheck:
    if not _MIC.fullmatch(value):
        return IdentifierCheck(
            "invalid_format", "A MIC is four capital letters or digits, such as XNAS."
        )
    if ref is None or ref.mics is None:
        return IdentifierCheck(
            "structure_valid",
            "Well formed. No ISO 10383 list is loaded, so it was not checked against one.",
        )
    state = ref.mics.get(value)
    if state is None:
        return IdentifierCheck("unrecognized", "It is not in the ISO 10383 list loaded here.")
    if state != "ACTIVE":
        return IdentifierCheck("inactive", f"The loaded ISO 10383 list marks it {state.lower()}.")
    return IdentifierCheck("recognized", "Listed as active in the ISO 10383 list loaded here.")


def check_mapping(
    isin: str, mic: str, ticker: str, name: str, ref: ReferenceData | None
) -> IdentifierCheck | None:
    """Whether a security master the user supplied agrees with this position; ``None`` when
    there is none, or it does not list the ISIN. Never guesses a correction."""
    if ref is None or ref.securities is None:
        return None
    record = ref.securities.get(isin)
    if record is None:
        return None
    problems: list[str] = []
    if record.mic and record.mic != mic:
        problems.append(f"the master lists MIC {record.mic}")
    if record.ticker and record.ticker.casefold() != ticker.casefold():
        problems.append(f"the master lists ticker {record.ticker}")
    if record.name and record.name.casefold() != name.casefold():
        problems.append(f"the master lists the name {record.name!r}")
    if problems:
        sentence = "; ".join(problems)
        # Not str.capitalize(), which would lowercase the MIC and ticker it names.
        return IdentifierCheck("mismatch", sentence[0].upper() + sentence[1:] + ".")
    return IdentifierCheck("verified", "Matches the security master loaded here.")
