"""Every documented SuperAlpha operator, where it may run, and what is known about its signature.

The inventory (161 names, their Selection/Combo context) is the documented one. What is *not*
in the documents handed to this application is each operator's signature, so none is taken on
trust: a signature here is a draft until it is verified, and only a verified operator is ever
generated. Two things verify one:

* the argument forms the Classic recipes already run on BRAIN (``verified_by="classic-recipes"``);
* BRAIN's own ``GET /operators`` metadata, via :meth:`Registry.reconciled`, which checks the
  draft's arity and the operator's scope against what BRAIN publishes
  (``verified_by="brain-operators"``).

An operator whose draft disagrees with BRAIN stays pending and says why. One with no draft at
all is registered, shown, and never generated.
"""

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from ..fastexpr import OperatorInfo, operator_table


class Ctx(StrEnum):
    SELECTION = "selection"
    COMBO = "combo"


class Ty(StrEnum):
    """What a value is. ``VEC`` has one entry per Alpha, ``MATRIX`` one per pair of Alphas."""

    SCALAR = "scalar"
    VEC = "vector"
    MATRIX = "matrix"
    STATS = "stats"


class Status(StrEnum):
    VERIFIED = "verified"
    PENDING = "pending_verification"
    INCOMPATIBLE = "incompatible"


#: Parameter kinds. ``num`` takes any shape and broadcasts; the rest are strict.
NUM, VEC, MAT, INT, FLOAT, BOOL, STR, ENUM, ALPHA, UNIVERSE = (
    "num",
    "vec",
    "mat",
    "int",
    "float",
    "bool",
    "str",
    "enum",
    "alpha",
    "universe",
)
DATA_KINDS = frozenset({NUM, VEC, MAT})
CONST_KINDS = frozenset({INT, FLOAT, BOOL, STR, ENUM})

#: Returned shape when it follows the operands: a matrix if any is, else a vector, else a scalar.
JOIN = "join"


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: str
    required: bool = True
    default: str | None = None
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    variadic: bool = False

    def text(self) -> str:
        if self.variadic:
            return f"{self.name}, ..."
        return self.name if self.required else f"{self.name}={self.default}"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    contexts: frozenset[Ctx]
    family: str
    description: str
    #: ``None`` when no signature is documented here: the operator is registered and shown, and
    #: nothing more.
    params: tuple[Param, ...] | None = None
    returns: str = JOIN
    status: Status = Status.PENDING
    verified_by: str | None = None
    #: Why the status is what it is, when it is not simply "verified".
    note: str = ""
    #: Operators this call adds to a SuperAlpha's count, where known.
    cost: int = 1
    doc: str = "SuperAlpha Operators"
    #: Set by :meth:`Registry.reconciled`: BRAIN's own definition, for the explorer to show.
    brain_definition: str | None = None

    @property
    def has_signature(self) -> bool:
        return self.params is not None

    @property
    def signature(self) -> str | None:
        if self.params is None:
            return None
        return f"{self.name}({', '.join(p.text() for p in self.params)})"

    @property
    def required_count(self) -> int:
        return sum(1 for p in (self.params or ()) if p.required and not p.variadic)

    @property
    def maximum(self) -> int | None:
        params = self.params or ()
        return None if any(p.variadic for p in params) else len(params)


# --- signature mini-language -------------------------------------------------------------

_PARAM = re.compile(
    r"^(?P<star>\*)?(?P<name>\w+):(?P<kind>[nvmcbsAUif]|e\[[^\]]+\])"
    r"(?:\[(?P<low>-?[\d.]+)\.\.(?P<high>-?[\d.]+)\])?(?P<opt>\?)?(?:=(?P<default>.*))?$"
)
_KINDS = {
    "n": NUM,
    "v": VEC,
    "m": MAT,
    "i": INT,
    "f": FLOAT,
    "c": FLOAT,
    "b": BOOL,
    "s": STR,
    "A": ALPHA,
    "U": UNIVERSE,
}


def sig(text: str) -> tuple[Param, ...]:
    """Parse ``"x:v, d:i[1..512], constant:f[-1..1]?=0"`` into parameters."""
    params: list[Param] = []
    for raw in (part.strip() for part in text.split(";") if part.strip()):
        match = _PARAM.match(raw)
        if match is None:
            raise ValueError(f"Bad parameter {raw!r}")
        kind = match["kind"]
        choices: tuple[str, ...] = ()
        if kind.startswith("e["):
            choices = tuple(kind[2:-1].split("|"))
            resolved = ENUM
        else:
            resolved = _KINDS[kind]
        optional = match["opt"] is not None
        params.append(
            Param(
                name=match["name"],
                kind=resolved,
                required=not optional and match["star"] is None,
                default=match["default"],
                low=float(match["low"]) if match["low"] is not None else None,
                high=float(match["high"]) if match["high"] is not None else None,
                choices=choices,
                variadic=match["star"] is not None,
            )
        )
    return tuple(params)


BOTH = frozenset({Ctx.SELECTION, Ctx.COMBO})
SEL = frozenset({Ctx.SELECTION})
CMB = frozenset({Ctx.COMBO})

#: Verified for the exact argument forms the Classic recipes send to BRAIN:
#: ``generate_stats(alpha)``, ``ts_ir(x, 250)``, ``self_corr(x, 500)``, ``reduce_max(m)``,
#: ``if_else(c, nan, m)``, ``max(a, b)``, ``ts_std_dev(x, 120)``, ``sqrt(universe_size(universe))``.
CLASSIC_VERIFIED = frozenset(
    {
        "generate_stats",
        "ts_ir",
        "self_corr",
        "reduce_max",
        "if_else",
        "max",
        "ts_std_dev",
        "sqrt",
        "universe_size",
    }
)

_W = "d:i[1..512]"


def _spec(
    name: str,
    contexts: frozenset[Ctx],
    family: str,
    description: str,
    signature: str | None,
    returns: str = JOIN,
) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        contexts=contexts,
        family=family,
        description=description,
        params=None if signature is None else sig(signature),
        returns=returns,
        note="" if signature is not None else "No signature is documented here: never generated.",
    )


# Elementwise operators take any shape and join them. Signatures use ``;`` between parameters.
_ELEMENTWISE_1 = (
    ("abs", "math", "Absolute value."),
    ("arc_cos", "math", "Inverse cosine."),
    ("arc_sin", "math", "Inverse sine."),
    ("arc_tan", "math", "Inverse tangent."),
    ("ceiling", "math", "Round up to a whole number."),
    ("exp", "math", "Natural exponential."),
    ("floor", "math", "Round down to a whole number."),
    ("frac", "math", "Fractional part."),
    ("inverse", "math", "One divided by the value."),
    ("log", "math", "Natural logarithm."),
    ("negate", "math", "Change the sign."),
    ("purify", "nan_handling", "Replace infinities with NaN."),
    ("reverse", "math", "Negate the value."),
    ("round", "math", "Round to the nearest whole number."),
    ("s_log_1p", "math", "Signed log of one plus the magnitude."),
    ("sigmoid", "math", "Logistic squash into 0 to 1."),
    ("sign", "math", "Sign of the value: -1, 0 or 1."),
    ("sqrt", "math", "Square root."),
    ("tanh", "math", "Hyperbolic tangent."),
    ("not", "logical", "Logical not."),
    ("is_finite", "logical", "1 where the value is finite."),
    ("is_nan", "logical", "1 where the value is NaN."),
    ("is_not_finite", "logical", "1 where the value is not finite."),
    ("is_not_nan", "logical", "1 where the value is not NaN."),
)
_ELEMENTWISE_2 = (
    ("and", "logical", "Logical and."),
    ("or", "logical", "Logical or."),
    ("equal", "comparison", "1 where the two are equal."),
    ("greater", "comparison", "1 where the first is larger."),
    ("less", "comparison", "1 where the first is smaller."),
    ("divide", "arithmetic", "First divided by second."),
    ("power", "math", "First raised to the second."),
    ("signed_power", "math", "Sign-preserving power."),
    ("nan_mask", "nan_handling", "NaN where the mask is set."),
)


def _baseline() -> list[OperatorSpec]:
    out: list[OperatorSpec] = []
    for name, family, text in _ELEMENTWISE_1:
        out.append(_spec(name, BOTH, family, text, "x:n"))
    for name, family, text in _ELEMENTWISE_2:
        out.append(_spec(name, BOTH, family, text, "x:n;y:n"))
    out += [
        _spec("add", BOTH, "arithmetic", "Sum of two or more.", "x:n;y:n;filter:b?=false"),
        _spec("subtract", BOTH, "arithmetic", "First less second.", "x:n;y:n;filter:b?=false"),
        _spec("multiply", BOTH, "arithmetic", "Product of two or more.", "x:n;y:n;filter:b?=false"),
        _spec("max", BOTH, "arithmetic", "Larger of the inputs.", "x:n;y:n;*more:n"),
        _spec("min", BOTH, "arithmetic", "Smaller of the inputs.", "x:n;y:n;*more:n"),
        _spec("if_else", BOTH, "conditional", "Pick by condition.", "c:n;a:n;b:n"),
        _spec(
            "clamp",
            BOTH,
            "risk_safeguard",
            "Limit to a range.",
            "x:n;lower:f[-1000..1000]?=0;upper:f[-1000..1000]?=1",
        ),
        _spec(
            "nan_out",
            BOTH,
            "nan_handling",
            "NaN outside a band.",
            "x:n;lower:f[-1000..1000]?=0;upper:f[-1000..1000]?=0",
        ),
        _spec("to_nan", BOTH, "nan_handling", "A value to NaN.", "x:n;value:f[-1000..1000]?=0"),
        _spec("universe_size", SEL, "selection_helper", "Instruments in the universe.", "u:U", "c"),
    ]
    ts1 = (
        ("ts_arg_max", "Days since the window's maximum."),
        ("ts_arg_min", "Days since the window's minimum."),
        ("ts_av_diff", "Value less its window mean."),
        ("ts_count_nans", "NaNs in the window."),
        ("ts_decay_linear", "Linearly weighted window mean."),
        ("ts_delay", "Value d days ago."),
        ("ts_delta", "Change over d days."),
        ("ts_entropy", "Entropy of the window."),
        ("ts_ir", "Window mean over its deviation."),
        ("ts_kurtosis", "Window kurtosis."),
        ("ts_max", "Window maximum."),
        ("ts_max_diff", "Value less its window maximum."),
        ("ts_mean", "Window mean."),
        ("ts_median", "Window median."),
        ("ts_min", "Window minimum."),
        ("ts_min_diff", "Value less its window minimum."),
        ("ts_product", "Window product."),
        ("ts_skewness", "Window skewness."),
        ("ts_std_dev", "Window standard deviation."),
        ("ts_sum", "Window sum."),
        ("ts_zscore", "Window z-score."),
    )
    for name, text in ts1:
        family = "ts_statistics" if name not in {"ts_delay", "ts_delta"} else "time_series"
        if name == "ts_decay_linear":
            family = "decay_smoothing"
        out.append(_spec(name, CMB, family, text, f"x:v;{_W}"))
    out += [
        _spec(
            "ts_rank",
            CMB,
            "time_series",
            "Rank within the window.",
            f"x:v;{_W};constant:f[-10..10]?=0",
        ),
        _spec(
            "ts_scale",
            CMB,
            "time_series",
            "Scale into the window's range.",
            f"x:v;{_W};constant:f[-10..10]?=0",
        ),
        _spec("ts_corr", CMB, "ts_correlation", "Window correlation.", f"x:v;y:v;{_W}"),
        _spec("ts_covariance", CMB, "ts_correlation", "Window covariance.", f"y:v;x:v;{_W}"),
        _spec("ts_theilsen", CMB, "ts_regression", "Theil-Sen slope.", f"x:v;y:v;{_W}"),
        _spec(
            "ts_triple_corr", CMB, "ts_correlation", "Three-way correlation.", f"x:v;y:v;z:v;{_W}"
        ),
        _spec(
            "ts_partial_corr", CMB, "ts_correlation", "Partial correlation.", f"x:v;y:v;z:v;{_W}"
        ),
        _spec("ts_co_kurtosis", CMB, "ts_correlation", "Co-kurtosis.", f"x:v;y:v;{_W}"),
        _spec("ts_co_skewness", CMB, "ts_correlation", "Co-skewness.", f"x:v;y:v;{_W}"),
        _spec("ts_vector_neut", CMB, "vector", "Remove a series' projection.", f"x:v;y:v;{_W}"),
        _spec("ts_vector_proj", CMB, "vector", "Project onto a series.", f"x:v;y:v;{_W}"),
        _spec(
            "ts_regression",
            CMB,
            "ts_regression",
            "Window regression.",
            f"y:v;x:v;{_W};lag:i[0..512]?=0;rettype:i[0..9]?=0",
        ),
        _spec(
            "ts_poly_regression",
            CMB,
            "ts_regression",
            "Polynomial regression residual.",
            f"y:v;x:v;{_W};k:i[1..5]?=1",
        ),
        _spec("ts_returns", CMB, "time_series", "Window return.", f"x:v;{_W};mode:i[1..2]?=1"),
        _spec(
            "ts_percentage",
            CMB,
            "ts_statistics",
            "Window percentile value.",
            f"x:v;{_W};percentage:f[0..1]?=0.5",
        ),
        _spec(
            "ts_quantile",
            CMB,
            "ts_statistics",
            "Window quantile transform.",
            f"x:v;{_W};driver:e[gaussian|uniform|cauchy]?=gaussian",
        ),
        _spec("ts_moment", CMB, "ts_statistics", "Window moment.", f"x:v;{_W};k:i[0..8]?=0"),
        _spec(
            "ts_decay_exp_window",
            CMB,
            "decay_smoothing",
            "Exponentially weighted window.",
            f"x:v;{_W};factor:f[0..1]?=1",
        ),
        _spec(
            "ts_weighted_decay",
            CMB,
            "decay_smoothing",
            "Blend with the previous value.",
            "x:v;k:f[0..1]?=0.5",
        ),
        _spec(
            "ts_min_max_cps",
            CMB,
            "ts_statistics",
            "Window min-max composite.",
            f"x:v;{_W};f:f[0..10]?=2",
        ),
        _spec(
            "ts_min_max_diff",
            CMB,
            "ts_statistics",
            "Window min-max difference.",
            f"x:v;{_W};f:f[0..10]?=0.5",
        ),
        _spec(
            "ts_backfill",
            CMB,
            "nan_handling",
            "Fill NaNs from the window.",
            "x:v;lookback:i[1..512]?=60;k:i[1..10]?=1",
        ),
        _spec("ts_step", CMB, "time_series", "Days since the start.", "n:i[1..1]?=1", "c"),
        _spec(
            "ts_rank_gmean_amean_diff",
            CMB,
            "ts_statistics",
            "Geometric less arithmetic mean of ranks.",
            f"x:v;y:v;z:v;{_W}",
        ),
    ]
    cross = (
        ("zscore", "cross_sectional", "Cross-sectional z-score."),
        ("pasteurize", "cross_sectional", "NaN outside the universe."),
        ("log_diff", "time_series", "Log of the change."),
        ("days_from_last_change", "time_series", "Days since the value changed."),
        ("densify", "cross_sectional", "Compact group labels."),
    )
    for name, family, text in cross:
        out.append(_spec(name, CMB, family, text, "x:v" if name != "densify" else None, "v"))
    out += [
        _spec(
            "rank",
            CMB,
            "cross_sectional",
            "Rank across Alphas, 0 to 1.",
            "x:v;rate:i[0..2]?=2",
            "v",
        ),
        _spec(
            "scale",
            CMB,
            "weight_normalization",
            "Scale to a fixed gross.",
            "x:v;scale:f[0..1000]?=1",
            "v",
        ),
        _spec(
            "scale_down",
            CMB,
            "weight_normalization",
            "Shift down by a constant, then scale.",
            "x:v;constant:f[0..1000]?=0",
            "v",
        ),
        _spec(
            "normalize",
            CMB,
            "weight_normalization",
            "Subtract the mean.",
            "x:v;useStd:b?=false",
            "v",
        ),
        _spec(
            "quantile",
            CMB,
            "cross_sectional",
            "Quantile transform.",
            "x:v;driver:e[gaussian|uniform|cauchy]?=gaussian",
            "v",
        ),
        _spec(
            "winsorize",
            CMB,
            "risk_safeguard",
            "Clip outliers by deviations.",
            "x:v;std:f[0..10]?=4",
            "v",
        ),
        _spec(
            "truncate",
            CMB,
            "risk_safeguard",
            "Cap each weight's share.",
            "x:v;maxPercent:f[0..1]?=0.01",
            "v",
        ),
        _spec(
            "left_tail",
            CMB,
            "cross_sectional",
            "Keep values below a bound.",
            "x:v;maximum:f[-1000..1000]?=0",
            "v",
        ),
        _spec(
            "right_tail",
            CMB,
            "cross_sectional",
            "Keep values above a bound.",
            "x:v;minimum:f[-1000..1000]?=0",
            "v",
        ),
        _spec(
            "tail",
            CMB,
            "cross_sectional",
            "Replace values inside a band.",
            "x:v;lower:f[-1000..1000]?=0;upper:f[-1000..1000]?=0;newval:f[-1000..1000]?=0",
            "v",
        ),
        _spec("one_side", CMB, "cross_sectional", "Keep one side.", "x:v;side:e[long|short]", "v"),
        _spec(
            "rank_by_side",
            CMB,
            "cross_sectional",
            "Rank each side apart.",
            "x:v;rate:i[0..2]?=2;scale:f[0..1000]?=1",
            "v",
        ),
        _spec(
            "round_down",
            CMB,
            "cross_sectional",
            "Round down to a multiple.",
            "x:v;f:f[0..1000]?=1",
            "v",
        ),
        _spec(
            "hump", CMB, "risk_safeguard", "Suppress small changes.", "x:v;hump:f[0..1]?=0.01", "v"
        ),
        _spec(
            "hump_decay", CMB, "decay_smoothing", "Decay small changes.", "x:v;p:f[0..1]?=0", "v"
        ),
        _spec(
            "jump_decay",
            CMB,
            "decay_smoothing",
            "Decay after jumps.",
            f"x:v;{_W};sensitivity:f[0..1]?=0.5;force:f[0..1]?=0.1",
            "v",
        ),
        _spec(
            "kth_element",
            CMB,
            "ts_statistics",
            "The k-th value of a window.",
            f"x:v;{_W};k:i[1..512]?=1",
            "v",
        ),
        _spec("last_diff_value", CMB, "time_series", "Last value that differed.", f"x:v;{_W}", "v"),
        _spec("regression_neut", CMB, "ts_regression", "Residual of a regression.", "y:v;x:v", "v"),
        _spec(
            "regression_proj", CMB, "ts_regression", "Fitted part of a regression.", "y:v;x:v", "v"
        ),
        _spec("vector_neut", CMB, "vector", "Remove a vector's projection.", "x:v;y:v", "v"),
        _spec("vector_proj", CMB, "vector", "Project onto a vector.", "x:v;y:v", "v"),
        _spec(
            "rank_gmean_amean_diff",
            CMB,
            "cross_sectional",
            "Geometric less arithmetic mean of ranks.",
            "x:v;y:v;z:v",
            "v",
        ),
        _spec(
            "trade_when",
            CMB,
            "conditional",
            "Hold the last value unless triggered.",
            "x:v;y:v;z:v",
            "v",
        ),
        _spec(
            "replace",
            CMB,
            "utility",
            "Swap one value for another.",
            "x:v;target:f[-1000..1000];dest:f[-1000..1000]",
            "v",
        ),
    ]
    out += [
        _spec(
            "generate_stats", CMB, "stats_source", "Each Alpha's daily statistics.", "alpha:A", "S"
        ),
        _spec(
            "self_corr",
            CMB,
            "ts_correlation",
            "Correlation of every Alpha pair over a window.",
            "x:v;d:i[1..1000]",
            "m",
        ),
        _spec("reduce_max", CMB, "reduction", "Largest per row.", "x:m", "v"),
        _spec("reduce_min", CMB, "reduction", "Smallest per row.", "x:m", "v"),
        _spec("reduce_avg", CMB, "reduction", "Mean per row.", "x:m", "v"),
        _spec("reduce_sum", CMB, "reduction", "Sum per row.", "x:m", "v"),
        _spec("reduce_stddev", CMB, "reduction", "Deviation per row.", "x:m", "v"),
        _spec("reduce_range", CMB, "reduction", "Range per row.", "x:m", "v"),
        _spec("reduce_skewness", CMB, "reduction", "Skewness per row.", "x:m", "v"),
        _spec("reduce_kurtosis", CMB, "reduction", "Kurtosis per row.", "x:m", "v"),
        _spec("reduce_ir", CMB, "reduction", "Mean over deviation per row.", "x:m", "v"),
    ]
    named = {s.name for s in out}
    # Documented, registered, and not constructible until a signature is verified.
    unsigned: tuple[tuple[str, str, str], ...] = (
        ("bucket", "cross_sectional", "Bucket values into ranges."),
        ("combo_weight", "weight_normalization", "Weights from the combo expression."),
        ("convert", "utility", "Convert between data types."),
        ("count", "reduction", "Count entries."),
        ("filter", "decay_smoothing", "Filter a series."),
        ("group_backfill", "group", "Backfill within groups."),
        ("group_count", "group", "Count within groups."),
        ("group_extra", "group", "Fill group values."),
        ("group_max", "group", "Group maximum."),
        ("group_mean", "group", "Group mean."),
        ("group_median", "group", "Group median."),
        ("group_min", "group", "Group minimum."),
        ("group_neutralize", "group", "Neutralize within groups."),
        ("group_normalize", "group", "Normalize within groups."),
        ("group_percentage", "group", "Percentage within groups."),
        ("group_rank", "group", "Rank within groups."),
        ("group_scale", "group", "Scale within groups."),
        ("group_std_dev", "group", "Group standard deviation."),
        ("group_sum", "group", "Group sum."),
        ("group_vector_neut", "group", "Group vector neutralization."),
        ("group_vector_proj", "group", "Group vector projection."),
        ("group_zscore", "group", "Group z-score."),
        ("inst_pnl", "stats_source", "Instrument PnL."),
        ("inst_tvr", "stats_source", "Instrument turnover."),
        ("keep", "utility", "Keep values meeting a rule."),
        ("position_safeguard", "risk_safeguard", "Position limits."),
        ("reduce_choose", "reduction", "Choose the n-th per row."),
        ("reduce_count", "reduction", "Count per row."),
        ("reduce_norm", "reduction", "Norm per row."),
        ("reduce_percentage", "reduction", "Percentile per row."),
        ("reduce_powersum", "reduction", "Power sum per row."),
        ("unit", "utility", "Unit handling."),
    )
    for name, family, text in unsigned:
        if name not in named:
            out.append(_spec(name, CMB, family, text, None))
    return [
        replace(s, status=Status.VERIFIED, verified_by="classic-recipes", note="")
        if s.name in CLASSIC_VERIFIED
        else (
            replace(s, note="Draft signature: not yet confirmed against BRAIN.")
            if s.has_signature
            else s
        )
        for s in out
    ]


# --- the inventory the registry must hold ------------------------------------------------

BOTH_OPERATORS = [
    "abs",
    "add",
    "and",
    "arc_cos",
    "arc_sin",
    "arc_tan",
    "ceiling",
    "clamp",
    "divide",
    "equal",
    "exp",
    "floor",
    "frac",
    "greater",
    "if_else",
    "inverse",
    "is_finite",
    "is_nan",
    "is_not_finite",
    "is_not_nan",
    "less",
    "log",
    "max",
    "min",
    "multiply",
    "nan_mask",
    "nan_out",
    "negate",
    "not",
    "or",
    "power",
    "purify",
    "reverse",
    "round",
    "s_log_1p",
    "sigmoid",
    "sign",
    "signed_power",
    "sqrt",
    "subtract",
    "tanh",
    "to_nan",
]
SELECTION_ONLY_OPERATORS = ["universe_size"]
COMBO_ONLY_OPERATORS = [
    "bucket",
    "combo_weight",
    "convert",
    "count",
    "days_from_last_change",
    "densify",
    "filter",
    "generate_stats",
    "group_backfill",
    "group_count",
    "group_extra",
    "group_max",
    "group_mean",
    "group_median",
    "group_min",
    "group_neutralize",
    "group_normalize",
    "group_percentage",
    "group_rank",
    "group_scale",
    "group_std_dev",
    "group_sum",
    "group_vector_neut",
    "group_vector_proj",
    "group_zscore",
    "hump",
    "hump_decay",
    "inst_pnl",
    "inst_tvr",
    "jump_decay",
    "keep",
    "kth_element",
    "last_diff_value",
    "left_tail",
    "log_diff",
    "normalize",
    "one_side",
    "pasteurize",
    "position_safeguard",
    "quantile",
    "rank",
    "rank_by_side",
    "rank_gmean_amean_diff",
    "reduce_avg",
    "reduce_choose",
    "reduce_count",
    "reduce_ir",
    "reduce_kurtosis",
    "reduce_max",
    "reduce_min",
    "reduce_norm",
    "reduce_percentage",
    "reduce_powersum",
    "reduce_range",
    "reduce_skewness",
    "reduce_stddev",
    "reduce_sum",
    "regression_neut",
    "regression_proj",
    "replace",
    "right_tail",
    "round_down",
    "scale",
    "scale_down",
    "self_corr",
    "tail",
    "trade_when",
    "truncate",
    "ts_arg_max",
    "ts_arg_min",
    "ts_av_diff",
    "ts_backfill",
    "ts_co_kurtosis",
    "ts_co_skewness",
    "ts_corr",
    "ts_count_nans",
    "ts_covariance",
    "ts_decay_exp_window",
    "ts_decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_entropy",
    "ts_ir",
    "ts_kurtosis",
    "ts_max",
    "ts_max_diff",
    "ts_mean",
    "ts_median",
    "ts_min",
    "ts_min_diff",
    "ts_min_max_cps",
    "ts_min_max_diff",
    "ts_moment",
    "ts_partial_corr",
    "ts_percentage",
    "ts_poly_regression",
    "ts_product",
    "ts_quantile",
    "ts_rank",
    "ts_rank_gmean_amean_diff",
    "ts_regression",
    "ts_returns",
    "ts_scale",
    "ts_skewness",
    "ts_std_dev",
    "ts_step",
    "ts_sum",
    "ts_theilsen",
    "ts_triple_corr",
    "ts_vector_neut",
    "ts_vector_proj",
    "ts_weighted_decay",
    "ts_zscore",
    "unit",
    "vector_neut",
    "vector_proj",
    "winsorize",
    "zscore",
]


# --- fields a selection or a combo may read ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    ty: Ty
    status: Status
    verified_by: str | None = None
    note: str = ""


#: What a selection ranks: properties of each submitted Alpha. These are the ones the Classic
#: recipes already use on BRAIN. Others a SuperAlpha selection might read (Sharpe, Fitness, ...)
#: are not documented in what this application holds, so they are not here to be guessed at.
SELECTION_FIELDS: tuple[Symbol, ...] = tuple(
    Symbol(name, Ty.VEC, Status.VERIFIED, "classic-recipes")
    for name in (
        "operator_count",
        "self_correlation",
        "prod_correlation",
        "turnover",
        "long_count",
        "short_count",
        "datacategory_count",
    )
)
#: Members of ``generate_stats(alpha)``. Only ``returns`` is used by a Classic recipe.
STATS_MEMBERS: tuple[Symbol, ...] = (Symbol("returns", Ty.VEC, Status.VERIFIED, "classic-recipes"),)
#: Names that are values rather than fields in either context.
LITERAL_NAMES = {"nan": Ty.SCALAR, "true": Ty.SCALAR, "false": Ty.SCALAR}


@dataclass(frozen=True, slots=True)
class Eligibility:
    eligible: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Registry:
    """The operators, as verified so far. Immutable: :meth:`reconciled` returns a new one."""

    specs: dict[str, OperatorSpec] = field(default_factory=dict)
    stats_members: tuple[Symbol, ...] = STATS_MEMBERS
    selection_fields: tuple[Symbol, ...] = SELECTION_FIELDS

    def __len__(self) -> int:
        return len(self.specs)

    def get(self, name: str) -> OperatorSpec | None:
        return self.specs.get(name)

    def names(self, ctx: Ctx | None = None) -> list[str]:
        return [n for n, s in self.specs.items() if ctx is None or ctx in s.contexts]

    # -- what can be built -------------------------------------------------------------

    def producers(self, ctx: Ctx, ty: Ty) -> list[OperatorSpec]:
        """Verified operators that return ``ty`` in ``ctx``, whatever shape they take in."""
        return [
            s
            for s in self.specs.values()
            if ctx in s.contexts
            and s.status == Status.VERIFIED
            and s.has_signature
            and _returns(s) == ty
        ]

    def vec_sources(self, ctx: Ctx) -> list[Symbol]:
        fields = self.selection_fields if ctx == Ctx.SELECTION else self.stats_members
        return [f for f in fields if f.status == Status.VERIFIED and f.ty == Ty.VEC]

    def eligibility(
        self,
        name: str,
        ctx: Ctx,
        *,
        disabled: frozenset[str] = frozenset(),
        families_off: frozenset[str] = frozenset(),
    ) -> Eligibility:
        """Whether the generator may use ``name`` in ``ctx``, and if not, exactly why."""
        spec = self.specs.get(name)
        if spec is None:
            return Eligibility(False, "Not in the operator registry.")
        if ctx not in spec.contexts:
            where = " and ".join(sorted(c.value for c in spec.contexts))
            return Eligibility(False, f"Documented for {where} only; not valid in {ctx.value}.")
        if spec.status == Status.INCOMPATIBLE:
            return Eligibility(
                False, spec.note or "Incompatible with BRAIN's published definition."
            )
        if spec.params is None:
            return Eligibility(False, spec.note or "No signature is documented here.")
        if spec.status != Status.VERIFIED:
            return Eligibility(False, spec.note or "Signature pending verification against BRAIN.")
        if name in disabled:
            return Eligibility(False, "Switched off for this task.")
        if spec.family in families_off:
            return Eligibility(False, f"Its family, {spec.family}, is switched off.")
        gap = self._unfillable(spec, ctx)
        if gap:
            return Eligibility(False, gap)
        return Eligibility(True)

    def _unfillable(self, spec: OperatorSpec, ctx: Ctx) -> str:
        for param in spec.params or ():
            if not param.required and not param.variadic:
                continue
            if param.kind == ALPHA:
                if ctx != Ctx.COMBO:
                    return "Takes the alpha input, which only a combo has."
            elif param.kind == UNIVERSE:
                if ctx != Ctx.SELECTION:
                    return "Takes the universe, which only a selection has."
            elif param.kind in (NUM, VEC):
                if not self.vec_sources(ctx):
                    return f"No verified {ctx.value} input exists to feed {param.name}."
            elif param.kind == MAT and not any(
                p.params is not None and all(q.kind != MAT for q in p.params if q.required)
                for p in self.producers(ctx, Ty.MATRIX)
            ):
                return "Needs a matrix input and no verified operator produces one."
        return ""

    # -- verification against BRAIN ------------------------------------------------------

    def reconciled(
        self, operators: list[dict[str, Any]], *, strict_scope: bool = True
    ) -> tuple[Registry, ReconcileReport]:
        """Check every draft against BRAIN's operator list, promoting or demoting each.

        An operator is promoted when BRAIN publishes it, BRAIN's stated scope covers every
        context this registry allows it in, and the draft's arity matches the published
        definition. A name alone is never enough.

        ``strict_scope`` off accepts an operator whose published scope names neither COMBO nor
        SELECTION (some lists only say REGULAR), on its arity alone. That is the consultant's
        explicit choice, and is recorded as ``brain-operators (arity only)``.
        """
        published = {str(o["name"]): o for o in operators if isinstance(o.get("name"), str)}
        report = ReconcileReport()
        updated: dict[str, OperatorSpec] = {}
        for name, spec in self.specs.items():
            raw = published.get(name)
            if raw is None:
                updated[name] = replace(
                    spec,
                    status=Status.PENDING if spec.status != Status.VERIFIED else spec.status,
                    note=spec.note or "BRAIN's operator list does not include it.",
                )
                report.missing.append(name)
                continue
            raw_def = raw.get("definition")
            definition = raw_def if isinstance(raw_def, str) else None
            stated = {str(s).upper() for s in (raw.get("scope") or [])}
            names_super = bool(stated & {"COMBO", "SELECTION"})
            if names_super:
                lacking = [c.value for c in spec.contexts if c.value.upper() not in stated]
                if lacking:
                    updated[name] = replace(
                        spec,
                        status=Status.INCOMPATIBLE,
                        verified_by=None,
                        note=f"BRAIN's scope for it ({', '.join(sorted(stated))}) "
                        f"excludes {', '.join(lacking)}.",
                        brain_definition=definition,
                    )
                    report.incompatible.append(name)
                    continue
            elif strict_scope and stated:
                updated[name] = replace(
                    spec,
                    brain_definition=definition,
                    note=spec.note
                    or f"BRAIN lists it for {', '.join(sorted(stated))} only, so its "
                    "Combo/Selection scope cannot be confirmed.",
                )
                report.unconfirmed_scope.append(name)
                continue
            info = operator_table([{**raw, "scope": []}], "COMBO").get(name)
            if spec.params is None or info is None:
                updated[name] = replace(
                    spec,
                    brain_definition=definition,
                    note=spec.note
                    if spec.params is None
                    else "BRAIN's definition could not be read.",
                )
                report.unsigned.append(name)
                continue
            problem = arity_mismatch(spec, info)
            if problem:
                updated[name] = replace(
                    spec,
                    status=Status.PENDING,
                    verified_by=None,
                    note=problem,
                    brain_definition=definition,
                )
                report.mismatched.append(name)
                continue
            how = (
                spec.verified_by
                if spec.verified_by == "classic-recipes"
                else "brain-operators"
                if names_super
                else "brain-operators (arity only)"
            )
            updated[name] = replace(
                spec,
                status=Status.VERIFIED,
                verified_by=how,
                note="",
                brain_definition=definition,
            )
            report.verified.append(name)
        return replace(self, specs=updated), report

    # -- reporting -------------------------------------------------------------------------

    def counts(
        self,
        *,
        disabled: frozenset[str] = frozenset(),
        families_off: frozenset[str] = frozenset(),
    ) -> dict[str, int]:
        """Operator totals by validation and generation status, for the explorer and the report."""
        out = {
            "registered": len(self.specs),
            "selectionAndCombo": 0,
            "selectionOnly": 0,
            "comboOnly": 0,
            Status.VERIFIED: 0,
            Status.PENDING: 0,
            Status.INCOMPATIBLE: 0,
            "signed": 0,
            "eligible": 0,
        }
        for name, spec in self.specs.items():
            if spec.contexts == BOTH:
                out["selectionAndCombo"] += 1
            elif spec.contexts == SEL:
                out["selectionOnly"] += 1
            else:
                out["comboOnly"] += 1
            out[spec.status] += 1
            out["signed"] += int(spec.has_signature)
            if any(
                self.eligibility(name, c, disabled=disabled, families_off=families_off).eligible
                for c in spec.contexts
            ):
                out["eligible"] += 1
        return {str(k): v for k, v in out.items()}


@dataclass(slots=True)
class ReconcileReport:
    verified: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    incompatible: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unsigned: list[str] = field(default_factory=list)
    unconfirmed_scope: list[str] = field(default_factory=list)


def arity_mismatch(spec: OperatorSpec, info: OperatorInfo) -> str:
    """Why the draft's argument count disagrees with BRAIN's definition, or nothing."""
    if info.maximum is None:  # BRAIN's takes any number: only the required count can disagree
        if spec.required_count < info.required:
            return (
                f"BRAIN's definition needs {info.required} inputs; "
                f"the draft needs {spec.required_count}."
            )
        return ""
    if spec.required_count != info.required:
        return (
            f"BRAIN's definition needs {info.required} inputs; "
            f"the draft needs {spec.required_count}."
        )
    drafted = spec.maximum
    if drafted is not None and drafted > info.maximum:
        return f"BRAIN's definition takes at most {info.maximum} inputs; the draft names {drafted}."
    return ""


def _returns(spec: OperatorSpec) -> Ty | None:
    return {"c": Ty.SCALAR, "v": Ty.VEC, "m": Ty.MATRIX, "S": Ty.STATS}.get(spec.returns)


def returns_of(spec: OperatorSpec) -> str:
    return spec.returns


def baseline() -> Registry:
    specs = {s.name: s for s in _baseline()}
    return Registry(specs=specs)


BASELINE = baseline()
