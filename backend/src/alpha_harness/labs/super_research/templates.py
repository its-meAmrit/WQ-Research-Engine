"""Research templates: weighting ideas an analyst would recognise, with parameters to explore.

A template is text, not a guarantee. It is parsed and run through the same checker as anything
else, so one that needs an operator not yet verified is simply unavailable, and says which.
Every combo weights each Alpha by a non-negative number; a selection ranks Alphas, with the
8,000-operator gate applied by the caller.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..fastexpr import Node, parse
from .build import program

if TYPE_CHECKING:
    import random
    from collections.abc import Callable

WINDOWS = (10, 20, 40, 60, 120, 250, 500)
SHORT = (5, 10, 20, 40)
LONG = (60, 120, 250, 500)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
FLOORS = (0.01, 0.05, 0.1)


@dataclass(frozen=True, slots=True)
class Template:
    key: str
    family: str
    explanation: str
    #: parameter name -> the values it may take
    space: dict[str, tuple[float, ...]]
    body: Callable[..., str]
    #: Operators it calls, so eligibility is known without building it.
    needs: tuple[str, ...]
    ctx: str = "combo"

    def sample(self, rng: random.Random) -> dict[str, float]:
        return {k: rng.choice(v) for k, v in self.space.items()}

    def build(self, params: dict[str, float]) -> Node:
        return program(parse(self.body(**params)), hoist=True)

    def size(self) -> int:
        total = 1
        for values in self.space.values():
            total *= len(values)
        return total


R = "stats.returns"


def _t(
    key: str,
    family: str,
    explanation: str,
    space: dict[str, tuple[float, ...]],
    body: Callable[..., str],
    needs: tuple[str, ...] | list[str],
    ctx: str = "combo",
) -> Template:
    return Template(key, family, explanation, space, body, tuple(needs), ctx)


COMBO_TEMPLATES: tuple[Template, ...] = (
    _t(
        "equal_weight",
        "equal_weight",
        "Every Alpha counts the same: the baseline to beat.",
        {},
        lambda: "1",
        (),
    ),
    _t(
        "rolling_ir",
        "rolling_ir",
        "Weight by each Alpha's recent information ratio, ignoring negative ones.",
        {"window": WINDOWS},
        lambda window: f"max(ts_ir({R}, {int(window)}), 0)",
        ("generate_stats", "ts_ir", "max"),
    ),
    _t(
        "ir_stability",
        "rolling_ir",
        "Reward information ratios that are high and steady: the IR over a long window, "
        "discounted by how much it moved over a short one.",
        {"long": LONG, "short": SHORT},
        lambda long, short: (
            f"max(ts_ir({R}, {int(long)}), 0) / "
            f"(ts_std_dev(ts_ir({R}, {int(long)}), {int(short)}) + 0.1)"
        ),
        ("generate_stats", "ts_ir", "max", "ts_std_dev"),
    ),
    _t(
        "inverse_volatility",
        "inverse_volatility",
        "Smaller weight on Alphas whose returns swing more.",
        {"window": LONG, "floor": (0.0001, 0.001)},
        lambda window, floor: f"1 / (ts_std_dev({R}, {int(window)}) + {floor:g})",
        ("generate_stats", "ts_std_dev"),
    ),
    _t(
        "risk_adjusted_ir",
        "risk_adjusted",
        "Information ratio per unit of recent volatility.",
        {"ir": LONG, "vol": (20, 60, 120)},
        lambda ir, vol: f"max(ts_ir({R}, {int(ir)}), 0) / (ts_std_dev({R}, {int(vol)}) + 0.0001)",
        ("generate_stats", "ts_ir", "max", "ts_std_dev"),
    ),
    _t(
        "least_crowded",
        "correlation_aware",
        "Down-weight an Alpha that is highly correlated with another one in the pool.",
        {"window": (120, 250, 500), "floor": FLOORS},
        lambda window, floor: (
            f"m = self_corr({R}, {int(window)}); "
            f"ic = if_else(m == 1.0, nan, m); max(1 - reduce_max(ic), {floor:g})"
        ),
        ("generate_stats", "self_corr", "if_else", "reduce_max", "max"),
    ),
    _t(
        "decorrelated_ir",
        "correlation_aware",
        "Rolling information ratio, scaled down for Alphas that crowd another.",
        {"ir": LONG, "corr": (120, 250, 500), "floor": FLOORS},
        lambda ir, corr, floor: (
            f"m = self_corr({R}, {int(corr)}); ic = if_else(m == 1.0, nan, m); "
            f"max(ts_ir({R}, {int(ir)}), 0) * max(1 - reduce_max(ic), {floor:g})"
        ),
        ("generate_stats", "ts_ir", "self_corr", "if_else", "reduce_max", "max"),
    ),
    _t(
        "persistence",
        "performance_persistence",
        "Favour Alphas whose recent returns kept the sign of their longer-run ones.",
        {"short": SHORT, "long": LONG},
        lambda short, long: (
            f"max(ts_mean({R}, {int(short)}), 0) * max(ts_mean({R}, {int(long)}), 0)"
        ),
        ("generate_stats", "ts_mean", "max"),
    ),
    _t(
        "drawdown_aware",
        "drawdown_aware",
        "Penalise Alphas sitting far below their recent best return level.",
        {"window": LONG},
        lambda window: f"max(1 + ts_max_diff(ts_sum({R}, {int(window)}), {int(window)}), 0.05)",
        ("generate_stats", "ts_sum", "ts_max_diff", "max"),
    ),
    _t(
        "turnover_sensitive",
        "turnover_sensitive",
        "Smooth the weights so they do not churn, then keep them positive.",
        {"ir": LONG, "decay": (5, 10, 20)},
        lambda ir, decay: f"max(ts_decay_linear(ts_ir({R}, {int(ir)}), {int(decay)}), 0)",
        ("generate_stats", "ts_ir", "ts_decay_linear", "max"),
    ),
    _t(
        "rank_weight",
        "rank_quantile",
        "Weight by where an Alpha's information ratio ranks among the others.",
        {"window": LONG},
        lambda window: f"rank(ts_ir({R}, {int(window)}))",
        ("generate_stats", "ts_ir", "rank"),
    ),
    _t(
        "zscore_weight",
        "rank_quantile",
        "Standardise the information ratios across Alphas and keep the positive part.",
        {"window": LONG, "floor": FLOORS},
        lambda window, floor: f"max(zscore(ts_ir({R}, {int(window)})), {floor:g})",
        ("generate_stats", "ts_ir", "zscore", "max"),
    ),
    _t(
        "regime_switch",
        "regime_sensitive",
        "Trust long-run information ratio in calm regimes and the short-run one when the "
        "pool's volatility is rising.",
        {"long": LONG, "short": SHORT, "vol": (20, 60)},
        lambda long, short, vol: (
            f"calm = ts_std_dev({R}, {int(vol)}) < ts_std_dev({R}, {int(long)}); "
            f"max(if_else(calm, ts_ir({R}, {int(long)}), ts_ir({R}, {int(short)})), 0)"
        ),
        ("generate_stats", "ts_ir", "ts_std_dev", "if_else", "max"),
    ),
    _t(
        "smoothed_decay",
        "smoothing",
        "Information ratio with an exponentially fading memory, to steady the weights.",
        {"ir": LONG, "window": (10, 20, 40), "factor": (0.5, 0.8)},
        lambda ir, window, factor: (
            f"max(ts_decay_exp_window(ts_ir({R}, {int(ir)}), {int(window)}, factor={factor:g}), 0)"
        ),
        ("generate_stats", "ts_ir", "ts_decay_exp_window", "max"),
    ),
    _t(
        "winsorized_ir",
        "risk_safeguard",
        "Rolling information ratio with outliers clipped so one Alpha cannot dominate.",
        {"window": LONG, "std": (2, 3, 4)},
        lambda window, std: f"max(winsorize(ts_ir({R}, {int(window)}), std={std:g}), 0)",
        ("generate_stats", "ts_ir", "winsorize", "max"),
    ),
)

SELECTION_TEMPLATES: tuple[Template, ...] = (
    _t(
        "sel_least_correlated",
        "diversification",
        "Prefer Alphas that overlap little with each other and with what is already submitted.",
        {"self_cap": THRESHOLDS},
        lambda self_cap: f"(self_correlation < {self_cap:g}) * (1 - prod_correlation)",
        (),
        "selection",
    ),
    _t(
        "sel_tradeable_turnover",
        "turnover_sensitive",
        "Keep Alphas that trade in a band a SuperAlpha can hold, and favour the uncrowded.",
        {"low": (0.01, 0.02, 0.05), "high": (0.25, 0.35, 0.4)},
        lambda low, high: f"(turnover >= {low:g}) * (turnover < {high:g}) * (1 - self_correlation)",
        (),
        "selection",
    ),
    _t(
        "sel_broad_books",
        "diversification",
        "Prefer Alphas that hold many names, relative to the size of the universe.",
        {},
        lambda: "(long_count + short_count) / sqrt(universe_size(universe))",
        ("universe_size", "sqrt"),
        "selection",
    ),
    _t(
        "sel_slow_core",
        "turnover_sensitive",
        "Favour slower Alphas, which cost less to combine.",
        {"cap": (0.15, 0.25, 0.35), "soft": (0.02, 0.05, 0.1)},
        lambda cap, soft: f"(turnover < {cap:g}) / (turnover + {soft:g})",
        (),
        "selection",
    ),
    _t(
        "sel_many_sources",
        "dataset_diversification",
        "Favour Alphas built on several data categories and little overlap.",
        {"extra": (0, 1, 2)},
        lambda extra: f"(1 + datacategory_count + {int(extra)}) * (1 - prod_correlation)",
        (),
        "selection",
    ),
    _t(
        "sel_balanced_book",
        "diversification",
        "Favour books balanced between long and short names.",
        {"turn": (0.25, 0.35)},
        lambda turn: (
            f"(turnover < {turn:g}) * "
            "(1 - abs(long_count - short_count) / (long_count + short_count + 1))"
        ),
        ("abs",),
        "selection",
    ),
    _t(
        "sel_uncorrelated_slow",
        "risk_adjusted",
        "Low overlap with the rest of the pool, discounted for turnover.",
        {"soft": (0.05, 0.1, 0.2)},
        lambda soft: f"(1 - self_correlation) * (1 - prod_correlation) / (turnover + {soft:g})",
        (),
        "selection",
    ),
)

BY_KEY = {t.key: t for t in (*COMBO_TEMPLATES, *SELECTION_TEMPLATES)}


def gate(selection: Node, cap: int) -> Node:
    """A selection with the operator-count gate in front: an Alpha too costly scores zero and
    ``POSITIVE`` drops it, so the kept ones never pass BRAIN's 8,000 between them."""
    from .build import binop, num, ref

    return binop("*", binop("<=", ref("operator_count"), num(cap)), selection)
