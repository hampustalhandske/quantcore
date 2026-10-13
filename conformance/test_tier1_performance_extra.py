"""Tier-1 differential tests for Sortino, Calmar, information ratio, hit rate, profit factor.

Also covers rolling Sharpe -- against `empyrical-reloaded` and, for the trade statistics, a
`quantstats` second-oracle cross-check -- and the nine drawdown, payoff and tail ratios of
`quantcore.performance.ratios` against `quantstats`.
"""

import numpy as np
import pandas as pd
import quantstats.stats
from hypothesis import assume, given
from hypothesis import strategies as st

import conformance.targets.performance_extra  # noqa: F401
from conformance.compare import registry
from conformance.convert import annual_rate_to_per_period
from conformance.generators import (
    RETURN_FAMILIES,
    ReturnFamily,
    TaggedReturns,
    length_bucket_for,
    returns_series,
)
from conformance.harness import check, check_second_oracle

_SORTINO_RATIO = registry()["sortino_ratio"]
_CALMAR_RATIO = registry()["calmar_ratio"]
_CALMAR_RATIO_ARITHMETIC = registry()["calmar_ratio_arithmetic"]
_INFORMATION_RATIO = registry()["information_ratio"]
_HIT_RATE = registry()["hit_rate"]
_PROFIT_FACTOR = registry()["profit_factor"]
_ROLLING_SHARPE = registry()["rolling_sharpe"]
_ULCER_INDEX = registry()["ulcer_index"]
_ULCER_PERFORMANCE_INDEX = registry()["ulcer_performance_index"]
_SERENITY_INDEX = registry()["serenity_index"]
_GAIN_TO_PAIN = registry()["gain_to_pain"]
_COMMON_SENSE_RATIO = registry()["common_sense_ratio"]
_CPC_INDEX = registry()["cpc_index"]
_OUTLIER_WIN_RATIO = registry()["outlier_win_ratio"]
_OUTLIER_LOSS_RATIO = registry()["outlier_loss_ratio"]
_RISK_OF_RUIN = registry()["risk_of_ruin"]

SORTINO_RATIO_MIN_LEN = 2
SORTINO_RATIO_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
SORTINO_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# "near_constant" and "all_positive" never produce a non-zero drawdown (the path only ever rises,
# give or take float noise), so calmar is structurally undefined (empyrical: 0/0 = nan) on every
# case from those two families; they are excluded from declared coverage rather than forced to
# report a hollow zero-case cell.
_CALMAR_FAMILIES = tuple(f for f in RETURN_FAMILIES if f not in ("near_constant", "all_positive"))

CALMAR_RATIO_MIN_LEN = 2
CALMAR_RATIO_FAMILIES: tuple[str, ...] = _CALMAR_FAMILIES
CALMAR_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

CALMAR_RATIO_ARITHMETIC_MIN_LEN = 2
CALMAR_RATIO_ARITHMETIC_FAMILIES: tuple[str, ...] = _CALMAR_FAMILIES
CALMAR_RATIO_ARITHMETIC_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

INFORMATION_RATIO_MIN_LEN = 2
INFORMATION_RATIO_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
INFORMATION_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

HIT_RATE_MIN_LEN = 1
HIT_RATE_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
HIT_RATE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# "all_positive" and "near_constant" (base level 0.0004, noise std 1e-9) never draw a losing
# trade, so profit_factor = +inf by construction on every one of their cases (excluded from the
# comparison by the same assume() as every other family's occasional no-losses case); dropped
# from declared coverage rather than forced to report a hollow zero-case cell.
PROFIT_FACTOR_MIN_LEN = 1
PROFIT_FACTOR_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("all_positive", "near_constant")
)
PROFIT_FACTOR_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

ROLLING_SHARPE_MIN_LEN = 3
ROLLING_SHARPE_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
ROLLING_SHARPE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# The nine ratios below are drawn only from the families in which the ratio is defined, so every
# declared family is one the test can record a case for.
#
# "near_constant" (0.0004 plus noise of order 1e-9) and "all_positive" never draw a loss: the path
# is never below its peak and there is no negative return, which leaves the Ulcer Performance
# Index, the Serenity index, gain-to-pain, the common sense ratio and the outlier loss ratio
# unbounded or undefined (quantstats: nan or inf) on every case.
_FAMILIES_WITH_A_LOSS: tuple[ReturnFamily, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("near_constant", "all_positive")
)
# "all_negative" never draws a gain: no winning period for the CPC index's payoff ratio, and no
# non-negative return for the outlier win ratio's denominator.
_FAMILIES_WITH_A_GAIN: tuple[ReturnFamily, ...] = tuple(
    f for f in RETURN_FAMILIES if f != "all_negative"
)

ULCER_INDEX_MIN_LEN = 2
"""Smallest `returns` length `ulcer_index` accepts: its divisor is `n - 1`."""
ULCER_INDEX_FAMILIES: tuple[ReturnFamily, ...] = RETURN_FAMILIES
ULCER_INDEX_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

ULCER_PERFORMANCE_INDEX_MIN_LEN = 2
ULCER_PERFORMANCE_INDEX_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_LOSS
ULCER_PERFORMANCE_INDEX_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

SERENITY_INDEX_MIN_LEN = 2
SERENITY_INDEX_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_LOSS
SERENITY_INDEX_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

GAIN_TO_PAIN_MIN_LEN = 1
GAIN_TO_PAIN_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_LOSS
GAIN_TO_PAIN_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

COMMON_SENSE_RATIO_MIN_LEN = 1
COMMON_SENSE_RATIO_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_LOSS
COMMON_SENSE_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

CPC_INDEX_MIN_LEN = 1
CPC_INDEX_FAMILIES: tuple[ReturnFamily, ...] = tuple(
    f for f in _FAMILIES_WITH_A_LOSS if f in _FAMILIES_WITH_A_GAIN
)
CPC_INDEX_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

OUTLIER_WIN_RATIO_MIN_LEN = 1
OUTLIER_WIN_RATIO_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_GAIN
OUTLIER_WIN_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

OUTLIER_LOSS_RATIO_MIN_LEN = 1
OUTLIER_LOSS_RATIO_FAMILIES: tuple[ReturnFamily, ...] = _FAMILIES_WITH_A_LOSS
OUTLIER_LOSS_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

RISK_OF_RUIN_MIN_LEN = 1
RISK_OF_RUIN_FAMILIES: tuple[ReturnFamily, ...] = RETURN_FAMILIES
RISK_OF_RUIN_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    tagged=returns_series(min_len=SORTINO_RATIO_MIN_LEN),
    risk_free_annual=st.floats(
        min_value=-0.05, max_value=0.10, allow_nan=False, allow_infinity=False
    ),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_sortino_ratio_matches_empyrical(
    tagged: TaggedReturns, risk_free_annual: float, periods_per_year: int
) -> None:
    """Quantcore's `sortino_ratio` against `empyrical.sortino_ratio`."""
    risk_free_per_period = annual_rate_to_per_period(risk_free_annual, periods_per_year)
    downside = np.minimum(tagged.values - risk_free_per_period, 0.0)
    assume(float(np.mean(downside**2)) != 0.0)
    check(
        _SORTINO_RATIO,
        tagged.values,
        risk_free_per_period,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


def _max_drawdown_nonzero(values: np.ndarray) -> bool:
    cumulative = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(cumulative)
    return bool(np.max((peak - cumulative) / peak) != 0.0)


@given(
    tagged=returns_series(min_len=CALMAR_RATIO_MIN_LEN),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_calmar_ratio_matches_empyrical(tagged: TaggedReturns, periods_per_year: int) -> None:
    """Quantcore's CAGR-based `calmar_ratio` against `empyrical.calmar_ratio`.

    A max drawdown of exactly zero (a never-declining path) makes `empyrical.calmar_ratio`
    compute `0/0 = nan` rather than the `+inf` quantcore returns for the same case -- excluded
    here since the reference itself is undefined on that input, not a disagreement to record.
    """
    assume(_max_drawdown_nonzero(tagged.values))
    check(
        _CALMAR_RATIO,
        tagged.values,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=CALMAR_RATIO_ARITHMETIC_MIN_LEN),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_calmar_ratio_arithmetic_matches_hand_formula(
    tagged: TaggedReturns, periods_per_year: int
) -> None:
    """Quantcore's arithmetic-return Calmar variant against the hand formula `mean*ppy/|MDD|`."""
    assume(_max_drawdown_nonzero(tagged.values))
    check(
        _CALMAR_RATIO_ARITHMETIC,
        tagged.values,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=INFORMATION_RATIO_MIN_LEN),
    benchmark=returns_series(min_len=INFORMATION_RATIO_MIN_LEN),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_information_ratio_matches_empyrical(
    tagged: TaggedReturns, benchmark: TaggedReturns, periods_per_year: int
) -> None:
    """Quantcore's `information_ratio` against annualised `empyrical.excess_sharpe`."""
    n = min(tagged.n, benchmark.n)
    assume(n >= INFORMATION_RATIO_MIN_LEN)
    r = tagged.values[:n]
    b = benchmark.values[:n]
    assume(float(np.std(r - b, ddof=1)) != 0.0)
    check(
        _INFORMATION_RATIO,
        r,
        b,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=HIT_RATE_MIN_LEN))
def test_hit_rate_matches_hand_formula(tagged: TaggedReturns) -> None:
    """Quantcore's `hit_rate` (zero_policy="exclude") on trade P&L against the hand formula."""
    assume(bool(np.any(tagged.values != 0.0)))
    check(
        _HIT_RATE,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=HIT_RATE_MIN_LEN))
def test_hit_rate_quantstats_cross_check(tagged: TaggedReturns) -> None:
    """`quantstats.stats.win_rate` against the target's own hand-computed hit rate.

    Never asserts; only records agreement or disagreement. `quantstats` excludes zero-P&L trades
    from its denominator exactly like `zero_policy="exclude"`, so the two are expected to agree,
    except on a draw with no negative value and a maximum above 1, which `quantstats` takes for
    prices and replaces by its percentage change (recorded as a disagreement, like any other).
    `quantstats` needs a `pd.Series`, so that is what both callables are given.
    """
    assume(bool(np.any(tagged.values != 0.0)))
    check_second_oracle(
        _HIT_RATE,
        pd.Series(tagged.values),
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        oracle_name="quantstats",
        oracle_callable=quantstats.stats.win_rate,
        quantity="win_rate",
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=PROFIT_FACTOR_MIN_LEN))
def test_profit_factor_matches_hand_formula(tagged: TaggedReturns) -> None:
    """Quantcore's `profit_factor` on trade P&L against the hand formula.

    Excludes the no-losing-trades case: quantcore and the oracle both return `+inf` there by
    contract, which the harness's finiteness check treats as invalidating the comparison rather
    than as agreement.
    """
    assume(bool(np.any(tagged.values < 0.0)))
    check(
        _PROFIT_FACTOR,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=PROFIT_FACTOR_MIN_LEN))
def test_profit_factor_quantstats_cross_check(tagged: TaggedReturns) -> None:
    """`quantstats.stats.profit_factor` against the target's own hand-computed profit factor.

    Never asserts. `quantstats.profit_factor` includes exactly-zero P&L in the *wins* sum
    (`returns >= 0`) where the reference excludes zero from both sums, which changes neither sum.
    `quantstats` needs a `pd.Series`, so that is what both callables are given.
    """
    assume(bool(np.any(tagged.values < 0.0)))
    check_second_oracle(
        _PROFIT_FACTOR,
        pd.Series(tagged.values),
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        oracle_name="quantstats",
        oracle_callable=quantstats.stats.profit_factor,
        quantity="profit_factor",
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=ROLLING_SHARPE_MIN_LEN),
    window=st.integers(min_value=2, max_value=10),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_rolling_sharpe_matches_empyrical(
    tagged: TaggedReturns, window: int, periods_per_year: int
) -> None:
    """Quantcore's `rolling_sharpe` (trimmed to its finite tail) vs `empyrical.roll_sharpe_ratio`.

    Excludes any input with a zero-variance trailing window (undefined Sharpe in both
    quantcore and the oracle), the same exclusion `sharpe_ratio` itself makes.
    """
    assume(window <= tagged.n)
    windows_have_variance = all(
        float(np.std(tagged.values[i - window + 1 : i + 1], ddof=1)) != 0.0
        for i in range(window - 1, tagged.n)
    )
    assume(windows_have_variance)
    check(
        _ROLLING_SHARPE,
        tagged.values,
        window,
        0.0,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


def _drawdowns(values: np.ndarray) -> np.ndarray:
    """Drawdown at every period as a positive fraction of the running peak, starting capital 1."""
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return (peak - wealth) / peak


def _quantstats_reads_prices(values: np.ndarray) -> bool:
    """Whether `quantstats` would take this array of returns for a series of prices.

    `quantstats.utils._prepare_returns` replaces a series by its percentage change when its
    minimum is non-negative and its maximum exceeds 1, and `_prepare_prices` skips compounding
    when the minimum is non-negative and the maximum is at least 1. Either way the oracle then
    computes the statistic of a different series. A series with a negative value is never
    affected.
    """
    return bool(np.min(values) >= 0.0 and np.max(values) >= 1.0)


def _quantstats_rebases_drawdowns(values: np.ndarray) -> bool:
    """Whether `quantstats` would measure drawdowns from a starting capital other than 1.

    `quantstats.stats._get_baseline_value` takes the starting capital to be 100 when the first
    wealth value exceeds 10, and 1e5 when it exceeds 1000, so a first return above 9 can open the
    oracle's drawdown series with a decline that never happened.
    """
    return bool(1.0 + values[0] > 10.0)


@given(tagged=returns_series(min_len=ULCER_INDEX_MIN_LEN, families=ULCER_INDEX_FAMILIES))
def test_ulcer_index_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `ulcer_index` against `quantstats.stats.ulcer_index`.

    A path that is never below its peak has an index of exactly 0 on both sides and is compared
    like any other case.
    """
    assume(not _quantstats_reads_prices(tagged.values))
    assume(not _quantstats_rebases_drawdowns(tagged.values))
    check(
        _ULCER_INDEX,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(
        min_len=ULCER_PERFORMANCE_INDEX_MIN_LEN, families=ULCER_PERFORMANCE_INDEX_FAMILIES
    )
)
def test_ulcer_performance_index_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `ulcer_performance_index` against `quantstats.stats.ulcer_performance_index`.

    Excludes a path that is never below its peak: the Ulcer Index is 0 there, quantcore returns
    `+inf` and `quantstats` returns `nan`, so there is no finite reference to compare against.
    """
    assume(not _quantstats_rebases_drawdowns(tagged.values))
    assume(float(np.max(_drawdowns(tagged.values))) != 0.0)
    check(
        _ULCER_PERFORMANCE_INDEX,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=SERENITY_INDEX_MIN_LEN, families=SERENITY_INDEX_FAMILIES))
def test_serenity_index_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `serenity_index` against `quantstats.stats.serenity_index`.

    Excludes constant returns and a constant drawdown series (which includes a path never below
    its peak). Quantcore returns `nan` for both; `quantstats` returns `nan` when the standard
    deviation it computes is exactly zero and otherwise a number built from the rounding residue
    of that standard deviation, so neither is a reference.
    """
    assume(not _quantstats_rebases_drawdowns(tagged.values))
    assume(float(np.ptp(tagged.values)) != 0.0)
    assume(float(np.ptp(_drawdowns(tagged.values))) != 0.0)
    check(
        _SERENITY_INDEX,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=GAIN_TO_PAIN_MIN_LEN, families=GAIN_TO_PAIN_FAMILIES))
def test_gain_to_pain_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `gain_to_pain_ratio` against `quantstats.stats.gain_to_pain_ratio`.

    Excludes a series with no loss: quantcore returns `+inf` and `quantstats` returns `nan`.
    """
    assume(bool(np.any(tagged.values < 0.0)))
    check(
        _GAIN_TO_PAIN,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=COMMON_SENSE_RATIO_MIN_LEN, families=COMMON_SENSE_RATIO_FAMILIES)
)
def test_common_sense_ratio_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `common_sense_ratio` against `quantstats.stats.common_sense_ratio`.

    Excludes a series with no loss (the profit factor, and with it the ratio, is `+inf` or `nan`
    on both sides) and a lower tail quantile of exactly 0 (`nan` on both sides).
    """
    assume(bool(np.any(tagged.values < 0.0)))
    assume(float(np.quantile(tagged.values, 1.0 - 0.95)) != 0.0)
    check(
        _COMMON_SENSE_RATIO,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=CPC_INDEX_MIN_LEN, families=CPC_INDEX_FAMILIES))
def test_cpc_index_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `cpc_index` against `quantstats.stats.cpc_index`.

    Excludes a series with no winning or no losing period: the payoff ratio is undefined and both
    sides return `nan`.
    """
    assume(bool(np.any(tagged.values > 0.0)) and bool(np.any(tagged.values < 0.0)))
    check(
        _CPC_INDEX,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=OUTLIER_WIN_RATIO_MIN_LEN, families=OUTLIER_WIN_RATIO_FAMILIES)
)
def test_outlier_win_ratio_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `outlier_win_ratio` against `quantstats.stats.outlier_win_ratio`.

    Excludes a series with no positive return: the mean of its non-negative returns is missing or
    zero and both sides return `nan`.
    """
    assume(not _quantstats_reads_prices(tagged.values))
    assume(bool(np.any(tagged.values > 0.0)))
    check(
        _OUTLIER_WIN_RATIO,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=OUTLIER_LOSS_RATIO_MIN_LEN, families=OUTLIER_LOSS_RATIO_FAMILIES)
)
def test_outlier_loss_ratio_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `outlier_loss_ratio` against `quantstats.stats.outlier_loss_ratio`.

    Excludes a series with no loss: there is no mean negative return and both sides return `nan`.
    """
    assume(bool(np.any(tagged.values < 0.0)))
    check(
        _OUTLIER_LOSS_RATIO,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=RISK_OF_RUIN_MIN_LEN, families=RISK_OF_RUIN_FAMILIES))
def test_risk_of_ruin_matches_quantstats(tagged: TaggedReturns) -> None:
    """Quantcore's `risk_of_ruin` against `quantstats.stats.risk_of_ruin`."""
    assume(not _quantstats_reads_prices(tagged.values))
    check(
        _RISK_OF_RUIN,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )
