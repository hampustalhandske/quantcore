"""Tier-1 performance metrics: Sortino, Calmar, information ratio, hit rate, rolling Sharpe.

Sortino divides by the downside deviation taken over every observation (gains count as zero),
with `rf` de-annualised to the series' periodicity. Calmar is `CAGR / |max drawdown|`. The
information ratio is `mean(r - b) / std(r - b, ddof=1) * sqrt(ppy)`. Hit rate and profit factor
use trade net P&L (zero-P&L trades excluded from the hit rate). Rolling Sharpe drops the
unfilled first `window - 1` periods.

Nine drawdown, payoff and tail ratios (Ulcer Index, Ulcer Performance Index, Serenity index,
gain-to-pain, common sense ratio, CPC index, outlier win and loss ratios, risk of ruin) are checked
against `quantstats`, whose definitions `quantcore.performance.ratios` implements. None is
annualised and none takes a risk-free rate (`quantstats`' `rf` is passed as 0 where it has one).
`quantstats` guesses from the values whether it was handed returns or prices; the tier test keeps
out the inputs on which that guess is wrong, since quantcore always treats its input as returns.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import quantstats.stats as qs_stats
from empyrical import calmar_ratio as _empyrical_calmar_ratio
from empyrical import excess_sharpe as _empyrical_excess_sharpe
from empyrical import max_drawdown as _empyrical_max_drawdown
from empyrical import roll_sharpe_ratio as _empyrical_roll_sharpe_ratio
from empyrical import sortino_ratio as _empyrical_sortino_ratio

import quantcore.performance as qc_performance
from conformance.compare import ArrayComparator, ConformanceTarget, ScalarComparator

# --- Sortino ratio ----------------------------------------------------------------------------


def _sortino_ratio_quantcore(
    returns: npt.NDArray[np.float64], risk_free_per_period: float, periods_per_year: int
) -> float:
    return float(qc_performance.sortino_ratio(returns, risk_free_per_period, periods_per_year))


def _sortino_ratio_reference(
    returns: npt.NDArray[np.float64], risk_free_per_period: float, periods_per_year: int
) -> float:
    return float(
        _empyrical_sortino_ratio(
            returns, required_return=risk_free_per_period, annualization=periods_per_year
        )
    )


_SORTINO_RATIO = ConformanceTarget(
    name="sortino_ratio",
    tier=1,
    oracle="empyrical-reloaded",
    citation=(
        "Sortino, F.A. and van der Meer, R. (1991), 'Downside Risk.' "
        "Journal of Portfolio Management, 17(4), 27-31."
    ),
    compare=ScalarComparator(),
    quantcore=_sortino_ratio_quantcore,
    reference=_sortino_ratio_reference,
)


# --- Calmar ratio: standard (CAGR-based) and arithmetic -------------------------------------


def _calmar_ratio_quantcore(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    return float(qc_performance.calmar_ratio(returns, periods_per_year, return_method="cagr"))


def _calmar_ratio_reference(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    return float(_empyrical_calmar_ratio(returns, annualization=periods_per_year))


_CALMAR_RATIO = ConformanceTarget(
    name="calmar_ratio",
    tier=1,
    oracle="empyrical-reloaded",
    citation="Young, T.W. (1991), 'Calmar Ratio: A Smoother Tool.' Futures, 20(1), 40.",
    compare=ScalarComparator(),
    quantcore=_calmar_ratio_quantcore,
    reference=_calmar_ratio_reference,
)


def _calmar_ratio_arithmetic_quantcore(
    returns: npt.NDArray[np.float64], periods_per_year: int
) -> float:
    return float(qc_performance.calmar_ratio(returns, periods_per_year, return_method="arithmetic"))


def _calmar_ratio_arithmetic_reference(
    returns: npt.NDArray[np.float64], periods_per_year: int
) -> float:
    mdd = float(-_empyrical_max_drawdown(returns))
    if mdd == 0.0:
        return float(np.inf)
    return float(np.mean(returns)) * periods_per_year / mdd


_CALMAR_RATIO_ARITHMETIC = ConformanceTarget(
    name="calmar_ratio_arithmetic",
    tier=1,
    oracle="hand-computed (mean(r)*ppy / |max drawdown|)",
    citation=(
        'quantcore\'s non-standard Calmar variant (`return_method="arithmetic"`): the numerator '
        "is the arithmetic mean return times periods per year, not the CAGR."
    ),
    compare=ScalarComparator(),
    quantcore=_calmar_ratio_arithmetic_quantcore,
    reference=_calmar_ratio_arithmetic_reference,
)


# --- Information ratio ------------------------------------------------------------------------


def _information_ratio_quantcore(
    returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int,
) -> float:
    return float(qc_performance.information_ratio(returns, benchmark_returns, periods_per_year))


def _information_ratio_reference(
    returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int,
) -> float:
    # empyrical.excess_sharpe is per-period; annualised by `sqrt(time)` to match quantcore's
    # annualised information_ratio.
    per_period = float(_empyrical_excess_sharpe(returns, benchmark_returns))
    return per_period * float(np.sqrt(periods_per_year))


_INFORMATION_RATIO = ConformanceTarget(
    name="information_ratio",
    tier=1,
    oracle="empyrical-reloaded (excess_sharpe, annualised)",
    citation=(
        "Bacon, C.R. (2008), Practical Portfolio Performance Measurement and Attribution, "
        "2nd ed., ch. 5."
    ),
    compare=ScalarComparator(),
    quantcore=_information_ratio_quantcore,
    reference=_information_ratio_reference,
)


# --- Hit rate and profit factor, on trade P&L --------------------------------------------------


def _hit_rate_quantcore(pnl: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.hit_rate(pnl, zero_policy="exclude"))


def _hit_rate_reference(pnl: npt.NDArray[np.float64]) -> float:
    nonzero = pnl[pnl != 0.0]
    if nonzero.size == 0:
        return float("nan")
    return float(np.mean(nonzero > 0.0))


_HIT_RATE = ConformanceTarget(
    name="hit_rate",
    tier=1,
    oracle="hand-computed; quantstats.stats.win_rate (second oracle)",
    citation="Standard trade-level win-rate definition; see AFML ch. 14.",
    compare=ScalarComparator(),
    quantcore=_hit_rate_quantcore,
    reference=_hit_rate_reference,
)


def _profit_factor_quantcore(pnl: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.profit_factor(pnl))


def _profit_factor_reference(pnl: npt.NDArray[np.float64]) -> float:
    gains = float(np.sum(pnl[pnl > 0.0]))
    losses = float(np.sum(pnl[pnl < 0.0]))
    if losses == 0.0:
        return float(np.inf)
    return gains / abs(losses)


_PROFIT_FACTOR = ConformanceTarget(
    name="profit_factor",
    tier=1,
    oracle="hand-computed; quantstats.stats.profit_factor (second oracle)",
    citation="Standard trade-level profit-factor definition; see AFML ch. 14.",
    compare=ScalarComparator(),
    quantcore=_profit_factor_quantcore,
    reference=_profit_factor_reference,
)


# --- Rolling Sharpe -----------------------------------------------------------------------------


def _rolling_sharpe_quantcore(
    returns: npt.NDArray[np.float64],
    window: int,
    risk_free_per_period: float,
    periods_per_year: int,
) -> npt.NDArray[np.float64]:
    full = qc_performance.rolling_sharpe(returns, window, risk_free_per_period, periods_per_year)
    result: npt.NDArray[np.float64] = full[window - 1 :]
    return result


def _rolling_sharpe_reference(
    returns: npt.NDArray[np.float64],
    window: int,
    risk_free_per_period: float,
    periods_per_year: int,
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = np.asarray(
        _empyrical_roll_sharpe_ratio(
            returns,
            window,
            risk_free=risk_free_per_period,
            annualization=periods_per_year,
        ),
        dtype=np.float64,
    )
    return result


_ROLLING_SHARPE = ConformanceTarget(
    name="rolling_sharpe",
    tier=1,
    oracle="empyrical-reloaded",
    citation="Sharpe, W.F. (1966), 'Mutual Fund Performance.' Journal of Business, 39(1), 119-138.",
    compare=ArrayComparator(),
    quantcore=_rolling_sharpe_quantcore,
    reference=_rolling_sharpe_reference,
)


# --- Drawdown, payoff and tail ratios against quantstats -----------------------------------------
# Each oracle passes its input as a `pd.Series` with a throwaway index.


def _series(values: npt.NDArray[np.float64]) -> pd.Series:
    # quantstats needs a DatetimeIndex (e.g. for `to_drawdown_series`); the dates never affect
    # the values, only the type.
    index = pd.date_range("2000-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index)


# quantstats ships `py.typed` but several wrapped functions lack annotations; declaring their
# types here satisfies strict mode's disallow-untyped-calls without `Any`, `cast` or ignores.
_qs_ulcer_index: Callable[[pd.Series], float] = qs_stats.ulcer_index
_qs_ulcer_performance_index: Callable[[pd.Series, int], float] = qs_stats.ulcer_performance_index
_qs_gain_to_pain_ratio: Callable[[pd.Series, int], float] = qs_stats.gain_to_pain_ratio
_qs_common_sense_ratio: Callable[[pd.Series], float] = qs_stats.common_sense_ratio
_qs_risk_of_ruin: Callable[[pd.Series], float] = qs_stats.risk_of_ruin
_qs_serenity_index: Callable[[pd.Series, int], float] = qs_stats.serenity_index
_qs_outlier_win_ratio: Callable[[pd.Series], float] = qs_stats.outlier_win_ratio
_qs_outlier_loss_ratio: Callable[[pd.Series], float] = qs_stats.outlier_loss_ratio
_qs_cpc_index: Callable[[pd.Series], float] = qs_stats.cpc_index


def _ulcer_index_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.ulcer_index(returns))


def _ulcer_index_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_ulcer_index(_series(returns)))


_ULCER_INDEX = ConformanceTarget(
    name="ulcer_index",
    tier=1,
    oracle="quantstats.stats.ulcer_index",
    citation=(
        "Martin, P.G. and McCann, B.B. (1989), The Investor's Guide to Fidelity Funds, Wiley; "
        "with quantstats' n - 1 divisor and fractional drawdowns."
    ),
    compare=ScalarComparator(),
    quantcore=_ulcer_index_quantcore,
    reference=_ulcer_index_reference,
)


def _ulcer_performance_index_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.ulcer_performance_index(returns))


def _ulcer_performance_index_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_ulcer_performance_index(_series(returns), 0))


_ULCER_PERFORMANCE_INDEX = ConformanceTarget(
    name="ulcer_performance_index",
    tier=1,
    oracle="quantstats.stats.ulcer_performance_index",
    citation=(
        "Martin, P.G. and McCann, B.B. (1989), The Investor's Guide to Fidelity Funds, Wiley; "
        "with quantstats' numerator, the total compounded return, not annualised."
    ),
    compare=ScalarComparator(),
    quantcore=_ulcer_performance_index_quantcore,
    reference=_ulcer_performance_index_reference,
)


def _gain_to_pain_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.gain_to_pain_ratio(returns))


def _gain_to_pain_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_gain_to_pain_ratio(_series(returns), 0))


_GAIN_TO_PAIN = ConformanceTarget(
    name="gain_to_pain",
    tier=1,
    oracle="quantstats.stats.gain_to_pain_ratio",
    citation=(
        "Schwager, J.D. (2012), Hedge Fund Market Wizards, Wiley, Appendix A; applied to the "
        "returns at the frequency given, not resampled to months."
    ),
    compare=ScalarComparator(),
    quantcore=_gain_to_pain_quantcore,
    reference=_gain_to_pain_reference,
)


def _common_sense_ratio_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.common_sense_ratio(returns))


def _common_sense_ratio_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_common_sense_ratio(_series(returns)))


_COMMON_SENSE_RATIO = ConformanceTarget(
    name="common_sense_ratio",
    tier=1,
    oracle="quantstats.stats.common_sense_ratio",
    citation=(
        "quantstats.stats.common_sense_ratio = profit_factor * tail_ratio; attributed to Laurent "
        "Bernut."
    ),
    compare=ScalarComparator(),
    quantcore=_common_sense_ratio_quantcore,
    reference=_common_sense_ratio_reference,
)


def _risk_of_ruin_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.risk_of_ruin(returns))


def _risk_of_ruin_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_risk_of_ruin(_series(returns)))


_RISK_OF_RUIN = ConformanceTarget(
    name="risk_of_ruin",
    tier=1,
    oracle="quantstats.stats.risk_of_ruin",
    citation=(
        "quantstats.stats.risk_of_ruin = ((1 - win_rate) / (1 + win_rate)) ** n, the fixed-stake "
        "gambler's-ruin formula (Feller 1968, vol. 1, ch. XIV) with the win rate in place of the "
        "advantage and the number of periods in place of the units of capital."
    ),
    compare=ScalarComparator(),
    quantcore=_risk_of_ruin_quantcore,
    reference=_risk_of_ruin_reference,
)


def _serenity_index_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.serenity_index(returns))


def _serenity_index_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_serenity_index(_series(returns), 0))


_SERENITY_INDEX = ConformanceTarget(
    name="serenity_index",
    tier=1,
    oracle="quantstats.stats.serenity_index",
    citation=(
        "quantstats.stats.serenity_index, after the KeyQuant white paper quantstats names as its "
        "source (KeyQuant_WhitePaper_APT_Part1.pdf); conditional drawdown at risk from Chekhlov, "
        "A., Uryasev, S. and Zabarankin, M. (2005), 'Drawdown Measure in Portfolio Optimization', "
        "International Journal of Theoretical and Applied Finance, 8(1), 13-58."
    ),
    compare=ScalarComparator(),
    quantcore=_serenity_index_quantcore,
    reference=_serenity_index_reference,
)


def _outlier_win_ratio_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.outlier_win_ratio(returns))


def _outlier_win_ratio_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_outlier_win_ratio(_series(returns)))


_OUTLIER_WIN_RATIO = ConformanceTarget(
    name="outlier_win_ratio",
    tier=1,
    oracle="quantstats.stats.outlier_win_ratio",
    citation=(
        "99th percentile return over the mean non-negative return; quantstats' own definition, "
        "no published source."
    ),
    compare=ScalarComparator(),
    quantcore=_outlier_win_ratio_quantcore,
    reference=_outlier_win_ratio_reference,
)


def _outlier_loss_ratio_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.outlier_loss_ratio(returns))


def _outlier_loss_ratio_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_outlier_loss_ratio(_series(returns)))


_OUTLIER_LOSS_RATIO = ConformanceTarget(
    name="outlier_loss_ratio",
    tier=1,
    oracle="quantstats.stats.outlier_loss_ratio",
    citation=(
        "1st percentile return over the mean negative return; quantstats' own definition, no "
        "published source."
    ),
    compare=ScalarComparator(),
    quantcore=_outlier_loss_ratio_quantcore,
    reference=_outlier_loss_ratio_reference,
)


def _cpc_index_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.cpc_index(returns))


def _cpc_index_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(_qs_cpc_index(_series(returns)))


_CPC_INDEX = ConformanceTarget(
    name="cpc_index",
    tier=1,
    oracle="quantstats.stats.cpc_index",
    citation=(
        "quantstats.stats.cpc_index = profit_factor * win_rate * win_loss_ratio; attributed to "
        "Sunny J. Harris; no primary text is cited."
    ),
    compare=ScalarComparator(),
    quantcore=_cpc_index_quantcore,
    reference=_cpc_index_reference,
)
