"""Portfolio performance and risk-adjusted return metrics.

    Sharpe ratio (Sharpe 1966):
      SR = (mean(r) - r_f) / std(r) * sqrt(periods_per_year), std uses ddof=1

    Sortino ratio (Sortino & van der Meer 1991):
      Sortino = (mean(r) - r_f) / downside_deviation * sqrt(periods_per_year)
      downside_deviation = sqrt(mean(min(r - r_f, 0)^2))

    Calmar ratio (Young 1991):
      Calmar = annualized_return / |max_drawdown|
      annualized_return: CAGR by default (compound annual growth rate,
      matching empyrical/QuantStats/Young (1991) as commonly applied) —
      ending_value^(periods_per_year/n) - 1, ending_value = prod(1+r).
      The arithmetic mean * periods_per_year variant is available via
      `return_method="arithmetic"`.

    Information ratio:
      IR = mean(r - r_benchmark) / std(r - r_benchmark) * sqrt(periods_per_year)

    Maximum drawdown:
      MDD = max over t of (peak_{0..t} - value_t) / peak_{0..t}

    Component VaR (Gaussian parametric):
      CVaR_i = w_i * (Sigma*w)_i / sqrt(w^T Sigma w) * z_alpha
      where z_alpha = Phi^-1(confidence_level); sum(CVaR_i) = portfolio VaR

References:
    Sharpe, W.F. (1966), "Mutual Fund Performance." Sortino, F.A. and van der
    Meer, R. (1991), "Downside Risk." Young, T.W. (1991), "Calmar Ratio: A
    Smoother Tool." Component VaR follows the parametric (delta-normal)
    convention of J.P. Morgan/Reuters (1996), also used in `risk/var.py`.
    See docs/REFERENCES.md.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

AnnualizedReturnMethod = Literal["cagr", "arithmetic"]


def _validate_returns(returns: npt.NDArray[np.float64]) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")


def _validate_sharpe_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float,
    periods_per_year: int,
) -> None:
    _validate_returns(returns)


def sharpe_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio of `returns` in excess of `risk_free_rate`.

    Args:
        returns: Periodic simple returns (e.g. daily).
        risk_free_rate: A **per-period** rate (e.g. the daily risk-free
            rate if `returns` is daily), subtracted from each period's
            return before averaging — not an annual rate. Matches
            `empyrical.sharpe_ratio`'s `risk_free` convention (its own
            docstring: "Constant daily risk-free return"). Does NOT match
            `quantstats.stats.sharpe`'s `rf`, which is annualized and
            converted to a per-period rate internally — passing an annual
            rate here directly, expecting that same auto-conversion,
            silently produces a far too large excess-return adjustment.
            Convert an annual rate `r_annual` to match `returns`' frequency
            first if that's what you have, e.g.
            `(1 + r_annual) ** (1 / periods_per_year) - 1` for a
            compounding conversion, or `r_annual / periods_per_year` for a
            simple pro-rata one (match whichever convention `returns`
            itself uses).
        periods_per_year: Number of periods in a year (e.g. 252 for daily).
    """
    _validate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    return _sharpe_ratio(returns, risk_free_rate, periods_per_year)


def _sharpe_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    excess_mean = float(np.mean(returns)) - risk_free_rate
    std = float(np.std(returns, ddof=1))
    return float(excess_mean / std * np.sqrt(periods_per_year))


def _validate_sortino_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float,
    periods_per_year: int,
) -> None:
    _validate_returns(returns)


def sortino_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio; returns `np.inf` if downside deviation is 0.

    Downside deviation = sqrt(mean((returns - risk_free_rate) clipped above
    at 0, squared)) — the mean is over *all* observations (gains count as
    zero deviation, not excluded), matching `empyrical.downside_risk`'s
    convention.

    Args:
        returns: Periodic simple returns (e.g. daily).
        risk_free_rate: A **per-period** rate — see `sharpe_ratio`'s
            docstring; the same per-period-not-annual convention applies
            here, both as the excess-return subtrahend and as the downside
            deviation's target/threshold.
        periods_per_year: Number of periods in a year (e.g. 252 for daily).
    """
    _validate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    return _sortino_ratio(returns, risk_free_rate, periods_per_year)


def _sortino_ratio(
    returns: npt.NDArray[np.float64],
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    excess = returns - risk_free_rate
    downside_deviation = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside_deviation == 0.0:
        return float(np.inf)
    return float(np.mean(excess)) / downside_deviation * float(np.sqrt(periods_per_year))


def _validate_maximum_drawdown(returns: npt.NDArray[np.float64]) -> None:
    _validate_returns(returns)


def maximum_drawdown(returns: npt.NDArray[np.float64]) -> float:
    """Maximum drawdown of the cumulative return path, as a positive fraction.

    Measured against the running peak of the wealth path seeded with its
    starting capital of 1.0 (see `drawdown_series`), so
    `maximum_drawdown(np.array([-0.5])) == 0.5`.
    """
    _validate_maximum_drawdown(returns)
    return _maximum_drawdown(returns)


def _maximum_drawdown(returns: npt.NDArray[np.float64]) -> float:
    return float(np.max(_drawdown_series(returns)))


def drawdown_series(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Drawdown at every period, as a positive fraction (0 at a new high).

    drawdown_t = (running_peak_t - cumulative_t) / running_peak_t, where
    cumulative_t = prod(1 + returns[:t+1]) and running_peak_t is the running
    maximum of the wealth path *including its starting value of 1.0* (the
    capital in place before the first return; Bacon 2008), i.e.
    running_peak_t = max(1.0, cumulative_0, ..., cumulative_t). A loss on the
    first period is therefore a drawdown, matching
    `empyrical.stats.drawdown_series`.
    `maximum_drawdown(returns) == drawdown_series(returns).max()`.

    Args:
        returns: Periodic simple returns.

    Returns:
        Drawdown series, same length as `returns`, each entry in [0, 1).
    """
    _validate_returns(returns)
    return _drawdown_series(returns)


def _drawdown_series(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # The wealth path starts at 1.0 *before* the first return is applied, so
    # the running peak is seeded with that starting capital: a decline on the
    # very first period (or any run of losses before the path first exceeds
    # its starting value) is a drawdown from initial capital, not a new high.
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], cumulative)))[1:]
    result: npt.NDArray[np.float64] = (peak - cumulative) / peak
    return result


def time_under_water(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Number of consecutive periods (inclusive) since the last new equity high.

    0 at a period that is itself a new high (drawdown == 0); otherwise the
    count increments by 1 each period the drawdown persists, resetting to 0
    the moment a new high is reached.

    Args:
        returns: Periodic simple returns.

    Returns:
        Integer series, same length as `returns`.
    """
    _validate_returns(returns)
    return _time_under_water(returns)


def _time_under_water(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    drawdowns = _drawdown_series(returns)
    underwater = drawdowns > 0.0
    result = np.zeros(returns.shape[0], dtype=np.int64)
    counter = 0
    for t in range(returns.shape[0]):
        counter = counter + 1 if underwater[t] else 0
        result[t] = counter
    return result


def max_drawdown_duration(returns: npt.NDArray[np.float64]) -> int:
    """Longest historical drawdown duration, in periods.

    The maximum number of consecutive periods spent below a prior equity
    high, over the whole series (see `time_under_water`).

    Args:
        returns: Periodic simple returns.

    Returns:
        Longest drawdown duration, in periods (0 if the series never dips
        below its running peak).
    """
    _validate_returns(returns)
    return int(np.max(_time_under_water(returns)))


def _validate_cagr(returns: npt.NDArray[np.float64], periods_per_year: int) -> None:
    _validate_returns(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")


def cagr(returns: npt.NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Compound annual growth rate of the cumulative return path.

    CAGR = ending_value^(periods_per_year / n) - 1, ending_value =
    prod(1 + returns), n = len(returns). This is the standard "annualized
    return" figure (empyrical's `annual_return`, QuantStats' `cagr`,
    Young's (1991) Calmar ratio as commonly applied) — the geometric growth
    rate consistent with actually compounding the observed returns, as
    opposed to the arithmetic mean scaled by frequency (see
    `annualized_return`'s `"arithmetic"` method).
    """
    _validate_cagr(returns, periods_per_year)
    return _cagr(returns, periods_per_year)


def _cagr(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    ending_value = float(np.prod(1.0 + returns))
    if ending_value < 0.0:
        raise ValueError(
            "cumulative return path went negative (a simple return below -1.0 "
            "in the input); CAGR is undefined"
        )
    n_years = returns.size / periods_per_year
    if ending_value == 0.0:
        return -1.0
    return float(ending_value ** (1.0 / n_years) - 1.0)


def _validate_annualized_return(
    returns: npt.NDArray[np.float64], periods_per_year: int, method: AnnualizedReturnMethod
) -> None:
    _validate_returns(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if method not in ("cagr", "arithmetic"):
        raise ValueError('method must be "cagr" or "arithmetic"')


def annualized_return(
    returns: npt.NDArray[np.float64],
    periods_per_year: int = 252,
    method: AnnualizedReturnMethod = "cagr",
) -> float:
    """Annualized return of `returns`, by either of two conventions.

    Args:
        returns: Periodic simple returns.
        periods_per_year: Number of periods in a year (e.g. 252 for daily).
        method: `"cagr"` (default) — see `cagr`; the standard convention.
            `"arithmetic"` — mean(returns) * periods_per_year, a simpler but
            upward-biased approximation for volatile series (Jensen's
            inequality: E[compound growth] < (1 + E[r])^n for r with
            variance > 0), kept for compatibility with callers that already
            rely on it.
    """
    _validate_annualized_return(returns, periods_per_year, method)
    return _annualized_return(returns, periods_per_year, method)


def _annualized_return(
    returns: npt.NDArray[np.float64], periods_per_year: int, method: AnnualizedReturnMethod
) -> float:
    if method == "arithmetic":
        return float(np.mean(returns)) * periods_per_year
    return _cagr(returns, periods_per_year)


def _validate_calmar_ratio(
    returns: npt.NDArray[np.float64], periods_per_year: int, return_method: AnnualizedReturnMethod
) -> None:
    _validate_returns(returns)
    _validate_annualized_return(returns, periods_per_year, return_method)


def calmar_ratio(
    returns: npt.NDArray[np.float64],
    periods_per_year: int = 252,
    return_method: AnnualizedReturnMethod = "cagr",
) -> float:
    """Annualized return over maximum drawdown; returns `np.inf` if drawdown is 0.

    Args:
        returns: Periodic simple returns.
        periods_per_year: Number of periods in a year (e.g. 252 for daily).
        return_method: How the numerator is annualized — `"cagr"` (default;
            matches empyrical/QuantStats/Young (1991) as commonly applied)
            or `"arithmetic"` (the prior default: mean(returns) *
            periods_per_year). See `annualized_return`.
    """
    _validate_calmar_ratio(returns, periods_per_year, return_method)
    return _calmar_ratio(returns, periods_per_year, return_method)


def _calmar_ratio(
    returns: npt.NDArray[np.float64], periods_per_year: int, return_method: AnnualizedReturnMethod
) -> float:
    mdd = _maximum_drawdown(returns)
    if mdd == 0.0:
        return float(np.inf)
    return _annualized_return(returns, periods_per_year, return_method) / abs(mdd)


def _validate_information_ratio(
    returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int,
) -> None:
    _validate_returns(returns)
    if benchmark_returns.size == 0:
        raise ValueError("benchmark_returns must be non-empty")
    if returns.shape != benchmark_returns.shape:
        raise ValueError("returns and benchmark_returns must have the same shape")


def information_ratio(
    returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualized information ratio of `returns` relative to `benchmark_returns`."""
    _validate_information_ratio(returns, benchmark_returns, periods_per_year)
    return _information_ratio(returns, benchmark_returns, periods_per_year)


def _information_ratio(
    returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
    periods_per_year: int,
) -> float:
    diff = returns - benchmark_returns
    tracking_error = float(np.std(diff, ddof=1))
    return float(np.mean(diff)) / tracking_error * float(np.sqrt(periods_per_year))


def _validate_rolling_sharpe(
    returns: npt.NDArray[np.float64],
    window: int,
    risk_free_rate: float,
    periods_per_year: int,
) -> None:
    _validate_returns(returns)
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > returns.size:
        raise ValueError("window must not exceed the length of returns")


def rolling_sharpe(
    returns: npt.NDArray[np.float64],
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> npt.NDArray[np.float64]:
    """Rolling annualized Sharpe ratio; the first `window - 1` entries are NaN."""
    _validate_rolling_sharpe(returns, window, risk_free_rate, periods_per_year)
    return _rolling_sharpe(returns, window, risk_free_rate, periods_per_year)


def _rolling_sharpe(
    returns: npt.NDArray[np.float64],
    window: int,
    risk_free_rate: float,
    periods_per_year: int,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        result[i] = _sharpe_ratio(returns[i - window + 1 : i + 1], risk_free_rate, periods_per_year)
    return result


def _validate_hit_rate(returns: npt.NDArray[np.float64]) -> None:
    _validate_returns(returns)


HitRateZeroPolicy = Literal["loss", "win", "exclude"]


def hit_rate(returns: npt.NDArray[np.float64], zero_policy: HitRateZeroPolicy = "loss") -> float:
    """Fraction of periods with a positive return.

    A zero return is not a "win" by any of these conventions; `zero_policy`
    only controls whether it's counted as a loss (the denominator includes
    it, the default — matches the prior, only behavior) or excluded from
    the denominator entirely, which matters on a series with many exactly-
    zero periods (e.g. a daily equity-curve series padded over non-trading
    days, or an options book with no fills on some days) where those
    periods aren't really "trades that lost" so much as "no trade
    happened" — `hit_rate` as a trade statistic usually means the latter.

    Args:
        returns: Periodic simple returns.
        zero_policy: `"loss"` (default; matches the prior, only behavior —
            zero-return periods count in the denominator as a non-win,
            i.e. as if they were a loss). `"exclude"` — zero-return periods
            are dropped entirely; the rate is wins / (wins + losses),
            ignoring flat periods. `"win"` — zero-return periods count as
            a win.

    Returns:
        Fraction of periods counted as a win, in [0, 1]. `NaN` if
        `zero_policy="exclude"` and every period is exactly zero.
    """
    _validate_hit_rate(returns)
    if zero_policy not in ("loss", "win", "exclude"):
        raise ValueError('zero_policy must be "loss", "win", or "exclude"')
    return _hit_rate(returns, zero_policy)


def _hit_rate(returns: npt.NDArray[np.float64], zero_policy: HitRateZeroPolicy) -> float:
    if zero_policy == "exclude":
        nonzero = returns[returns != 0.0]
        if nonzero.size == 0:
            return float("nan")
        return float(np.mean(nonzero > 0.0))
    if zero_policy == "win":
        return float(np.mean(returns >= 0.0))
    return float(np.mean(returns > 0.0))


def _validate_profit_factor(returns: npt.NDArray[np.float64]) -> None:
    _validate_returns(returns)


def profit_factor(returns: npt.NDArray[np.float64]) -> float:
    """Sum of gains over absolute sum of losses; `np.inf` if there are no losses.

    Zero-return periods contribute to neither the gains nor the losses sum
    (they're excluded from both `returns > 0.0` and `returns < 0.0`) — on a
    series with many exactly-zero periods (e.g. non-trading days padded
    into a daily series), this already behaves like a trade-level
    statistic without needing a separate option, unlike `hit_rate`, whose
    denominator (a period *count*, not a *sum*) does need `zero_policy` to
    get the same effect.
    """
    _validate_profit_factor(returns)
    return _profit_factor(returns)


def _profit_factor(returns: npt.NDArray[np.float64]) -> float:
    gains = float(np.sum(returns[returns > 0.0]))
    losses = float(np.sum(returns[returns < 0.0]))
    if losses == 0.0:
        return float(np.inf)
    return gains / abs(losses)


def _validate_component_var(
    weights: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    confidence_level: float,
) -> None:
    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2-D array")
    if weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("weights length must match cov_matrix dimension")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")


def component_var(
    weights: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    confidence_level: float,
) -> npt.NDArray[np.float64]:
    """Parametric Gaussian component VaR per asset; components sum to portfolio VaR.

    Returns an all-zero vector, rather than dividing by zero, when
    `portfolio_std` is at or below 1e-12 (e.g. an all-cash/all-zero-weight
    book).
    """
    _validate_component_var(weights, cov_matrix, confidence_level)
    return _component_var(weights, cov_matrix, confidence_level)


_COMPONENT_VAR_STD_EPSILON = 1e-12


def _component_var(
    weights: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    confidence_level: float,
) -> npt.NDArray[np.float64]:
    z_alpha = float(norm.ppf(confidence_level))
    portfolio_variance = float(weights @ cov_matrix @ weights)
    portfolio_std = np.sqrt(portfolio_variance)
    if portfolio_std <= _COMPONENT_VAR_STD_EPSILON:
        return np.zeros_like(weights)
    marginal = cov_matrix @ weights
    result: npt.NDArray[np.float64] = (weights * marginal) / portfolio_std * z_alpha
    return result
