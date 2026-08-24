"""Portfolio performance and risk-adjusted return metrics.

    Sharpe ratio (Sharpe 1966):
      SR = (mean(r) - r_f) / std(r) * sqrt(periods_per_year), std uses ddof=1

    Sortino ratio (Sortino & van der Meer 1991):
      Sortino = (mean(r) - r_f) / downside_deviation * sqrt(periods_per_year)
      downside_deviation = sqrt(mean(min(r - r_f, 0)^2))

    Calmar ratio (Young 1991):
      Calmar = annualized_return / |max_drawdown|
      annualized_return = mean(r) * periods_per_year  (arithmetic)

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

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


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
    """Annualized Sharpe ratio of `returns` in excess of `risk_free_rate`."""
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
    """Annualized Sortino ratio; returns `np.inf` if downside deviation is 0."""
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
    """Maximum drawdown of the cumulative return path, as a positive fraction."""
    _validate_maximum_drawdown(returns)
    return _maximum_drawdown(returns)


def _maximum_drawdown(returns: npt.NDArray[np.float64]) -> float:
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak
    return float(np.max(drawdowns))


def _validate_calmar_ratio(returns: npt.NDArray[np.float64], periods_per_year: int) -> None:
    _validate_returns(returns)


def calmar_ratio(
    returns: npt.NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualized return over maximum drawdown; returns `np.inf` if drawdown is 0."""
    _validate_calmar_ratio(returns, periods_per_year)
    return _calmar_ratio(returns, periods_per_year)


def _calmar_ratio(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    mdd = _maximum_drawdown(returns)
    if mdd == 0.0:
        return float(np.inf)
    annualized_return = float(np.mean(returns)) * periods_per_year
    return annualized_return / abs(mdd)


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


def hit_rate(returns: npt.NDArray[np.float64]) -> float:
    """Fraction of `returns` strictly greater than 0."""
    _validate_hit_rate(returns)
    return _hit_rate(returns)


def _hit_rate(returns: npt.NDArray[np.float64]) -> float:
    return float(np.mean(returns > 0.0))


def _validate_profit_factor(returns: npt.NDArray[np.float64]) -> None:
    _validate_returns(returns)


def profit_factor(returns: npt.NDArray[np.float64]) -> float:
    """Sum of gains over absolute sum of losses; `np.inf` if there are no losses."""
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
    """Parametric Gaussian component VaR per asset; components sum to portfolio VaR."""
    _validate_component_var(weights, cov_matrix, confidence_level)
    return _component_var(weights, cov_matrix, confidence_level)


def _component_var(
    weights: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    confidence_level: float,
) -> npt.NDArray[np.float64]:
    z_alpha = float(norm.ppf(confidence_level))
    portfolio_variance = float(weights @ cov_matrix @ weights)
    portfolio_std = np.sqrt(portfolio_variance)
    marginal = cov_matrix @ weights
    result: npt.NDArray[np.float64] = (weights * marginal) / portfolio_std * z_alpha
    return result
