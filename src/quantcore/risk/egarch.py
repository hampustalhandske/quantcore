"""Asymmetric conditional volatility models: EGARCH, GJR-GARCH, and EWMA.

Standard GARCH(1,1) (see `risk/volatility.py`) treats positive and negative
shocks symmetrically. The models here capture the leverage effect -- the
empirical tendency of volatility to rise more after negative shocks than
after positive ones of the same size.

EGARCH(1,1) (Nelson 1991):
    ln(sigma_t^2) = omega + beta*ln(sigma_{t-1}^2)
                  + alpha*(|z_{t-1}| - E[|z|]) + gamma*z_{t-1}
    where z_t = epsilon_t / sigma_t, E[|z|] = sqrt(2/pi) for standard normal

GJR-GARCH(1,1) (Glosten, Jagannathan & Runkle 1993):
    sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + gamma*epsilon_{t-1}^2*I_{t-1}
              + beta*sigma_{t-1}^2
    where I_{t-1} = 1 if epsilon_{t-1} < 0 (bad news), else 0

EWMA variance (RiskMetrics):
    sigma_t^2 = lambda*sigma_{t-1}^2 + (1-lambda)*epsilon_{t-1}^2

References:
    Nelson, D.B. (1991), "Conditional Heteroskedasticity in Asset Returns: A
    New Approach." (EGARCH(1,1).)

    Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993), "On the
    Relation Between the Expected Value and the Volatility of the Nominal
    Excess Return on Stocks." (GJR-GARCH(1,1) leverage effect.)

    J.P. Morgan/Reuters (1996), *RiskMetrics -- Technical Document* (4th
    ed.). (EWMA variance.)

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt

_E_ABS_Z = float(np.sqrt(2.0 / np.pi))


def _validate_egarch_inputs(
    returns: npt.NDArray[np.float64],
    omega: float,
    beta: float,
) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if not (-1.0 < beta < 1.0):
        raise ValueError("beta must be strictly between -1 and 1 for stationarity")


def egarch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Compute the EGARCH(1,1) log-conditional-variance series.

    Args:
        returns: Mean-adjusted return residuals (epsilon_t).
        omega: Constant in the log-variance equation.
        alpha: ARCH effect -- response to the magnitude of the shock.
        gamma: Leverage / asymmetry term.
        beta: Persistence (must satisfy |beta| < 1 for stationarity).

    Returns:
        ln(sigma_t^2), the log-conditional-variance series, same length as
        `returns`. Exponentiate the result if variance is needed.

    References:
        Nelson (1991), "Conditional Heteroskedasticity in Asset Returns: A
        New Approach." See docs/REFERENCES.md.
    """
    _validate_egarch_inputs(returns, omega, beta)
    return _egarch_11_variance(returns, omega, alpha, gamma, beta)


@numba.njit(cache=True)
def _egarch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    log_variance = np.empty(n, dtype=np.float64)
    log_variance[0] = omega / (1.0 - beta)

    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_variance[t - 1]))
        z_prev = returns[t - 1] / sigma_prev
        log_variance[t] = (
            omega
            + beta * log_variance[t - 1]
            + alpha * (np.abs(z_prev) - _E_ABS_Z)
            + gamma * z_prev
        )

    return log_variance


def _validate_gjr_garch_inputs(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if omega <= 0.0:
        raise ValueError("omega must be strictly positive")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if alpha + gamma < 0.0:
        raise ValueError("alpha + gamma must be non-negative")
    if alpha + gamma / 2.0 + beta >= 1.0:
        raise ValueError("alpha + gamma / 2 + beta must be < 1 for covariance stationarity")


def gjr_garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Compute the GJR-GARCH(1,1) conditional variance series.

    Args:
        returns: Mean-adjusted residuals (epsilon_t).
        omega: Constant (omega > 0).
        alpha: ARCH term (alpha >= 0).
        gamma: Leverage term (alpha + gamma >= 0).
        beta: GARCH persistence (alpha + gamma/2 + beta < 1 for stationarity).

    Returns:
        Conditional variance series sigma_t^2, same length as `returns`.

    References:
        Glosten, Jagannathan & Runkle (1993). See docs/REFERENCES.md.
    """
    _validate_gjr_garch_inputs(returns, omega, alpha, gamma, beta)
    return _gjr_garch_11_variance(returns, omega, alpha, gamma, beta)


@numba.njit(cache=True)
def _gjr_garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = omega / (1.0 - alpha - gamma / 2.0 - beta)

    for t in range(1, n):
        bad_news = 1.0 if returns[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + gamma * returns[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )

    return variance


def _validate_ewma_inputs(returns: npt.NDArray[np.float64], lambda_: float) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if not (0.0 < lambda_ < 1.0):
        raise ValueError("lambda_ must be strictly between 0 and 1")


def ewma_variance(
    returns: npt.NDArray[np.float64],
    lambda_: float,
) -> npt.NDArray[np.float64]:
    """Compute the EWMA (RiskMetrics) conditional variance series.

    Args:
        returns: Mean-adjusted residuals.
        lambda_: Decay factor, typically 0.94 (daily) or 0.97 (monthly).
            Must be strictly between 0 and 1.

    Returns:
        EWMA variance series, same length as `returns`, seeded with
        `returns[0] ** 2`.

    References:
        J.P. Morgan/Reuters (1996), RiskMetrics Technical Document. See
        docs/REFERENCES.md.
    """
    _validate_ewma_inputs(returns, lambda_)
    return _ewma_variance(returns, lambda_)


@numba.njit(cache=True)
def _ewma_variance(
    returns: npt.NDArray[np.float64],
    lambda_: float,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = returns[0] ** 2

    for t in range(1, n):
        variance[t] = lambda_ * variance[t - 1] + (1.0 - lambda_) * returns[t - 1] ** 2

    return variance
