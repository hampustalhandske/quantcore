"""Conditional volatility modeling: ARCH / GARCH.

Engle (1982) introduced the ARCH(q) model, in which conditional variance
depends on past squared residuals:

    sigma_t^2 = omega + sum_{i=1}^{q} alpha_i * epsilon_{t-i}^2

Bollerslev (1986) generalized this to GARCH(p, q) by adding lagged
conditional variance terms. This module implements the standard GARCH(1,1)
special case, the industry-default volatility model:

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

where epsilon_t = r_t - mu is the mean-adjusted return residual, and
omega > 0, alpha >= 0, beta >= 0, with alpha + beta < 1 required for
covariance stationarity (mean-reverting variance).

References:
    Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
    Estimates of the Variance of United Kingdom Inflation." *Econometrica*,
    50(4), 987-1007. (Foundational ARCH model.)

    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
    Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
    (GARCH(1,1).)

    See REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_garch_inputs(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if omega <= 0.0:
        raise ValueError("omega must be strictly positive")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1 for covariance stationarity")


def garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Compute the GARCH(1,1) conditional variance series.

        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Args:
        returns: Mean-adjusted return series (epsilon_t).
        omega: Long-run variance weight (omega > 0).
        alpha: ARCH term weight — reaction to the most recent shock
            (alpha >= 0).
        beta: GARCH term weight — persistence of past variance (beta >= 0).
            Requires alpha + beta < 1 for covariance stationarity.

    Returns:
        Conditional variance series sigma_t^2, same length as `returns`.

    References:
        Bollerslev (1986), "Generalized Autoregressive Conditional
        Heteroskedasticity." See REFERENCES.md.
    """
    _validate_garch_inputs(returns, omega, alpha, beta)
    return _garch_11_variance(returns, omega, alpha, beta)


def _garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)

    # Seed the recursion with the model-implied long-run (unconditional)
    # variance omega / (1 - alpha - beta), the stationary variance of the
    # GARCH(1,1) process under alpha + beta < 1 (Bollerslev 1986). This is
    # not pinned down by the recursion formula itself, since sigma_t^2
    # depends on sigma_{t-1}^2 and there is no sigma_0^2 in the input data;
    # anchoring to the theoretical stationary variance (rather than, e.g.,
    # the sample variance of `returns`) keeps the series purely a function
    # of the given model parameters.
    variance[0] = omega / (1.0 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

    return variance

