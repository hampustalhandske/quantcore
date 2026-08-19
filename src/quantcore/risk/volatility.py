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
    (GARCH(1,1); also the source for the Gaussian log-likelihood used by
    `fit_garch_11`'s maximum-likelihood parameter estimation.)

    See REFERENCES.md.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
from scipy.optimize import NonlinearConstraint, minimize


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


def _garch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the GARCH(1,1) recursion."""
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


@numba.njit(cache=True)
def _garch_11_variance_numba_kernel(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Numba-accelerated GARCH(1,1) recursion, identical to the NumPy reference."""
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = omega / (1.0 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

    return variance


def _garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    # No silent fallback: if the Numba kernel fails to compile/run, this
    # raises rather than transparently falling back to the NumPy reference.
    return _garch_11_variance_numba_kernel(returns, omega, alpha, beta)


_STATIONARITY_MARGIN = 1e-6
_MIN_OMEGA = 1e-10


_INFEASIBLE_PENALTY = 1e10


def _negative_log_likelihood(
    params: npt.NDArray[np.float64],
    epsilon: npt.NDArray[np.float64],
) -> float:
    # SLSQP's line search and finite-difference Jacobian routinely probe
    # points outside the feasible region (e.g. alpha + beta >= 1, which
    # drives the recursion's seed variance omega / (1 - alpha - beta)
    # negative). Feeding such a point through log(variance) produces NaN,
    # which SLSQP cannot recover from and aborts on -- so every infeasible
    # or non-finite evaluation returns a large finite penalty instead,
    # keeping the objective well-defined everywhere the optimizer might look.
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        return _INFEASIBLE_PENALTY
    variance = _garch_11_variance_numpy(epsilon, omega, alpha, beta)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        return _INFEASIBLE_PENALTY
    log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance)
    if not np.isfinite(log_likelihood):
        return _INFEASIBLE_PENALTY
    return float(-log_likelihood)


def _validate_fit_garch_inputs(returns: npt.NDArray[np.float64]) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")


def fit_garch_11(returns: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Estimate GARCH(1,1) parameters (omega, alpha, beta) via Gaussian MLE.

    `returns` is the raw (not pre-mean-adjusted) return series. It is
    demeaned internally with the sample mean, `mu_hat = returns.mean()`,
    giving residuals `epsilon = returns - mu_hat`, which are then fit to the
    standard GARCH(1,1) recursion:

        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    seeded at `sigma_0^2 = omega / (1 - alpha - beta)`, matching
    `garch_11_variance`'s convention. Parameters are estimated by maximizing
    the Gaussian log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject to the covariance-stationarity constraints omega > 0, alpha >=
    0, beta >= 0, alpha + beta < 1.

    Args:
        returns: Raw (not mean-adjusted) return series.

    Returns:
        Tuple (omega_hat, alpha_hat, beta_hat) of fitted GARCH(1,1)
        parameters.

    Raises:
        ValueError: If `returns` is empty.
        RuntimeError: If maximum-likelihood optimization fails to converge
            to a feasible solution from any of several starting points. No
            silent fallback is returned in this case -- a non-converged fit
            is not a usable GARCH parameterization.

    References:
        Bollerslev, T. (1986), "Generalized Autoregressive Conditional
        Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
        (Gaussian log-likelihood and ML estimation of GARCH(1,1).) See
        REFERENCES.md.
    """
    _validate_fit_garch_inputs(returns)
    return _fit_garch_11(returns)


# Several starting points spanning low/typical/high persistence, so a poor
# choice of x0 for a given series doesn't silently masquerade as "the" MLE
# fit. SLSQP is a local optimizer with no restart logic of its own.
_STARTING_POINTS = (
    (0.05, 0.90),
    (0.10, 0.80),
    (0.20, 0.60),
    (0.05, 0.50),
)


def _fit_garch_11(returns: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    mu_hat = float(returns.mean())
    epsilon = returns - mu_hat
    sample_var = max(float(np.var(epsilon)), _MIN_OMEGA)

    bounds = [
        (_MIN_OMEGA, None),
        (0.0, None),
        (0.0, None),
    ]
    # alpha + beta <= 1 - margin, i.e. keep strictly inside the stationarity
    # region rather than allowing the optimizer to land exactly on the
    # boundary alpha + beta == 1.
    stationarity_constraint = NonlinearConstraint(
        lambda params: params[1] + params[2],
        -np.inf,
        1.0 - _STATIONARITY_MARGIN,
    )

    best_result = None
    for alpha0, beta0 in _STARTING_POINTS:
        omega0 = sample_var * (1.0 - alpha0 - beta0)
        omega0 = max(omega0, _MIN_OMEGA)
        x0 = np.array([omega0, alpha0, beta0], dtype=np.float64)

        result = minimize(
            _negative_log_likelihood,
            x0,
            args=(epsilon,),
            method="SLSQP",
            bounds=bounds,
            constraints=[stationarity_constraint],
        )
        if not result.success:
            continue
        omega, alpha, beta = result.x
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
            continue
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "fit_garch_11: GARCH(1,1) maximum-likelihood optimization failed to "
            "converge to a feasible solution from any starting point"
        )

    omega_raw, alpha_raw, beta_raw = best_result.x

    # Guard against boundary-adjacent floating point fuzz only (the
    # optimality check above already confirms the raw result is feasible and
    # converged) -- not a correction for non-convergence, which is raised
    # above instead of being silently patched over.
    omega_hat = max(float(omega_raw), _MIN_OMEGA)
    alpha_hat = max(float(alpha_raw), 0.0)
    beta_hat = max(float(beta_raw), 0.0)
    if alpha_hat + beta_hat >= 1.0:
        scale = (1.0 - _STATIONARITY_MARGIN) / (alpha_hat + beta_hat)
        alpha_hat *= scale
        beta_hat *= scale

    return omega_hat, alpha_hat, beta_hat
