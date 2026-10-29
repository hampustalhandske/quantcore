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

`egarch_fit` and `gjr_garch_fit` estimate (omega, alpha, gamma, beta) for
their respective recursions via Gaussian maximum likelihood, following the
same multi-start SLSQP approach as `volatility.fit_garch_11`, and the same
convergence convention: all three GARCH-family fitters raise
`volatility.ConvergenceError` (carrying the best attempt found) rather
than returning a partial or unconverged result silently.

References:
    Nelson, D.B. (1991), "Conditional Heteroskedasticity in Asset Returns: A
    New Approach." (EGARCH(1,1); also the source for the Gaussian
    log-likelihood used by `egarch_fit`'s maximum-likelihood estimation.)

    Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993), "On the
    Relation Between the Expected Value and the Volatility of the Nominal
    Excess Return on Stocks." (GJR-GARCH(1,1) leverage effect; also the
    source for the Gaussian log-likelihood used by `gjr_garch_fit`'s
    maximum-likelihood estimation.)

    J.P. Morgan/Reuters (1996), *RiskMetrics -- Technical Document* (4th
    ed.). (EWMA variance.)

    See docs/REFERENCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

from quantcore.risk.volatility import (
    _fit_with_arch,
    _presample,
    _raise_unless_converged,
    variance_backcast,
)

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
    presample_variance: float | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the EGARCH(1,1) log-conditional-variance series.

    Args:
        returns: Mean-adjusted return residuals (epsilon_t).
        omega: Constant in the log-variance equation.
        alpha: ARCH effect -- response to the magnitude of the shock.
        gamma: Leverage / asymmetry term.
        beta: Persistence (must satisfy |beta| < 1 for stationarity).
        presample_variance: The variance assumed before the first observation,
            strictly positive: ln(sigma_0^2) is then
            `omega + beta * ln(presample_variance)`. `None` (default) uses
            `variance_backcast`'s rule on `returns` (`arch`'s backcast), the
            start `egarch_fit` fits with, so the default path is the fitted
            path. Pass `exp(omega / (1 - beta))` to seed ln(sigma_0^2) at
            `omega / (1 - beta)`.

    Returns:
        ln(sigma_t^2), the log-conditional-variance series, same length as
        `returns`. Exponentiate the result if variance is needed.

    References:
        Nelson (1991), "Conditional Heteroskedasticity in Asset Returns: A
        New Approach." See docs/REFERENCES.md.
    """
    _validate_egarch_inputs(returns, omega, beta)
    presample = _presample(returns, presample_variance)
    initial = float(np.exp(omega + beta * np.log(presample)))
    return _egarch_11_variance(returns, omega, alpha, gamma, beta, initial)


@numba.njit(cache=True)
def _egarch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = np.nan,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    log_variance = np.empty(n, dtype=np.float64)
    if np.isnan(initial_variance):
        log_variance[0] = omega / (1.0 - beta)
    else:
        log_variance[0] = np.log(initial_variance)

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


def _egarch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the EGARCH(1,1) recursion.

    Correctness reference for `_egarch_11_variance`'s Numba kernel — see
    `test_egarch.py` for the agreement test between the two.
    """
    n = returns.shape[0]
    log_variance = np.empty(n, dtype=np.float64)
    log_variance[0] = (
        omega / (1.0 - beta) if np.isnan(initial_variance) else np.log(initial_variance)
    )

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


def _validate_fit_returns(returns: npt.NDArray[np.float64]) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")


@dataclass(frozen=True)
class EGARCHParams:
    """Fitted EGARCH(1,1) parameters and fit-quality diagnostics.

    Attributes:
        omega: Constant in the log-variance equation.
        alpha: ARCH effect -- response to the magnitude of the shock.
        gamma: Leverage / asymmetry term.
        beta: Persistence.
        log_likelihood: Gaussian log-likelihood attained at the reported
            parameters.
        presample_variance: The pre-sample variance the fit used
            (`volatility.variance_backcast`); `None` when the parameters
            were not produced by `egarch_fit`. Pass it as
            `egarch_11_variance`'s `presample_variance` to reproduce the path
            the parameters were fitted to.
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    presample_variance: float | None = None


def egarch_fit(returns: npt.NDArray[np.float64]) -> EGARCHParams:
    """Estimate EGARCH(1,1) parameters (omega, alpha, gamma, beta) via Gaussian MLE, using `arch`.

    `returns` is the raw (not pre-mean-adjusted) return series. It is
    demeaned internally with the sample mean, `mu_hat = returns.mean()`,
    giving residuals `epsilon = returns - mu_hat`, which are then fit to the
    `egarch_11_variance` log-variance recursion (Nelson 1991):

        ln(sigma_t^2) = omega + beta*ln(sigma_{t-1}^2)
                      + alpha*(|z_{t-1}| - E[|z|]) + gamma*z_{t-1}

    started from `arch`'s backcast (`volatility.variance_backcast`), which is
    returned as `presample_variance`. Parameters are estimated by `arch`
    (`vol="EGARCH"`, `p=1, o=1, q=1`) by maximizing the Gaussian
    log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject to `arch`'s stationarity bound on beta; EGARCH's log-variance
    formulation places no sign restriction on omega, alpha, or gamma. The
    fit runs on the residuals divided by their standard deviation and the
    constant is converted back, `omega + (1 - beta) * ln(sd^2)`, so the
    result does not depend on the units of `returns`.

    Like `volatility.fit_garch_11`, this function raises
    `volatility.ConvergenceError` (a `RuntimeError` subclass) when `arch`
    does not report convergence, rather than returning a partial result;
    catch it and read its `params`/`log_likelihood`/`optimizer_message` if
    you need the best attempt anyway.

    EGARCH(1,1) has four free parameters and no closed-form corner solution
    to fall back on, so short windows are noticeably harder to fit reliably
    than GARCH(1,1)/GJR-GARCH(1,1). Empirically, both 60- and 120-observation
    windows (a trading quarter/half-year) show a non-convergence rate around
    5-8%; this drops close to 0% only once the window reaches several
    hundred observations. There is no window length at which convergence is
    guaranteed, only increasingly likely — callers working with short
    windows should be prepared to handle `ConvergenceError`.

    Args:
        returns: Raw (not mean-adjusted) return series. Reliable convergence
            generally needs several hundred observations; shorter windows
            (e.g. 60-120) still converge most of the time but fail
            noticeably more often.

    Returns:
        `EGARCHParams` with the fitted (omega, alpha, gamma, beta) and the
        attained log-likelihood.

    Raises:
        ValueError: If `returns` is empty or has no variation.
        ConvergenceError: If `arch` does not report convergence (see
            `volatility.ConvergenceError`; its `params` are
            `(omega, alpha, gamma, beta)` in that order for this function).

    References:
        Nelson, D.B. (1991), "Conditional Heteroskedasticity in Asset
        Returns: A New Approach." (Gaussian log-likelihood and ML
        estimation of EGARCH(1,1).) See docs/REFERENCES.md.
    """
    _validate_fit_returns(returns)
    fitted, epsilon, spread = _fit_with_arch(returns, "egarch_fit", vol="EGARCH", o=1)
    omega_s, alpha, gamma, beta = (float(v) for v in np.asarray(fitted.params, dtype=np.float64))
    omega = omega_s + (1.0 - beta) * float(np.log(spread * spread))
    params = (omega, alpha, gamma, beta)
    feasible = all(np.isfinite(params)) and -1.0 < beta < 1.0
    _raise_unless_converged(fitted, "egarch_fit", params, bool(feasible), epsilon.size, spread)
    return EGARCHParams(
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        log_likelihood=float(fitted.loglikelihood) - epsilon.size * float(np.log(spread)),
        presample_variance=variance_backcast(returns),
    )


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
    presample_variance: float | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the GJR-GARCH(1,1) conditional variance series.

    Args:
        returns: Mean-adjusted residuals (epsilon_t).
        omega: Constant (omega > 0).
        alpha: ARCH term (alpha >= 0).
        gamma: Leverage term (alpha + gamma >= 0).
        beta: GARCH persistence (alpha + gamma/2 + beta < 1 for stationarity).
        presample_variance: The variance and squared residual assumed before
            the first observation, strictly positive: sigma_0^2 is then
            `omega + (alpha + gamma/2 + beta) * presample_variance`. `None`
            (default) uses `variance_backcast`'s rule on `returns` (`arch`'s
            backcast), the start `gjr_garch_fit` fits with, so the default
            path is the fitted path. Pass the unconditional variance
            `omega / (1 - alpha - gamma/2 - beta)` to start the recursion
            there.

    Returns:
        Conditional variance series sigma_t^2, same length as `returns`.

    References:
        Glosten, Jagannathan & Runkle (1993). See docs/REFERENCES.md.
    """
    _validate_gjr_garch_inputs(returns, omega, alpha, gamma, beta)
    presample = _presample(returns, presample_variance)
    initial = omega + (alpha + gamma / 2.0 + beta) * presample
    return _gjr_garch_11_variance(returns, omega, alpha, gamma, beta, initial)


@numba.njit(cache=True)
def _gjr_garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = np.nan,
) -> npt.NDArray[np.float64]:
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    if np.isnan(initial_variance):
        variance[0] = omega / (1.0 - alpha - gamma / 2.0 - beta)
    else:
        variance[0] = initial_variance

    for t in range(1, n):
        bad_news = 1.0 if returns[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + gamma * returns[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )

    return variance


def _gjr_garch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the GJR-GARCH(1,1) recursion.

    Correctness reference for `_gjr_garch_11_variance`'s Numba kernel — see
    `test_egarch.py` for the agreement test between the two.
    """
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = (
        omega / (1.0 - alpha - gamma / 2.0 - beta)
        if np.isnan(initial_variance)
        else initial_variance
    )

    for t in range(1, n):
        bad_news = 1.0 if returns[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + gamma * returns[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )

    return variance


@dataclass(frozen=True)
class GJRGARCHParams:
    """Fitted GJR-GARCH(1,1) parameters and fit-quality diagnostics.

    Attributes:
        omega: Constant (omega > 0).
        alpha: ARCH term (alpha >= 0).
        gamma: Leverage term (alpha + gamma >= 0).
        beta: GARCH persistence.
        log_likelihood: Gaussian log-likelihood attained at the reported
            parameters.
        presample_variance: The pre-sample variance the fit used
            (`volatility.variance_backcast`); `None` when the parameters
            were not produced by `gjr_garch_fit`. Pass it as
            `gjr_garch_11_variance`'s `presample_variance` to reproduce the
            path the parameters were fitted to.
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    presample_variance: float | None = None


def gjr_garch_fit(returns: npt.NDArray[np.float64]) -> GJRGARCHParams:
    """Estimate GJR-GARCH(1,1) parameters (omega, alpha, gamma, beta) via Gaussian MLE.

    Fitted with `arch`. `returns` is the raw (not pre-mean-adjusted) return
    series. It is demeaned internally with the sample mean,
    `mu_hat = returns.mean()`, giving residuals `epsilon = returns - mu_hat`,
    which are then fit to the `gjr_garch_11_variance` recursion (Glosten,
    Jagannathan & Runkle 1993):

        sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + gamma*epsilon_{t-1}^2*I_{t-1}
                  + beta*sigma_{t-1}^2

    started from `arch`'s backcast (`volatility.variance_backcast`), which is
    returned as `presample_variance`. Parameters are estimated by `arch`
    (`vol="GARCH"`, `p=1, o=1, q=1`) by maximizing the Gaussian
    log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject to `arch`'s constraints: omega > 0, alpha >= 0, alpha + gamma >=
    0, and alpha + gamma/2 + beta < 1 for covariance stationarity. The fit
    runs on the residuals divided by their standard deviation and `omega`
    is converted back, so the result does not depend on the units of
    `returns`.

    Like `volatility.fit_garch_11`, this function raises
    `volatility.ConvergenceError` (a `RuntimeError` subclass) when `arch`
    does not report convergence, rather than returning a partial result;
    catch it and read its `params`/`log_likelihood`/`optimizer_message` if
    you need the best attempt anyway.

    Args:
        returns: Raw (not mean-adjusted) return series.

    Returns:
        `GJRGARCHParams` with the fitted (omega, alpha, gamma, beta) and
        the attained log-likelihood.

    Raises:
        ValueError: If `returns` is empty or has no variation.
        ConvergenceError: If `arch` does not report convergence (see
            `volatility.ConvergenceError`; its `params` are
            `(omega, alpha, gamma, beta)` in that order for this function).

    References:
        Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993), "On the
        Relation Between the Expected Value and the Volatility of the
        Nominal Excess Return on Stocks." (Gaussian log-likelihood and ML
        estimation of GJR-GARCH(1,1).) See docs/REFERENCES.md.
    """
    _validate_fit_returns(returns)
    fitted, epsilon, spread = _fit_with_arch(returns, "gjr_garch_fit", vol="GARCH", o=1)
    omega_s, alpha, gamma, beta = (float(v) for v in np.asarray(fitted.params, dtype=np.float64))
    omega = omega_s * spread * spread
    params = (omega, alpha, gamma, beta)
    feasible = (
        all(np.isfinite(params))
        and omega > 0.0
        and alpha >= 0.0
        and alpha + gamma >= 0.0
        and alpha + gamma / 2.0 + beta < 1.0
    )
    _raise_unless_converged(fitted, "gjr_garch_fit", params, bool(feasible), epsilon.size, spread)
    return GJRGARCHParams(
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        log_likelihood=float(fitted.loglikelihood) - epsilon.size * float(np.log(spread)),
        presample_variance=variance_backcast(returns),
    )


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
