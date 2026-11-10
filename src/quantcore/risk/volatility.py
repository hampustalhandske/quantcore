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

import warnings
from typing import Any, Literal

import numba
import numpy as np
import numpy.typing as npt

from quantcore.errors import UndefinedResultError, _finite_result


class ConvergenceError(RuntimeError):
    """Maximum-likelihood optimization failed to converge to a fully
    feasible solution from any starting point.

    `fit_garch_11`, `egarch_fit`, `gjr_garch_fit` and
    `quantcore.time_series.arima_fit` all raise this in the analogous case
    (see each function's docstring) — no silent fallback
    and no partial-result return value; the fit is not trustworthy enough
    to hand back as if it were a normal answer. Unlike a plain
    `RuntimeError`, it carries the best attempt actually found across all
    starting points (by objective value, preferring a feasible point over
    an infeasible one), so a caller that wants to inspect a best-effort
    fit despite the failure can still do so by catching this and reading
    `params`/`log_likelihood`/`optimizer_message`, rather than receiving
    a `converged=False` result as if it were usable.

    Attributes:
        params: The best attempt's raw parameter tuple, in the same order
            the raising function's docstring describes, or `None` if no
            starting point produced any usable (finite-objective) result
            at all.
        log_likelihood: The best attempt's Gaussian log-likelihood, or
            `None` under the same condition as `params`.
        optimizer_message: The underlying `scipy.optimize.minimize`
            result's `message` for the best attempt, or `None` under the
            same condition as `params`.

    It pickles and unpickles with every attribute, and with anything else
    set on the instance (such as notes added with `add_note`).
    """

    def __init__(
        self,
        message: str,
        params: tuple[float, ...] | None,
        log_likelihood: float | None,
        optimizer_message: str | None,
    ) -> None:
        super().__init__(message)
        self.params = params
        self.log_likelihood = log_likelihood
        self.optimizer_message = optimizer_message

    def __reduce__(self) -> tuple[Any, ...]:
        # BaseException's default replays only `args` (the message) into __init__, which also
        # needs params, log_likelihood and optimizer_message.
        constructor_args = (self.args[0], self.params, self.log_likelihood, self.optimizer_message)
        return (type(self), constructor_args, self.__dict__)


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
    presample_variance: float | None = None,
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
        presample_variance: The variance and squared residual assumed before
            the first observation, strictly positive: sigma_0^2 is then
            `omega + (alpha + beta) * presample_variance`. `None` (default)
            uses `variance_backcast`'s rule on `returns` (`arch`'s backcast),
            the start `fit_garch_11` fits with, so the default path is the
            fitted path. Pass the unconditional variance
            `omega / (1 - alpha - beta)` to start the recursion there.

    Returns:
        Conditional variance series sigma_t^2, same length as `returns`.

    References:
        Bollerslev (1986), "Generalized Autoregressive Conditional
        Heteroskedasticity." See REFERENCES.md.
    """
    _validate_garch_inputs(returns, omega, alpha, beta)
    presample = _presample(returns, presample_variance)
    return _garch_11_variance(returns, omega, alpha, beta, omega + (alpha + beta) * presample)


def _backcast(residuals: npt.NDArray[np.float64]) -> float:
    """`arch`'s pre-sample variance: weights `0.94**j` on the first `min(75, n)` squares."""
    tau = min(75, residuals.shape[0])
    weights = 0.94 ** np.arange(tau)
    return float(np.sum(residuals[:tau] ** 2 * (weights / weights.sum())))


def _presample(residuals: npt.NDArray[np.float64], presample_variance: float | None) -> float:
    """The pre-sample variance to use: `presample_variance`, or the backcast of `residuals`."""
    if presample_variance is None:
        return _backcast(residuals)
    if not (np.isfinite(presample_variance) and presample_variance > 0.0):
        raise ValueError("presample_variance must be finite and strictly positive")
    return float(presample_variance)


@numba.njit(cache=True)
def _garch_11_variance_numba_kernel(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float = np.nan,
) -> npt.NDArray[np.float64]:
    """Numba-accelerated GARCH(1,1) recursion, identical to the NumPy reference."""
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    if np.isnan(initial_variance):
        variance[0] = omega / (1.0 - alpha - beta)
    else:
        variance[0] = initial_variance

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

    return variance


def _garch_11_variance(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    # No silent fallback: if the Numba kernel fails to compile/run, this
    # raises rather than transparently falling back to the NumPy reference.
    return _garch_11_variance_numba_kernel(returns, omega, alpha, beta, initial_variance)


def _validate_fit_garch_inputs(returns: npt.NDArray[np.float64]) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")


def variance_backcast(returns: npt.NDArray[np.float64]) -> float:
    """Pre-sample variance `arch` uses when fitting a GARCH-family model to `returns`.

    The mean-adjusted returns' first squared values, `min(75, n)` of them,
    averaged with exponentially decaying weights `0.94**j`. `fit_garch_11`,
    `egarch_fit` and `gjr_garch_fit` start their recursions from this value
    (as the variance and squared residual before the first observation), and
    `garch_11_variance`, `egarch_11_variance` and `gjr_garch_11_variance`
    use the same rule on the residuals they are given when no
    `presample_variance` is passed.

    Args:
        returns: Raw (not mean-adjusted) return series, non-empty.

    Returns:
        The backcast variance, in the units of `returns` squared.

    Raises:
        ValueError: If `returns` is empty or contains NaN or infinity.
        UndefinedResultError: The squared residuals overflow float64.
    """
    _validate_fit_garch_inputs(returns)
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values")
    return _finite_result(_backcast(returns - returns.mean()), "variance backcast")


def _fit_with_arch(
    returns: npt.NDArray[np.float64], caller: str, vol: Literal["GARCH", "EGARCH"], o: int
) -> tuple[Any, npt.NDArray[np.float64], float]:
    """Fit a Gaussian zero-mean `arch` model to the mean-adjusted `returns`.

    The residuals are fitted in units of their standard deviation, where
    `arch`'s optimizer is well scaled. `arch` is imported on first use, so
    importing `quantcore.risk` does not load it (and pandas with it).

    Returns:
        The fitted `arch` result, the mean-adjusted returns and their
        standard deviation.

    Raises:
        UndefinedResultError: `returns` has zero range, or the mean-adjusted returns are all
            zero.
    """
    import arch

    epsilon = returns - returns.mean()
    spread = float(np.std(epsilon))
    if np.ptp(returns) == 0.0 or not spread > 0.0:
        raise UndefinedResultError(f"returns have no variation; {caller} cannot be fit to them")
    model = arch.arch_model(
        epsilon / spread, mean="Zero", vol=vol, p=1, o=o, q=1, dist="normal", rescale=False
    )
    with warnings.catch_warnings():
        # Non-convergence is reported by raising, not by a warning.
        from arch.utility.exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = model.fit(disp="off", options={"maxiter": 1000})
    return fitted, epsilon, spread


def _raise_unless_converged(
    fitted: Any,
    caller: str,
    params: tuple[float, ...],
    feasible: bool,
    n_obs: int,
    spread: float,
) -> None:
    """Raise `ConvergenceError` carrying `params` unless `arch` converged to a feasible point.

    `arch` can report convergence at a point outside the model's parameter
    region (for EGARCH, `|beta| >= 1`); `feasible` says whether `params` lie
    inside it, and a point that does not is treated as non-convergence.
    """
    if fitted.convergence_flag != 0 or not feasible:
        raise ConvergenceError(
            f"{caller}: arch's maximum-likelihood optimization did not converge "
            "to a feasible point",
            params=params,
            # Likelihood of the residuals in their own units: the fit is on
            # residuals divided by `spread`, which shifts the log-density of
            # each observation by log(spread).
            log_likelihood=float(fitted.loglikelihood) - n_obs * float(np.log(spread)),
            optimizer_message=str(fitted.optimization_result.message),
        )


def fit_garch_11(returns: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Estimate GARCH(1,1) parameters (omega, alpha, beta) via Gaussian MLE, using `arch`.

    `returns` is the raw (not pre-mean-adjusted) return series. It is
    demeaned internally with the sample mean, `mu_hat = returns.mean()`,
    giving residuals `epsilon = returns - mu_hat`, which are then fit to the
    standard GARCH(1,1) recursion:

        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters are estimated by maximizing the Gaussian log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject to the covariance-stationarity constraints omega > 0, alpha >=
    0, beta >= 0, alpha + beta < 1. The pre-sample variance is `arch`'s backcast
    (`variance_backcast`), and `arch` runs one SLSQP solve on the residuals
    divided by their standard deviation; `omega` is converted back
    afterwards, so the result does not depend on the units of `returns`.

    Args:
        returns: Raw (not mean-adjusted) return series.

    Returns:
        Tuple (omega_hat, alpha_hat, beta_hat) of fitted GARCH(1,1)
        parameters. `garch_11_variance(returns - returns.mean(), *params,
        presample_variance=variance_backcast(returns))` (the default start of
        `garch_11_variance`) is the variance path
        they were fitted to.

    Raises:
        ValueError: If `returns` is empty.
        UndefinedResultError: If `returns` has no variation (`np.ptp == 0`).
        ConvergenceError: If `arch`'s optimization does not report
            convergence (a `RuntimeError` subclass — see `ConvergenceError`).
            No silent fallback and no partial result is returned in this
            case; catch `ConvergenceError` and read its `params` if you
            need the best attempt anyway. `egarch_fit`/`gjr_garch_fit`
            (`risk/egarch.py`) raise the same exception in the analogous
            case.

    References:
        Bollerslev, T. (1986), "Generalized Autoregressive Conditional
        Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
        (Gaussian log-likelihood and ML estimation of GARCH(1,1).) See
        REFERENCES.md.
    """
    _validate_fit_garch_inputs(returns)
    fitted, epsilon, spread = _fit_with_arch(returns, "fit_garch_11", vol="GARCH", o=0)
    omega, alpha, beta = (float(v) for v in np.asarray(fitted.params, dtype=np.float64))
    params = (omega * spread * spread, alpha, beta)
    feasible = (
        all(np.isfinite(params))
        and params[0] > 0.0
        and alpha >= 0.0
        and beta >= 0.0
        and alpha + beta < 1.0
    )
    _raise_unless_converged(fitted, "fit_garch_11", params, bool(feasible), epsilon.size, spread)
    return params
