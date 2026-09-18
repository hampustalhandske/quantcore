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
same multi-start SLSQP approach as `volatility.fit_garch_11`. Unlike
`fit_garch_11`, which raises if every starting point fails to converge, these
fitters are meant to support live callers that need a graceful degradation
path: they return a `converged` diagnostic on the result instead, so the
caller can decide whether to trust a given fit.

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
from scipy.optimize import NonlinearConstraint, minimize

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


def _egarch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the EGARCH(1,1) recursion.

    Used only by the MLE objective in `egarch_fit`, since scipy.optimize
    cannot call into Numba JIT functions directly as objectives (matching
    the pattern established by `volatility._garch_11_variance_numpy`).
    """
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


_EGARCH_STATIONARITY_MARGIN = 1e-6
_EGARCH_INFEASIBLE_PENALTY = 1e10


def _egarch_negative_log_likelihood(
    params: npt.NDArray[np.float64],
    epsilon: npt.NDArray[np.float64],
    omega_scale: float,
) -> float:
    # Same rationale as `volatility._negative_log_likelihood`'s
    # `_INFEASIBLE_PENALTY` comment: SLSQP routinely probes points outside
    # the feasible region (here, |beta| >= 1, which makes the recursion's
    # seed log-variance omega / (1 - beta) blow up or flip sign), so every
    # infeasible or non-finite evaluation returns a large finite penalty
    # instead of NaN, which SLSQP cannot recover from.
    #
    # `params[0]` is omega_scaled = omega / omega_scale, not omega itself.
    # Unlike `fit_garch_11`'s omega, EGARCH's omega is already a log-variance
    # intercept -- roughly log_sample_var * (1 - beta), an O(1-10) quantity
    # rather than the O(1e-4 to 1e-6) variance-scale omega of GARCH/GJR-GARCH
    # -- but it can still run several times larger than alpha/gamma (O(0.01-
    # 0.2)), which is enough to measurably hurt SLSQP's identity-initialized
    # quasi-Newton Hessian. `omega_scale = max(|log_sample_var|, 1.0)`
    # brings it in line with the other three parameters.
    omega_scaled, alpha, gamma, beta = params
    omega = omega_scaled * omega_scale
    if not (-1.0 + _EGARCH_STATIONARITY_MARGIN < beta < 1.0 - _EGARCH_STATIONARITY_MARGIN):
        return _EGARCH_INFEASIBLE_PENALTY
    # Unlike GARCH/GJR-GARCH, EGARCH places no bound on omega or alpha, so
    # SLSQP's probing routinely drives the log-variance recursion (and its
    # exponentiation back to variance) to overflow -- an expected, handled
    # condition given the isfinite checks below, not a real numerical bug,
    # so the resulting RuntimeWarnings are suppressed rather than left to
    # spam every out-of-domain evaluation.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_variance = _egarch_11_variance_numpy(epsilon, omega, alpha, gamma, beta)
        if not np.all(np.isfinite(log_variance)):
            return _EGARCH_INFEASIBLE_PENALTY
        variance = np.exp(log_variance)
        if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
            return _EGARCH_INFEASIBLE_PENALTY
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance
        )
    if not np.isfinite(log_likelihood):
        return _EGARCH_INFEASIBLE_PENALTY
    return float(-log_likelihood)


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
        converged: Whether the underlying SLSQP optimization reported
            success at a feasible point (|beta| < 1). `False` means the fit
            is a best-effort result only -- callers that need a trustworthy
            volatility estimate should fall back to a simpler model (e.g.
            EWMA) rather than use it.
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    converged: bool


# Starting points spanning a range of ARCH response, leverage sign/magnitude,
# and persistence, so a poor choice of x0 for a given series doesn't silently
# masquerade as "the" MLE fit. SLSQP is a local optimizer with no restart
# logic of its own.
_EGARCH_STARTING_POINTS = (
    (0.10, -0.05, 0.90),
    (0.05, -0.10, 0.85),
    (0.15, 0.00, 0.70),
    (0.20, -0.05, 0.60),
)


def egarch_fit(returns: npt.NDArray[np.float64]) -> EGARCHParams:
    """Estimate EGARCH(1,1) parameters (omega, alpha, gamma, beta) via Gaussian MLE.

    `returns` is the raw (not pre-mean-adjusted) return series. It is
    demeaned internally with the sample mean, `mu_hat = returns.mean()`,
    giving residuals `epsilon = returns - mu_hat`, which are then fit to the
    `egarch_11_variance` log-variance recursion (Nelson 1991):

        ln(sigma_t^2) = omega + beta*ln(sigma_{t-1}^2)
                      + alpha*(|z_{t-1}| - E[|z|]) + gamma*z_{t-1}

    seeded at `ln(sigma_0^2) = omega / (1 - beta)`, matching
    `egarch_11_variance`'s convention. Parameters are estimated by
    maximizing the Gaussian log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject only to the stationarity constraint |beta| < 1 -- unlike GARCH
    and GJR-GARCH, EGARCH's log-variance formulation places no sign or
    joint-magnitude restriction on omega, alpha, or gamma (see
    `_validate_egarch_inputs`).

    Unlike `volatility.fit_garch_11`, this function does not raise on
    optimizer non-convergence. Live callers are expected to check the
    `converged` field on the returned `EGARCHParams` and fall back to a
    simpler model themselves if it is `False`, rather than have a
    partially-usable fit hidden behind an exception.

    EGARCH(1,1) has four free parameters and no closed-form corner solution
    to fall back on, so short windows are noticeably harder to fit reliably
    than GARCH(1,1)/GJR-GARCH(1,1). Empirically, both 60- and 120-observation
    windows (a trading quarter/half-year) show a non-convergence rate around
    5-8%; this drops close to 0% only once the window reaches several
    hundred observations. Always check `converged` regardless of window
    length -- there is no length at which convergence is guaranteed, only
    increasingly likely.

    Args:
        returns: Raw (not mean-adjusted) return series. Reliable convergence
            generally needs several hundred observations; shorter windows
            (e.g. 60-120) still converge most of the time but fail
            noticeably more often -- always check `converged`.

    Returns:
        `EGARCHParams` with the fitted (omega, alpha, gamma, beta), the
        attained log-likelihood, and a `converged` flag.

    Raises:
        ValueError: If `returns` is empty.
        RuntimeError: If every starting point produced a non-finite
            objective value, leaving no usable result at all to report (not
            raised merely because SLSQP reported non-convergence -- that
            case is instead surfaced via `converged=False`).

    References:
        Nelson, D.B. (1991), "Conditional Heteroskedasticity in Asset
        Returns: A New Approach." (Gaussian log-likelihood and ML
        estimation of EGARCH(1,1).) See docs/REFERENCES.md.
    """
    _validate_fit_returns(returns)
    return _fit_egarch_11(returns)


def _fit_egarch_11(returns: npt.NDArray[np.float64]) -> EGARCHParams:
    mu_hat = float(returns.mean())
    epsilon = returns - mu_hat
    sample_var = max(float(np.var(epsilon)), 1e-10)
    log_sample_var = float(np.log(sample_var))
    omega_scale = max(abs(log_sample_var), 1.0)

    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (-1.0 + _EGARCH_STATIONARITY_MARGIN, 1.0 - _EGARCH_STATIONARITY_MARGIN),
    ]

    best_result = None
    best_feasible = False
    for alpha0, gamma0, beta0 in _EGARCH_STARTING_POINTS:
        omega0 = log_sample_var * (1.0 - beta0)
        x0 = np.array([omega0 / omega_scale, alpha0, gamma0, beta0], dtype=np.float64)

        result = minimize(
            _egarch_negative_log_likelihood,
            x0,
            args=(epsilon, omega_scale),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 500},
        )
        if not np.isfinite(result.fun):
            continue

        beta_candidate = float(result.x[3])
        feasible = (
            -1.0 + _EGARCH_STATIONARITY_MARGIN < beta_candidate < 1.0 - _EGARCH_STATIONARITY_MARGIN
        )
        is_better = (
            best_result is None
            or (feasible and not best_feasible)
            or (feasible == best_feasible and result.fun < best_result.fun)
        )
        if is_better:
            best_result = result
            best_feasible = feasible

    if best_result is None:
        raise RuntimeError(
            "egarch_fit: EGARCH(1,1) maximum-likelihood optimization produced no "
            "usable (finite-objective) result from any starting point"
        )

    omega_hat = float(best_result.x[0]) * omega_scale
    alpha_hat, gamma_hat, beta_hat = (float(v) for v in best_result.x[1:])
    converged = bool(best_result.success) and best_feasible

    return EGARCHParams(
        omega=omega_hat,
        alpha=alpha_hat,
        gamma=gamma_hat,
        beta=beta_hat,
        log_likelihood=-float(best_result.fun),
        converged=converged,
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


def _gjr_garch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the GJR-GARCH(1,1) recursion.

    Used only by the MLE objective in `gjr_garch_fit`, since scipy.optimize
    cannot call into Numba JIT functions directly as objectives (matching
    the pattern established by `volatility._garch_11_variance_numpy`).
    """
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


_GJR_GARCH_STATIONARITY_MARGIN = 1e-6
_GJR_GARCH_MIN_OMEGA = 1e-10
_GJR_GARCH_INFEASIBLE_PENALTY = 1e10
# Persistence is bounded to keep the search space well behaved -- the four
# constraints `_validate_gjr_garch_inputs` enforces bound alpha, gamma, and
# the alpha/gamma/beta combination, but leave beta itself unbounded below;
# without some floor, SLSQP's line search can wander to large negative beta
# values that are numerically pathological (huge seed variance) without ever
# violating those four constraints.
_GJR_GARCH_BETA_BOUNDS = (
    -1.0 + _GJR_GARCH_STATIONARITY_MARGIN,
    1.0 - _GJR_GARCH_STATIONARITY_MARGIN,
)


def _gjr_garch_negative_log_likelihood(
    params: npt.NDArray[np.float64],
    epsilon: npt.NDArray[np.float64],
    sample_var: float,
) -> float:
    # Same rationale as `volatility._negative_log_likelihood`'s
    # `_INFEASIBLE_PENALTY` comment: infeasible or non-finite evaluations
    # return a large finite penalty instead of NaN, which SLSQP cannot
    # recover from.
    #
    # `params[0]` is omega_scaled = omega / sample_var, not omega itself --
    # see `volatility._negative_log_likelihood` for why the raw omega/alpha
    # scale mismatch is a conditioning hazard for SLSQP.
    omega_scaled, alpha, gamma, beta = params
    omega = omega_scaled * sample_var
    if (
        omega <= 0.0
        or alpha < 0.0
        or alpha + gamma < 0.0
        or alpha + gamma / 2.0 + beta >= 1.0
        or not (_GJR_GARCH_BETA_BOUNDS[0] < beta < _GJR_GARCH_BETA_BOUNDS[1])
    ):
        return _GJR_GARCH_INFEASIBLE_PENALTY
    variance = _gjr_garch_11_variance_numpy(epsilon, omega, alpha, gamma, beta)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        return _GJR_GARCH_INFEASIBLE_PENALTY
    log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance)
    if not np.isfinite(log_likelihood):
        return _GJR_GARCH_INFEASIBLE_PENALTY
    return float(-log_likelihood)


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
        converged: Whether the underlying SLSQP optimization reported
            success at a feasible point. `False` means the fit is a
            best-effort result only -- callers that need a trustworthy
            volatility estimate should fall back to a simpler model (e.g.
            EWMA) rather than use it.
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    converged: bool


# Starting points spanning low/typical/high leverage asymmetry and
# persistence, so a poor choice of x0 for a given series doesn't silently
# masquerade as "the" MLE fit. SLSQP is a local optimizer with no restart
# logic of its own.
_GJR_GARCH_STARTING_POINTS = (
    (0.05, 0.05, 0.85),
    (0.02, 0.10, 0.80),
    (0.10, -0.05, 0.70),
    (0.05, 0.00, 0.90),
)


def gjr_garch_fit(returns: npt.NDArray[np.float64]) -> GJRGARCHParams:
    """Estimate GJR-GARCH(1,1) parameters (omega, alpha, gamma, beta) via Gaussian MLE.

    `returns` is the raw (not pre-mean-adjusted) return series. It is
    demeaned internally with the sample mean, `mu_hat = returns.mean()`,
    giving residuals `epsilon = returns - mu_hat`, which are then fit to the
    `gjr_garch_11_variance` recursion (Glosten, Jagannathan & Runkle 1993):

        sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + gamma*epsilon_{t-1}^2*I_{t-1}
                  + beta*sigma_{t-1}^2

    seeded at `sigma_0^2 = omega / (1 - alpha - gamma/2 - beta)`, matching
    `gjr_garch_11_variance`'s convention. Parameters are estimated by
    maximizing the Gaussian log-likelihood

        log L = -0.5 * sum_t [ log(2*pi) + log(sigma_t^2) + epsilon_t^2 / sigma_t^2 ]

    subject to the same constraints `gjr_garch_11_variance` already
    validates: omega > 0, alpha >= 0, alpha + gamma >= 0, and
    alpha + gamma/2 + beta < 1 for covariance stationarity.

    Unlike `volatility.fit_garch_11`, this function does not raise on
    optimizer non-convergence. Live callers are expected to check the
    `converged` field on the returned `GJRGARCHParams` and fall back to a
    simpler model themselves if it is `False`, rather than have a
    partially-usable fit hidden behind an exception.

    Args:
        returns: Raw (not mean-adjusted) return series.

    Returns:
        `GJRGARCHParams` with the fitted (omega, alpha, gamma, beta), the
        attained log-likelihood, and a `converged` flag.

    Raises:
        ValueError: If `returns` is empty.
        RuntimeError: If every starting point produced a non-finite
            objective value, leaving no usable result at all to report (not
            raised merely because SLSQP reported non-convergence -- that
            case is instead surfaced via `converged=False`).

    References:
        Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993), "On the
        Relation Between the Expected Value and the Volatility of the
        Nominal Excess Return on Stocks." (Gaussian log-likelihood and ML
        estimation of GJR-GARCH(1,1).) See docs/REFERENCES.md.
    """
    _validate_fit_returns(returns)
    return _fit_gjr_garch_11(returns)


def _fit_gjr_garch_11(returns: npt.NDArray[np.float64]) -> GJRGARCHParams:
    mu_hat = float(returns.mean())
    epsilon = returns - mu_hat
    sample_var = max(float(np.var(epsilon)), _GJR_GARCH_MIN_OMEGA)

    bounds = [
        (_GJR_GARCH_MIN_OMEGA / sample_var, None),
        (0.0, None),
        (None, None),
        _GJR_GARCH_BETA_BOUNDS,
    ]
    # alpha + gamma >= 0 (non-negativity of the "bad news" variance response).
    nonnegativity_constraint = NonlinearConstraint(
        lambda params: params[1] + params[2],
        0.0,
        np.inf,
    )
    # alpha + gamma/2 + beta <= 1 - margin, i.e. keep strictly inside the
    # stationarity region rather than allowing the optimizer to land exactly
    # on the boundary.
    stationarity_constraint = NonlinearConstraint(
        lambda params: params[1] + params[2] / 2.0 + params[3],
        -np.inf,
        1.0 - _GJR_GARCH_STATIONARITY_MARGIN,
    )

    best_result = None
    best_feasible = False
    for alpha0, gamma0, beta0 in _GJR_GARCH_STARTING_POINTS:
        omega0 = sample_var * (1.0 - alpha0 - gamma0 / 2.0 - beta0)
        omega0 = max(omega0, _GJR_GARCH_MIN_OMEGA)
        x0 = np.array([omega0 / sample_var, alpha0, gamma0, beta0], dtype=np.float64)

        result = minimize(
            _gjr_garch_negative_log_likelihood,
            x0,
            args=(epsilon, sample_var),
            method="SLSQP",
            bounds=bounds,
            constraints=[nonnegativity_constraint, stationarity_constraint],
        )
        if not np.isfinite(result.fun):
            continue

        omega_candidate = float(result.x[0]) * sample_var
        alpha_candidate, gamma_candidate, beta_candidate = (float(v) for v in result.x[1:])
        feasible = (
            omega_candidate > 0.0
            and alpha_candidate >= 0.0
            and alpha_candidate + gamma_candidate >= 0.0
            and alpha_candidate + gamma_candidate / 2.0 + beta_candidate < 1.0
        )
        is_better = (
            best_result is None
            or (feasible and not best_feasible)
            or (feasible == best_feasible and result.fun < best_result.fun)
        )
        if is_better:
            best_result = result
            best_feasible = feasible

    if best_result is None:
        raise RuntimeError(
            "gjr_garch_fit: GJR-GARCH(1,1) maximum-likelihood optimization produced no "
            "usable (finite-objective) result from any starting point"
        )

    omega_hat = float(best_result.x[0]) * sample_var
    alpha_hat, gamma_hat, beta_hat = (float(v) for v in best_result.x[1:])
    converged = bool(best_result.success) and best_feasible

    return GJRGARCHParams(
        omega=omega_hat,
        alpha=alpha_hat,
        gamma=gamma_hat,
        beta=beta_hat,
        log_likelihood=-float(best_result.fun),
        converged=converged,
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
