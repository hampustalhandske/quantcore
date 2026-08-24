"""Portfolio weight optimization: mean-variance, risk parity, and Kelly sizing.

Mean-variance (Markowitz 1952):
    min_w  w^T Sigma w  s.t.  w^T mu = mu_target, sum(w) = 1, w >= 0
    or for unconstrained tangency:
    w_tangency = Sigma^{-1}(mu - r_f) / 1^T Sigma^{-1}(mu - r_f)

Risk parity / Equal Risk Contribution (Maillard et al. 2010):
    RC_i = w_i * (Sigma*w)_i / (w^T Sigma w)  (risk contribution of asset i)
    Objective: minimize sum_{i,j} (RC_i - RC_j)^2
               i.e., make all RC_i equal to 1/k

Kelly fraction (Kelly 1956, single-asset):
    f* = (mu - r_f) / sigma^2

References:
    Markowitz, H. (1952), "Portfolio Selection." Kelly, J.L. (1956), "A New
    Interpretation of Information Rate." Maillard, Roncalli & Teïletche
    (2010), "The Properties of Equally Weighted Risk Contribution
    Portfolios." See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import LinearConstraint, minimize


def _validate_cov_matrix(cov_matrix: npt.NDArray[np.float64]) -> None:
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2D array")


def _validate_min_variance_inputs(cov_matrix: npt.NDArray[np.float64]) -> None:
    _validate_cov_matrix(cov_matrix)


def min_variance_weights(
    cov_matrix: npt.NDArray[np.float64],
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute the minimum-variance portfolio: min w^T Sigma w s.t. sum(w) = 1.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1.
    """
    _validate_min_variance_inputs(cov_matrix)
    return _min_variance_weights(cov_matrix, allow_short)


def _min_variance_weights(
    cov_matrix: npt.NDArray[np.float64],
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    x0 = np.full(k, 1.0 / k)
    bounds = None if allow_short else [(0.0, None)] * k
    constraints = [LinearConstraint(np.ones(k), 1.0, 1.0)]

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(w @ cov_matrix @ w)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return np.asarray(result.x, dtype=np.float64)


def _validate_mean_variance_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    if expected_returns.ndim != 1 or expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError("expected_returns must be a 1D array matching cov_matrix's dimension")
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")


def mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    """Maximize w^T mu - 0.5*risk_aversion*w^T Sigma w s.t. sum(w) = 1, w >= 0.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        risk_aversion: Risk-aversion coefficient lambda >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1.
    """
    _validate_mean_variance_inputs(expected_returns, cov_matrix, risk_aversion)
    return _mean_variance_weights(expected_returns, cov_matrix, risk_aversion)


def _mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    x0 = np.full(k, 1.0 / k)
    bounds = [(0.0, None)] * k
    constraints = [LinearConstraint(np.ones(k), 1.0, 1.0)]

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(-(w @ expected_returns) + 0.5 * risk_aversion * (w @ cov_matrix @ w))

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return np.asarray(result.x, dtype=np.float64)


def _validate_risk_parity_inputs(cov_matrix: npt.NDArray[np.float64]) -> None:
    _validate_cov_matrix(cov_matrix)


def risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the equal-risk-contribution (risk parity) portfolio.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k).

    Returns:
        Portfolio weights of shape (k,), summing to 1, with equal risk
        contributions.
    """
    _validate_risk_parity_inputs(cov_matrix)
    return _risk_parity_weights(cov_matrix)


def _risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    x0 = np.full(k, 1.0 / k)
    bounds = [(1e-8, None)] * k
    constraints = [LinearConstraint(np.ones(k), 1.0, 1.0)]

    def objective(w: npt.NDArray[np.float64]) -> float:
        marginal = cov_matrix @ w
        contributions = w * marginal
        diffs = contributions[:, None] - contributions[None, :]
        return float(np.sum(diffs**2))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-16},
    )
    weights = np.asarray(result.x, dtype=np.float64)
    return weights / weights.sum()


def _validate_kelly_fraction_inputs(expected_return: float, variance: float) -> None:
    if variance <= 0.0:
        raise ValueError("variance must be strictly positive")


def kelly_fraction(expected_return: float, variance: float) -> float:
    """Compute the single-asset Kelly fraction f* = expected_return / variance.

    Args:
        expected_return: Excess expected return E[r] - r_f.
        variance: Return variance sigma^2 (must be strictly positive).

    Returns:
        Kelly fraction, clamped to [-1, 1].
    """
    _validate_kelly_fraction_inputs(expected_return, variance)
    return _kelly_fraction(expected_return, variance)


def _kelly_fraction(expected_return: float, variance: float) -> float:
    fraction = expected_return / variance
    return float(np.clip(fraction, -1.0, 1.0))
