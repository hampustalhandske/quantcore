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
from scipy.optimize import LinearConstraint, OptimizeResult, minimize


def _validate_cov_matrix(cov_matrix: npt.NDArray[np.float64]) -> None:
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2D array")


def _require_slsqp_success(result: OptimizeResult, func_name: str) -> None:
    """No silent fallbacks (see CLAUDE.md): a caller must be told when
    SLSQP didn't converge, rather than silently receiving whatever
    infeasible/non-optimal point it stopped at.
    """
    if not result.success:
        raise RuntimeError(
            f"{func_name}: SLSQP optimization did not converge "
            f"(status={result.status}, message={result.message!r})"
        )


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

    Raises:
        RuntimeError: If the SLSQP optimization does not converge. No
            silent fallback is returned in this case.
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
    _require_slsqp_success(result, "min_variance_weights")
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
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Maximize w^T mu - 0.5*risk_aversion*w^T Sigma w s.t. sum(w) = 1, w >= 0.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        risk_aversion: Risk-aversion coefficient lambda >= 0.
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1.

    Raises:
        RuntimeError: If the SLSQP optimization does not converge. No
            silent fallback is returned in this case.
    """
    _validate_mean_variance_inputs(expected_returns, cov_matrix, risk_aversion)
    return _mean_variance_weights(expected_returns, cov_matrix, risk_aversion, allow_short)


def _mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    x0 = np.full(k, 1.0 / k)
    bounds = None if allow_short else [(0.0, None)] * k
    constraints = [LinearConstraint(np.ones(k), 1.0, 1.0)]

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(-(w @ expected_returns) + 0.5 * risk_aversion * (w @ cov_matrix @ w))

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    _require_slsqp_success(result, "mean_variance_weights")
    return np.asarray(result.x, dtype=np.float64)


def _validate_unconstrained_mean_variance_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    if expected_returns.ndim != 1 or expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError("expected_returns must be a 1D array matching cov_matrix's dimension")
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be strictly positive")


def unconstrained_mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    """Solve the unconstrained mean-variance first-order condition Sigma @ w = mu / lambda.

    This is the closed-form maximizer of w^T mu - 0.5*risk_aversion*w^T Sigma w
    with NO constraints imposed at all: no budget constraint (sum(w) == 1), no
    long-only constraint. The resulting weights need not sum to 1, may be
    negative (short positions), and may exceed 1x gross exposure (leverage).
    Use this for a raw tangency-style allocation (e.g. as a prior or building
    block in a downstream optimizer that layers its own constraints); use
    `mean_variance_weights` instead when a fully-invested, optionally
    long-only portfolio is required.

    Because this is a direct linear solve rather than an SLSQP call, it is
    substantially faster than `mean_variance_weights`.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        risk_aversion: Risk-aversion coefficient lambda, must be strictly
            positive (weights are divided by it).

    Returns:
        Unconstrained portfolio weights of shape (k,). Not guaranteed to sum
        to 1.
    """
    _validate_unconstrained_mean_variance_inputs(expected_returns, cov_matrix, risk_aversion)
    return _unconstrained_mean_variance_weights(expected_returns, cov_matrix, risk_aversion)


def _unconstrained_mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    # np.linalg.solve over an explicit np.linalg.inv: forming the inverse
    # explicitly squares the condition number's effect on rounding error,
    # whereas solve dispatches to a direct LU factorization of cov_matrix.
    return np.linalg.solve(cov_matrix, expected_returns) / risk_aversion


def _validate_l1_turnover_penalized_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    if expected_returns.ndim != 1 or expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError("expected_returns must be a 1D array matching cov_matrix's dimension")
    if previous_weights.ndim != 1 or previous_weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("previous_weights must be a 1D array matching cov_matrix's dimension")
    if abs(previous_weights.sum() - 1.0) > 1e-6:
        raise ValueError("previous_weights must sum to 1")
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative")
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")


def l1_turnover_penalized_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Mean-variance weights penalized for turnover away from previous_weights.

    Maximizes w^T mu - 0.5*risk_aversion*w^T Sigma w - (cost_bps/10000)*||w -
    previous_weights||_1, s.t. sum(w) = 1, w >= 0 (unless allow_short). As
    cost_bps -> 0 this reduces to mean_variance_weights.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        previous_weights: Current portfolio weights before rebalancing, shape
            (k,), must sum to 1.
        cost_bps: Proportional transaction cost in basis points per unit of
            L1 turnover (must be non-negative).
        risk_aversion: Risk-aversion coefficient lambda >= 0.
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1.

    Raises:
        RuntimeError: If the SLSQP optimization does not converge. No
            silent fallback is returned in this case.
    """
    _validate_l1_turnover_penalized_inputs(
        expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion
    )
    return _l1_turnover_penalized_weights(
        expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion, allow_short
    )


def _l1_turnover_penalized_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    # ||w - previous_weights||_1 is non-differentiable at w_i == previous_weights_i,
    # which breaks SLSQP's finite-difference gradient near the (likely optimal, for
    # cost_bps > 0) no-trade point. Splitting the deviation into non-negative
    # buy/sell slacks (w = previous_weights + buy - sell) makes both the objective
    # and constraints smooth and linear in the slacks, since an optimal solution
    # never has buy_i > 0 and sell_i > 0 simultaneously (that would waste cost for
    # no change in w_i).
    x0 = np.zeros(2 * k)
    bounds = [(0.0, None)] * (2 * k)

    def _weights_from_slacks(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        buy = x[:k]
        sell = x[k:]
        return previous_weights + buy - sell

    budget_row = np.concatenate([np.ones(k), -np.ones(k)])
    budget_target = 1.0 - previous_weights.sum()
    constraints = [LinearConstraint(budget_row, budget_target, budget_target)]
    if not allow_short:
        non_negativity_matrix = np.concatenate([np.eye(k), -np.eye(k)], axis=1)
        constraints.append(LinearConstraint(non_negativity_matrix, -previous_weights, np.inf))

    cost_rate = cost_bps / 10000.0

    def objective(x: npt.NDArray[np.float64]) -> float:
        w = _weights_from_slacks(x)
        turnover = x.sum()
        return float(
            -(w @ expected_returns)
            + 0.5 * risk_aversion * (w @ cov_matrix @ w)
            + cost_rate * turnover
        )

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    _require_slsqp_success(result, "l1_turnover_penalized_weights")
    return np.asarray(_weights_from_slacks(result.x), dtype=np.float64)


def _validate_risk_parity_inputs(cov_matrix: npt.NDArray[np.float64]) -> None:
    _validate_cov_matrix(cov_matrix)


def risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the equal-risk-contribution (risk parity) portfolio.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k).

    Returns:
        Portfolio weights of shape (k,), summing to 1, with equal risk
        contributions.

    Raises:
        RuntimeError: If the SLSQP optimization does not converge. No
            silent fallback is returned in this case.
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
    _require_slsqp_success(result, "risk_parity_weights")
    weights = np.asarray(result.x, dtype=np.float64)
    return weights / weights.sum()


def _validate_kelly_fraction_inputs(expected_return: float, variance: float) -> None:
    if variance <= 0.0:
        raise ValueError("variance must be strictly positive")


def kelly_fraction(expected_return: float, variance: float, clip: bool = True) -> float:
    """Compute the single-asset Kelly fraction f* = expected_return / variance.

    Args:
        expected_return: Excess expected return E[r] - r_f.
        variance: Return variance sigma^2 (must be strictly positive).
        clip: If True (default), clamp the result to [-1, 1] — a leverage
            cap, not part of Kelly's (1956) original formula, which has no
            such bound (a high-edge, low-variance input can legitimately
            call for a fraction outside [-1, 1] under the textbook
            criterion). Pass `clip=False` for the unclamped textbook value.

    Returns:
        Kelly fraction; clamped to [-1, 1] if `clip` is True.
    """
    _validate_kelly_fraction_inputs(expected_return, variance)
    return _kelly_fraction(expected_return, variance, clip)


def _kelly_fraction(expected_return: float, variance: float, clip: bool) -> float:
    fraction = expected_return / variance
    if not clip:
        return float(fraction)
    return float(np.clip(fraction, -1.0, 1.0))
