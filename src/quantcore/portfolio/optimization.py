"""Portfolio weight optimization: mean-variance, risk parity, and Kelly sizing.

Mean-variance (Markowitz 1952):
    min_w  w^T Sigma w  s.t.  w^T mu = mu_target, sum(w) = 1, w >= 0
    or for unconstrained tangency:
    w_tangency = Sigma^{-1}(mu - r_f) / 1^T Sigma^{-1}(mu - r_f)

Risk parity / Equal Risk Contribution (Maillard et al. 2010):
    RC_i = w_i * (Sigma*w)_i / (w^T Sigma w)  (risk contribution of asset i)
    Target: all RC_i equal to 1/k. Solved via Spinu's (2013) strictly convex
    reformulation
        min_{w > 0}  0.5 * w^T Sigma w - sum_i b_i * ln(w_i),   b_i = 1/k
    whose first-order condition w_i * (Sigma*w)_i = b_i is exactly the
    equal-risk-contribution condition once w is rescaled to sum to 1. The
    Hessian Sigma + diag(b_i / w_i^2) is positive definite for any PSD
    Sigma, so the minimiser is unique and a damped Newton iteration
    converges from any interior starting point.

Kelly fraction (Kelly 1956, single-asset):
    f* = (mu - r_f) / sigma^2

References:
    Markowitz, H. (1952), "Portfolio Selection." Kelly, J.L. (1956), "A New
    Interpretation of Information Rate." Maillard, Roncalli & Teïletche
    (2010), "The Properties of Equally Weighted Risk Contribution
    Portfolios." Spinu, F. (2013), "An Algorithm for Computing Risk Parity
    Weights." See docs/REFERENCES.md.
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
    if np.any(np.diag(cov_matrix) <= 0.0):
        raise ValueError("cov_matrix must have strictly positive variances on its diagonal")


def risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the equal-risk-contribution (risk parity) portfolio.

    Solves Spinu's (2013) strictly convex reformulation
    `min_{w > 0} 0.5 * w'Sigma*w - (1/k) * sum(ln w)` by damped Newton
    iteration from the inverse-volatility portfolio, then rescales `w` to
    sum to 1. The result is the unique long-only portfolio whose normalised
    risk contributions `w_i * (Sigma*w)_i / (w'Sigma*w)` all equal `1/k`
    (Maillard, Roncalli & Teiletche 2010).

    Convergence target is a maximum risk-contribution deviation from `1/k`
    of 1e-10. On covariances so ill-conditioned that floating-point
    round-off in `Sigma*w` prevents the iteration from resolving the
    contributions that finely (a near-singular matrix, or eigenvalues
    spanning 1e8), the iteration stops once it can no longer improve and
    the best iterate is returned provided its deviation is within 1e-6;
    otherwise it raises.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k), with
            strictly positive variances on the diagonal.

    Returns:
        Portfolio weights of shape (k,), strictly positive and summing to
        1, with equal risk contributions.

    Raises:
        ValueError: If `cov_matrix` is not square or has a non-positive
            variance on its diagonal.
        RuntimeError: If the Newton iteration neither reaches the 1e-10
            target within `_RISK_PARITY_MAX_ITER` iterations nor stalls
            within the 1e-6 fallback tolerance. No silent fallback is
            returned in this case.
    """
    _validate_risk_parity_inputs(cov_matrix)
    return _risk_parity_weights(cov_matrix)


_RISK_PARITY_MAX_ITER = 200
_RISK_PARITY_TOL = 1e-10
_RISK_PARITY_STALL_TOL = 1e-6
_RISK_PARITY_ROUNDOFF_ITERS = 5


def _risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    # Rescaling Sigma by a positive constant rescales the unconstrained
    # minimiser by its inverse square root and leaves the normalised weights
    # unchanged, so the problem is solved on a unit-mean-variance copy for
    # conditioning; the covariance's own scale never enters the tolerance.
    sigma = cov_matrix / float(np.mean(np.diag(cov_matrix)))
    budget = np.full(k, 1.0 / k)

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(0.5 * (w @ sigma @ w) - budget @ np.log(w))

    def gradient(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = sigma @ w - budget / w
        return result

    def contribution_deviation(w: npt.NDArray[np.float64]) -> float:
        sigma_w = sigma @ w
        return float(np.max(np.abs(w * sigma_w / (w @ sigma_w) - budget)))

    # Inverse-volatility start: already the exact solution for a diagonal
    # Sigma and on the right scale for any other one.
    w = 1.0 / np.sqrt(np.diag(sigma))
    w /= w.sum()
    best_w, best_dev = w, contribution_deviation(w)
    # Once the objective can no longer resolve the predicted decrease, the
    # iterate is within round-off of the optimum; Newton's quadratic
    # convergence means a handful of further iterations is all that can
    # still help, after which the iteration is only wandering in round-off.
    roundoff_iters = 0
    for _ in range(_RISK_PARITY_MAX_ITER):
        if best_dev < _RISK_PARITY_TOL:
            return np.asarray(best_w / best_w.sum(), dtype=np.float64)
        if roundoff_iters > _RISK_PARITY_ROUNDOFF_ITERS:
            break
        grad = gradient(w)
        hessian = sigma + np.diag(budget / w**2)
        direction = np.linalg.solve(hessian, -grad)
        f_current = objective(w)
        slope = float(grad @ direction)
        grad_norm = float(np.linalg.norm(grad))
        # Backtracking line search: stay strictly inside w > 0 and satisfy
        # the Armijo sufficient-decrease condition. Close to the optimum the
        # predicted decrease drops below what `objective` can resolve in
        # floating point, so the step is then accepted on gradient-norm
        # descent instead (the Newton direction of a strictly convex
        # function is also the Newton direction of its gradient system).
        step = 1.0
        accepted = False
        while step >= 1e-12:
            candidate = w + step * direction
            if np.all(candidate > 0.0):
                predicted_decrease = -1e-4 * step * slope
                if objective(candidate) <= f_current - predicted_decrease:
                    accepted = True
                    break
                if predicted_decrease < 1e-12 * (1.0 + abs(f_current)):
                    roundoff_iters += 1
                    if float(np.linalg.norm(gradient(candidate))) < grad_norm:
                        accepted = True
                    break
            step *= 0.5
        if not accepted:
            break
        w = candidate
        dev = contribution_deviation(w)
        if dev < best_dev:
            best_w, best_dev = w, dev

    if best_dev < _RISK_PARITY_STALL_TOL:
        return np.asarray(best_w / best_w.sum(), dtype=np.float64)
    raise RuntimeError(
        f"risk_parity_weights: Newton iteration did not reach the risk-contribution "
        f"tolerance {_RISK_PARITY_TOL} within {_RISK_PARITY_MAX_ITER} iterations "
        f"(best deviation {best_dev:.3e} exceeds the {_RISK_PARITY_STALL_TOL} "
        f"fallback tolerance); the covariance matrix may be numerically invalid"
    )


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
