"""Tier-2 portfolio construction: covariance, mean-variance, risk parity, Kelly, Black-Litterman.

Each quantity is registered as a `ConformanceTarget` pairing a quantcore adapter (a thin call into
`quantcore.portfolio`) with an independent oracle. Importing this module registers all of them.

Covariances: unbiased sample, EWMA (RiskMetrics, seeded at the sample covariance) and
Ledoit-Wolf (population `1/T` normalisation, identity-scaled target). The four mean-variance
optimisers are solved in `cvxpy` and compared on the attained objective, not the weights;
`risk_parity_weights` (unique, Spinu 2013) is compared on its weights.

Full citations for Markowitz (1952), Kelly (1956), Maillard, Roncalli & Teiletche (2010), Spinu
(2013), Black & Litterman (1992), He & Litterman (1999), Ledoit & Wolf (2004) and RiskMetrics
(1996) are in `docs/REFERENCES.md` under "Portfolio optimization" and "Covariance estimation".
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from cvxpy import CLARABEL, Maximize, Minimize, Problem, Variable
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.quad_form import quad_form
from cvxpy.settings import OPTIMAL, OPTIMAL_INACCURATE
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import quantcore.portfolio as qc_portfolio
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    DecisiveFloatArrayComparator,
    ScalarComparator,
)

type FloatArray = npt.NDArray[np.float64]

_CVXPY_SOLVER = CLARABEL


def _require_cvxpy_optimal(problem: Problem, func_name: str) -> None:
    """Mirror quantcore's own `_require_slsqp_success`: no silent fallback on a bad solve."""
    if problem.status not in (OPTIMAL, OPTIMAL_INACCURATE):
        raise RuntimeError(f"{func_name}: cvxpy solver did not reach optimality: {problem.status}")


def _variable_value(variable: Variable, func_name: str) -> FloatArray:
    value = variable.value
    if value is None:
        raise RuntimeError(f"{func_name}: cvxpy returned no solution value")
    return np.asarray(value, dtype=np.float64)


# --- sample_covariance ---------------------------------------------------------------------------


def _sample_covariance_quantcore(returns: FloatArray) -> FloatArray:
    return np.asarray(qc_portfolio.sample_covariance(returns), dtype=np.float64)


def _sample_covariance_reference(returns: FloatArray) -> FloatArray:
    """`numpy.cov(returns, rowvar=False, ddof=1)`: an independent library's own unbiased covariance.

    This anchors the oracle to an independent library's code, rather than re-implementing
    quantcore's own documented two-pass formula by hand.
    """
    return np.asarray(np.cov(returns, rowvar=False, ddof=1), dtype=np.float64)


_SAMPLE_COVARIANCE = ConformanceTarget(
    name="sample_covariance",
    tier=2,
    oracle="numpy.cov(returns, rowvar=False, ddof=1)",
    citation="Standard unbiased sample covariance (ddof=1).",
    compare=ArrayComparator(),
    quantcore=_sample_covariance_quantcore,
    reference=_sample_covariance_reference,
)


# --- ewma_covariance -------------------------------------------------------------------------


def _ewma_covariance_quantcore(returns: FloatArray, lambda_: float) -> FloatArray:
    return np.asarray(qc_portfolio.ewma_covariance(returns, lambda_), dtype=np.float64)


def _ewma_covariance_reference(returns: FloatArray, lambda_: float) -> FloatArray:
    """RiskMetrics zero-mean EWMA covariance built from a pandas exponentially weighted mean.

    The recursion `S_t = lambda*S_{t-1} + (1-lambda)*r_{t-1} r_{t-1}'` with seed `S_0` equal to the
    sample covariance is, for `T` observations and `p_t = r_t r_t'`,
    `S_{T-1} = m_{T-2} + lambda^(T-1) * (S_0 - p_0)`, where `m` is
    `pandas.DataFrame(p).ewm(alpha=1-lambda_, adjust=False).mean()` (seeded at `p_0`), i.e. the
    exponentially weighted mean of the outer products lagged by one observation, with the seed
    swapped from `p_0` to `S_0`. Returns are not demeaned. A single observation keeps
    `outer(r_0, r_0)`.
    """
    num_obs, num_assets = returns.shape
    if num_obs < 2:
        outer = np.outer(returns[0], returns[0])
        return np.asarray(outer, dtype=np.float64)
    products = pd.DataFrame((returns[:, :, None] * returns[:, None, :]).reshape(num_obs, -1))
    lagged_mean = products.ewm(alpha=1.0 - lambda_, adjust=False).mean().iloc[num_obs - 2]
    mean_matrix = lagged_mean.to_numpy(dtype=np.float64).reshape(num_assets, num_assets)
    seed = np.atleast_2d(np.cov(returns.T, ddof=1))
    first = np.outer(returns[0], returns[0])
    result = mean_matrix + lambda_ ** (num_obs - 1) * (seed - first)
    return np.asarray(result, dtype=np.float64)


_EWMA_COVARIANCE = ConformanceTarget(
    name="ewma_covariance",
    tier=2,
    oracle=(
        "pandas.DataFrame.ewm(alpha=1-lambda_, adjust=False).mean() of the outer products "
        "r_t r_t', lagged one observation, with the sample-covariance seed (no demeaning)"
    ),
    citation=(
        "J.P. Morgan/Reuters (1996), RiskMetrics -- Technical Document (4th ed.): zero-mean "
        "exponentially weighted covariance of raw return outer products."
    ),
    compare=ArrayComparator(),
    quantcore=_ewma_covariance_quantcore,
    reference=_ewma_covariance_reference,
)


# --- ledoit_wolf_shrinkage --------------------------------------------------------------------


def _lw_sample_and_target(returns: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Theorem 1's own population (1/T) sample covariance and its scaled-identity target."""
    num_obs, num_assets = returns.shape
    demeaned = returns - returns.mean(axis=0)
    sample = (demeaned.T @ demeaned) / num_obs
    mu_hat = float(np.trace(sample)) / num_assets
    target = mu_hat * np.eye(num_assets)
    return sample, target


def _recovered_shrinkage(cov: FloatArray, sample: FloatArray, target: FloatArray) -> float:
    """Recover the shrinkage intensity actually used to produce `cov`, from `cov` alone.

    `cov` = (1-alpha)*sample + alpha*target = sample + alpha*(target - sample) for whichever alpha
    the estimator used internally (quantcore never returns alpha itself; sklearn's `shrinkage_`
    does). This is the projection of `(cov - sample)` onto `(target - sample)` in the Frobenius
    inner product, which recovers `alpha` exactly whenever the estimator truly followed that affine
    formula (quantcore and sklearn both do, by construction and by `Ledoit & Wolf`'s Theorem 1) --
    not an approximation.
    """
    direction = target - sample
    denominator = float(np.sum(direction * direction))
    if denominator == 0.0:
        return 0.0
    return float(np.sum((cov - sample) * direction)) / denominator


def _pack_cov_and_shrinkage(cov: FloatArray, alpha: float) -> FloatArray:
    """Concatenate a `(k, k)` matrix and a scalar into one flat array for a single-quantity check.

    `harness.check()`'s per-quantity comparator splitting only special-cases `TupleComparator` and
    `DecisiveFloatArrayComparator` (see `conformance.compare`); rather than adding a third
    bespoke multi-quantity `Comparator` shape to that shared machinery for this one function, the
    matrix and the shrinkage intensity are packed into a single flat `FloatArray` and compared
    together as one ordinary `ArrayComparator` quantity -- comparing both the shrinkage intensity
    and the matrix holds exactly as well this way, since a mismatch anywhere in the packed array
    still fails the comparison.
    """
    return np.asarray(np.concatenate([cov.ravel(), [alpha]]), dtype=np.float64)


def _ledoit_wolf_shrinkage_quantcore(returns: FloatArray) -> FloatArray:
    cov = np.asarray(qc_portfolio.ledoit_wolf_shrinkage(returns), dtype=np.float64)
    sample, target = _lw_sample_and_target(returns)
    alpha = _recovered_shrinkage(cov, sample, target)
    return _pack_cov_and_shrinkage(cov, alpha)


def _ledoit_wolf_shrinkage_reference(returns: FloatArray) -> FloatArray:
    """`sklearn.covariance.LedoitWolf(assume_centered=False)`.

    sklearn centres by the sample mean (matching quantcore's `demeaned = returns -
    returns.mean(axis=0)`) and normalises its internal empirical covariance by `1/T`
    (`emp_cov_trace = np.sum(X2, axis=0) / n_samples` in `sklearn.covariance.ledoit_wolf_shrinkage`,
    sklearn 1.9.1), the same population convention Ledoit & Wolf's Theorem 1 and quantcore's own
    docstring both specify -- so `assume_centered=False` (the default) follows from the definition
    itself rather than from a choice made to match quantcore. Measured against quantcore: max abs
    covariance-matrix difference ~1.2e-15 and exact shrinkage-intensity agreement (~1e-13) on a
    200x5 standard-normal panel.
    """
    estimator = LedoitWolf(assume_centered=False).fit(returns)
    cov = np.asarray(estimator.covariance_, dtype=np.float64)
    return _pack_cov_and_shrinkage(cov, float(estimator.shrinkage_))


_LEDOIT_WOLF_SHRINKAGE = ConformanceTarget(
    name="ledoit_wolf_shrinkage",
    tier=2,
    oracle="sklearn.covariance.LedoitWolf(assume_centered=False)",
    citation="Ledoit & Wolf (2004). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_ledoit_wolf_shrinkage_quantcore,
    reference=_ledoit_wolf_shrinkage_reference,
)


# --- mean-variance family: min_variance / mean_variance / unconstrained / l1-turnover -------------
# Each is registered as (objective, weights) under `DecisiveFloatArrayComparator`, classified by
# the sign-test rule for iterative estimators (every case is `well_specified`: a convex program
# has no misspecified family).

OptimizerResult = tuple[float, FloatArray]
"""`(objective, weights)`, as every optimiser `ConformanceTarget` in this module returns it."""


def _min_variance_objective(weights: FloatArray, cov_matrix: FloatArray) -> float:
    return float(weights @ cov_matrix @ weights)


def _min_variance_weights_quantcore(cov_matrix: FloatArray, allow_short: bool) -> OptimizerResult:
    weights = np.asarray(
        qc_portfolio.min_variance_weights(cov_matrix, allow_short), dtype=np.float64
    )
    return _min_variance_objective(weights, cov_matrix), weights


def _min_variance_weights_reference(cov_matrix: FloatArray, allow_short: bool) -> OptimizerResult:
    """Same problem quantcore solves with SLSQP, written as a DCP in `cvxpy`.

    `min_variance_weights`: `min w^T Sigma w  s.t.  sum(w) = 1, [w >= 0]` -- read directly from
    `quantcore.portfolio.optimization._min_variance_weights`'s `LinearConstraint`/`bounds`.
    """
    k = cov_matrix.shape[0]
    weights = Variable(k)
    constraints = [cp_sum(weights) == 1]
    if not allow_short:
        constraints.append(weights >= 0)
    problem = Problem(Minimize(quad_form(weights, psd_wrap(cov_matrix))), constraints)
    problem.solve(solver=_CVXPY_SOLVER)
    _require_cvxpy_optimal(problem, "min_variance_weights")
    w = _variable_value(weights, "min_variance_weights")
    return _min_variance_objective(w, cov_matrix), w


_MIN_VARIANCE_WEIGHTS = ConformanceTarget(
    name="min_variance_weights",
    tier=2,
    oracle="cvxpy (same long-only/sum-to-one QP quantcore solves with SLSQP)",
    citation="Markowitz (1952). Full citation in docs/REFERENCES.md.",
    compare=DecisiveFloatArrayComparator(),
    quantcore=_min_variance_weights_quantcore,
    reference=_min_variance_weights_reference,
)


def _mean_variance_objective(
    weights: FloatArray, expected_returns: FloatArray, cov_matrix: FloatArray, risk_aversion: float
) -> float:
    risk_term = weights @ cov_matrix @ weights
    return float(weights @ expected_returns - 0.5 * risk_aversion * risk_term)


def _mean_variance_weights_quantcore(
    expected_returns: FloatArray, cov_matrix: FloatArray, risk_aversion: float, allow_short: bool
) -> OptimizerResult:
    weights = np.asarray(
        qc_portfolio.mean_variance_weights(
            expected_returns, cov_matrix, risk_aversion, allow_short
        ),
        dtype=np.float64,
    )
    return _mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion), weights


def _mean_variance_weights_reference(
    expected_returns: FloatArray, cov_matrix: FloatArray, risk_aversion: float, allow_short: bool
) -> OptimizerResult:
    """`max w^T mu - 0.5*ra*w^T Sigma w` s.t. `sum(w)=1, [w>=0]`; same constraints as quantcore."""
    k = cov_matrix.shape[0]
    weights = Variable(k)
    constraints = [cp_sum(weights) == 1]
    if not allow_short:
        constraints.append(weights >= 0)
    objective = weights @ expected_returns - 0.5 * risk_aversion * quad_form(
        weights, psd_wrap(cov_matrix)
    )
    problem = Problem(Maximize(objective), constraints)
    problem.solve(solver=_CVXPY_SOLVER)
    _require_cvxpy_optimal(problem, "mean_variance_weights")
    w = _variable_value(weights, "mean_variance_weights")
    return _mean_variance_objective(w, expected_returns, cov_matrix, risk_aversion), w


_MEAN_VARIANCE_WEIGHTS = ConformanceTarget(
    name="mean_variance_weights",
    tier=2,
    oracle="cvxpy (same long-only/sum-to-one QP quantcore solves with SLSQP)",
    citation="Markowitz (1952). Full citation in docs/REFERENCES.md.",
    compare=DecisiveFloatArrayComparator(),
    quantcore=_mean_variance_weights_quantcore,
    reference=_mean_variance_weights_reference,
)


def _unconstrained_mean_variance_weights_quantcore(
    expected_returns: FloatArray, cov_matrix: FloatArray, risk_aversion: float
) -> OptimizerResult:
    weights = np.asarray(
        qc_portfolio.unconstrained_mean_variance_weights(
            expected_returns, cov_matrix, risk_aversion
        ),
        dtype=np.float64,
    )
    return _mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion), weights


def _unconstrained_mean_variance_weights_reference(
    expected_returns: FloatArray, cov_matrix: FloatArray, risk_aversion: float
) -> OptimizerResult:
    """`max w^T mu - 0.5*ra*w^T Sigma w`, NO constraints at all; closed form Sigma^-1 mu / ra.

    Solved directly via `numpy.linalg.solve` (an independent direct linear solve, not `cvxpy`) for
    the same reason quantcore's own docstring gives for preferring `solve` over `inv`: no
    constraints means there is nothing for a QP solver to add over the closed form itself, and a
    direct solve avoids squaring the condition number's effect on rounding error that an explicit
    inverse would.
    """
    weights = np.asarray(
        np.linalg.solve(cov_matrix, expected_returns) / risk_aversion, dtype=np.float64
    )
    return _mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion), weights


_UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS = ConformanceTarget(
    name="unconstrained_mean_variance_weights",
    tier=2,
    oracle="independent closed-form linear solve (no constraints)",
    citation="Markowitz (1952). Full citation in docs/REFERENCES.md.",
    compare=DecisiveFloatArrayComparator(),
    quantcore=_unconstrained_mean_variance_weights_quantcore,
    reference=_unconstrained_mean_variance_weights_reference,
)


def _l1_turnover_objective(
    weights: FloatArray,
    expected_returns: FloatArray,
    cov_matrix: FloatArray,
    previous_weights: FloatArray,
    cost_bps: float,
    risk_aversion: float,
) -> float:
    turnover = float(np.sum(np.abs(weights - previous_weights)))
    return (
        _mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion)
        - (cost_bps / 10_000.0) * turnover
    )


def _l1_turnover_penalized_weights_quantcore(
    expected_returns: FloatArray,
    cov_matrix: FloatArray,
    previous_weights: FloatArray,
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool,
) -> OptimizerResult:
    weights = np.asarray(
        qc_portfolio.l1_turnover_penalized_weights(
            expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion, allow_short
        ),
        dtype=np.float64,
    )
    objective = _l1_turnover_objective(
        weights, expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion
    )
    return objective, weights


def _l1_turnover_penalized_weights_reference(
    expected_returns: FloatArray,
    cov_matrix: FloatArray,
    previous_weights: FloatArray,
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool,
) -> OptimizerResult:
    """Same objective and `sum(w) = 1, [w >= 0]` constraints, with `cvxpy.norm1` for the L1 term.

    quantcore splits the L1 penalty into non-negative buy/sell slacks so SLSQP's finite-difference
    gradient sees a smooth problem (its own module docstring); `cvxpy.norm1` needs no such
    reformulation since `cvxpy` handles the nonsmooth term directly as a DCP atom, so this reference
    solves the mathematically identical convex program without replicating that implementation
    detail.
    """
    k = cov_matrix.shape[0]
    weights = Variable(k)
    constraints = [cp_sum(weights) == 1]
    if not allow_short:
        constraints.append(weights >= 0)
    objective = (
        weights @ expected_returns
        - 0.5 * risk_aversion * quad_form(weights, psd_wrap(cov_matrix))
        - (cost_bps / 10_000.0) * norm1(weights - previous_weights)
    )
    problem = Problem(Maximize(objective), constraints)
    problem.solve(solver=_CVXPY_SOLVER)
    _require_cvxpy_optimal(problem, "l1_turnover_penalized_weights")
    w = _variable_value(weights, "l1_turnover_penalized_weights")
    result_objective = _l1_turnover_objective(
        w, expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion
    )
    return result_objective, w


_L1_TURNOVER_PENALIZED_WEIGHTS = ConformanceTarget(
    name="l1_turnover_penalized_weights",
    tier=2,
    oracle="cvxpy (same QP/L1 program quantcore solves via a smooth buy/sell-slack SLSQP)",
    citation="Markowitz (1952). Full citation in docs/REFERENCES.md.",
    compare=DecisiveFloatArrayComparator(),
    quantcore=_l1_turnover_penalized_weights_quantcore,
    reference=_l1_turnover_penalized_weights_reference,
)


# --- risk_parity_weights: invariant test against an independent convex solver ---------------------


def _spinu_risk_parity_weights(cov_matrix: FloatArray) -> FloatArray:
    """Solve Spinu (2013)'s strictly convex reformulation: min 0.5*w^T Sigma w - sum(log w), w > 0.

    First-order condition `Sigma w - 1/w = 0` is exactly `w_i*(Sigma w)_i = 1` for every `i`, i.e.
    equal risk contributions; the Hessian `Sigma + diag(1/w_i^2)` is positive definite whenever
    `Sigma` is PSD, so the minimiser (and hence, after normalising to sum to 1, the risk-parity
    portfolio) is unique. Solved in `log w` (`x = log w`) so the `w > 0` constraint is automatic
    and `scipy.optimize.minimize` sees an unconstrained smooth problem, exactly mirroring
    quantcore's own preference for a numerically well-posed reformulation over a constrained one.
    """
    k = cov_matrix.shape[0]

    def objective(log_w: FloatArray) -> float:
        w = np.exp(log_w)
        return float(0.5 * w @ cov_matrix @ w - np.sum(log_w))

    def gradient(log_w: FloatArray) -> FloatArray:
        w = np.exp(log_w)
        return np.asarray((cov_matrix @ w - 1.0 / w) * w, dtype=np.float64)

    def solve_from(x0: FloatArray) -> tuple[FloatArray, float]:
        """One L-BFGS-B solve plus a damped-Newton polish; returns `(weights, max_rc_deviation)`.

        The Newton polish sits on top of L-BFGS-B's local-only (gradient-only) optimum: analytic
        second-order information converges quadratically once already close, closing the last few
        orders of magnitude L-BFGS-B's line search alone leaves on the most ill-conditioned draws.
        The exact Hessian in log_w coordinates is `diag(w) @ Sigma @ diag(w) + diag((Sigma @ w) *
        w)`, positive definite whenever Sigma is PSD and w > 0 (Boyd & Vandenberghe 2004, ch. 9,
        damped Newton with backtracking).
        """
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(-50.0, 50.0)] * k,
            options={"maxiter": 20_000, "maxfun": 40_000, "ftol": 1e-18, "gtol": 1e-14},
        )
        x = result.x
        for _ in range(30):
            w = np.exp(x)
            s_w = cov_matrix @ w
            grad = s_w * w - 1.0
            hessian = (w[:, None] * cov_matrix * w[None, :]) + np.diag(s_w * w)
            try:
                newton_step = np.linalg.solve(hessian, -grad)
            except np.linalg.LinAlgError:
                break
            step_size = 1.0
            f0 = 0.5 * w @ cov_matrix @ w - np.sum(x)
            directional_derivative = grad @ newton_step
            for _ in range(60):
                x_trial = np.clip(x + step_size * newton_step, -50.0, 50.0)
                w_trial = np.exp(x_trial)
                f_trial = 0.5 * w_trial @ cov_matrix @ w_trial - np.sum(x_trial)
                threshold = f0 + 1e-4 * step_size * directional_derivative
                if np.isfinite(f_trial) and f_trial <= threshold:
                    break
                step_size *= 0.5
            x = x_trial
        w = np.exp(x)
        w = w / w.sum()
        risk_contributions = w * (cov_matrix @ w) / (w @ cov_matrix @ w)
        deviation = float(np.max(np.abs(risk_contributions - 1.0 / k)))
        return np.asarray(w, dtype=np.float64), deviation

    # Three starting points (inverse-volatility, equal weights, inverse-diagonal) rather than one:
    # on ill-conditioned draws (cond(Sigma) up to ~1e6) a single local search can stall and scipy's
    # `result.success` is unreliable. The first-order condition (every risk contribution equals 1/k)
    # gates the raise, not solver bookkeeping, and the best of the three is kept.
    starting_points = (
        -0.5 * np.log(np.diag(cov_matrix)),
        np.zeros(k),
        -np.log(np.diag(cov_matrix)),
    )
    best_w, best_deviation = min(
        (solve_from(x0) for x0 in starting_points), key=lambda result: result[1]
    )
    # A fixed-seed local search around the best point for the rare cases (~0.3% of a 7500-case
    # sweep) where the three starts still stall before reaching 1e-6. The fixed seed keeps the
    # reference deterministic given `cov_matrix`.
    if not np.isfinite(best_deviation) or best_deviation > 1e-6:
        rng = np.random.default_rng(0)
        best_x = np.log(best_w) if np.all(best_w > 0.0) else starting_points[0]
        for _ in range(15):
            perturbed_w, perturbed_deviation = solve_from(best_x + rng.normal(0.0, 2.0, k))
            if perturbed_deviation < best_deviation:
                best_w, best_deviation = perturbed_w, perturbed_deviation
                best_x = np.log(best_w)
            if best_deviation <= 1e-6:
                break
    if not np.isfinite(best_deviation) or best_deviation > 1e-6:
        raise RuntimeError(
            "risk_parity_weights reference solver did not converge: best max risk-contribution "
            f"deviation from 1/k after multi-start retries is {best_deviation!r}"
        )
    return best_w


def _risk_parity_weights_quantcore(cov_matrix: FloatArray) -> FloatArray:
    return np.asarray(qc_portfolio.risk_parity_weights(cov_matrix), dtype=np.float64)


def _risk_parity_weights_reference(cov_matrix: FloatArray) -> FloatArray:
    return _spinu_risk_parity_weights(cov_matrix)


_RISK_PARITY_WEIGHTS = ConformanceTarget(
    name="risk_parity_weights",
    tier=2,
    oracle="independent Spinu (2013) convex reformulation, solved by L-BFGS-B",
    citation="Maillard, Roncalli & Teiletche (2010); Spinu (2013). Full citations in "
    "docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_risk_parity_weights_quantcore,
    reference=_risk_parity_weights_reference,
)


# --- kelly_fraction ---------------------------------------------------------------------------


def _kelly_fraction_quantcore(expected_return: float, variance: float, clip: bool) -> float:
    return float(qc_portfolio.kelly_fraction(expected_return, variance, clip))


def _kelly_fraction_reference(expected_return: float, variance: float, clip: bool) -> float:
    """Hand formula `f* = E[r] / sigma^2` (Kelly 1956; Thorp 2006), with the same `[-1, 1]` clip."""
    fraction = expected_return / variance
    if not clip:
        return float(fraction)
    return float(np.clip(fraction, -1.0, 1.0))


_KELLY_FRACTION = ConformanceTarget(
    name="kelly_fraction",
    tier=2,
    oracle="hand formula (Kelly 1956 / Thorp 2006)",
    citation="Kelly (1956), full citation in docs/REFERENCES.md; Thorp (2006).",
    compare=ScalarComparator(),
    quantcore=_kelly_fraction_quantcore,
    reference=_kelly_fraction_reference,
)


# --- implied_equilibrium_returns, black_litterman ------------------------------------------------


def _implied_equilibrium_returns_quantcore(
    cov_matrix: FloatArray, market_weights: FloatArray, risk_aversion: float
) -> FloatArray:
    return np.asarray(
        qc_portfolio.implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion),
        dtype=np.float64,
    )


def _implied_equilibrium_returns_reference(
    cov_matrix: FloatArray, market_weights: FloatArray, risk_aversion: float
) -> FloatArray:
    """Hand formula `Pi = risk_aversion * Sigma @ w_mkt` (He & Litterman 1999)."""
    return np.asarray(risk_aversion * cov_matrix @ market_weights, dtype=np.float64)


_IMPLIED_EQUILIBRIUM_RETURNS = ConformanceTarget(
    name="implied_equilibrium_returns",
    tier=2,
    oracle="hand-derived from He & Litterman 1999 eq. 4",
    citation="He & Litterman (1999), eq. 4. Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_implied_equilibrium_returns_quantcore,
    reference=_implied_equilibrium_returns_reference,
)


def _pack_bl_posterior(posterior_returns: FloatArray, posterior_cov: FloatArray) -> FloatArray:
    """Concatenate `(returns, cov)` into one flat array for a single-quantity check.

    Same rationale as `_pack_cov_and_shrinkage`: rather than a bespoke two-array `Comparator`
    shape, both posterior quantities are packed into one flat `FloatArray` and compared together
    as one ordinary `ArrayComparator` quantity.
    """
    return np.asarray(
        np.concatenate([posterior_returns.ravel(), posterior_cov.ravel()]), dtype=np.float64
    )


def _black_litterman_quantcore(
    cov_matrix: FloatArray,
    market_weights: FloatArray,
    p: FloatArray,
    q: FloatArray,
    omega: FloatArray,
    risk_aversion: float,
    tau: float,
) -> FloatArray:
    posterior_returns, posterior_cov = qc_portfolio.black_litterman(
        cov_matrix, market_weights, p, q, omega, risk_aversion, tau
    )
    return _pack_bl_posterior(
        np.asarray(posterior_returns, dtype=np.float64), np.asarray(posterior_cov, dtype=np.float64)
    )


def _black_litterman_reference(
    cov_matrix: FloatArray,
    market_weights: FloatArray,
    p: FloatArray,
    q: FloatArray,
    omega: FloatArray,
    risk_aversion: float,
    tau: float,
) -> FloatArray:
    """Hand He & Litterman (1999) / Black & Litterman (1992) posterior, via `numpy.linalg.solve`.

    `Pi = risk_aversion*Sigma@w_mkt`;
    `mu_BL = Pi + (tau*Sigma)@P^T@solve(P@(tau*Sigma)@P^T + Omega, Q - P@Pi)`;
    `Sigma_BL = tau*Sigma - (tau*Sigma)@P^T@solve(P@(tau*Sigma)@P^T + Omega, P@(tau*Sigma))`
    -- the same algebraically-equivalent, inverse-free form quantcore's own module docstring
    derives, coded independently here rather than reusing quantcore's implementation.
    """
    pi = risk_aversion * cov_matrix @ market_weights
    tau_sigma = tau * cov_matrix
    p_tau_sigma = p @ tau_sigma
    m = p_tau_sigma @ p.T + omega
    posterior_returns = pi + tau_sigma @ p.T @ np.linalg.solve(m, q - p @ pi)
    posterior_cov = tau_sigma - tau_sigma @ p.T @ np.linalg.solve(m, p_tau_sigma)
    return _pack_bl_posterior(
        np.asarray(posterior_returns, dtype=np.float64), np.asarray(posterior_cov, dtype=np.float64)
    )


_BLACK_LITTERMAN = ConformanceTarget(
    name="black_litterman",
    tier=2,
    oracle="hand-derived from He & Litterman 1999 eq. 8",
    citation="Black & Litterman (1992); He & Litterman (1999), eq. 8. Full citations in "
    "docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_black_litterman_quantcore,
    reference=_black_litterman_reference,
)
