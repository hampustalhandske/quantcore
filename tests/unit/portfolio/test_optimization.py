"""Tests for minimum-variance, unconstrained mean-variance and risk-parity weights."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import nnls

import quantcore.portfolio.optimization as opt_module
from quantcore.portfolio.optimization import (
    min_variance_weights,
    risk_parity_weights,
    unconstrained_mean_variance_weights,
)


class TestMinVarianceWeights:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            min_variance_weights(np.array([1.0, 2.0, 3.0]))

    def test_invalid_inputs_non_square(self) -> None:
        with pytest.raises(ValueError):
            min_variance_weights(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_weights_sum_to_one_and_non_negative(self) -> None:
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        weights = min_variance_weights(cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-8)

    def test_matches_analytical_solution_for_uncorrelated_assets(self) -> None:
        # For diagonal Sigma, the unconstrained min-variance weights are
        # w_i proportional to 1 / sigma_i^2.
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        expected = np.array([1.0 / 0.04, 1.0 / 0.01])
        expected = expected / expected.sum()
        weights = min_variance_weights(cov)
        assert weights[0] == pytest.approx(expected[0], abs=1e-12)
        assert weights[1] == pytest.approx(expected[1], abs=1e-12)


class TestRiskParityWeights:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.array([1.0, 2.0, 3.0]))

    def test_non_positive_variance_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_risk_contributions_are_equal(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T + 4 * np.eye(4)  # random PD matrix

        weights = risk_parity_weights(cov)
        contributions = weights * (cov @ weights) / (weights @ cov @ weights)
        np.testing.assert_allclose(contributions, 0.25, atol=1e-9)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(weights > 0.0)

    def test_one_dominant_variance_asset(self) -> None:
        # Asset 0's variance is ~2700x asset 1's; an equal-weight start on
        # the un-normalised least-squares objective used to terminate at
        # [0.5, 0.5] with asset 0 carrying 99.99% of the risk.
        cov = np.array([[4658.98781, -1.41048619], [-1.41048619, 1.72544360]])
        weights = risk_parity_weights(cov)
        contributions = weights * (cov @ weights) / (weights @ cov @ weights)
        np.testing.assert_allclose(contributions, 0.5, atol=1e-9)
        assert weights[0] < 0.05

    def test_diagonal_covariance_gives_inverse_volatility_weights(self) -> None:
        vols = np.array([0.1, 0.2, 0.4])
        weights = risk_parity_weights(np.diag(vols**2))
        expected = (1.0 / vols) / np.sum(1.0 / vols)
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_scale_invariant(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.normal(size=(5, 5))
        cov = a @ a.T + np.eye(5)
        np.testing.assert_allclose(
            risk_parity_weights(cov), risk_parity_weights(1e6 * cov), atol=1e-10
        )

    @pytest.mark.parametrize(
        "family",
        ["well_conditioned", "near_singular", "equal_vol", "one_dominant", "high_condition"],
    )
    def test_risk_contributions_equal_across_covariance_families(self, family: str) -> None:
        rng = np.random.default_rng(20240924)
        for _ in range(40):
            k = int(rng.integers(2, 11))
            if family == "well_conditioned":
                a = rng.normal(size=(k, k))
                cov = a @ a.T + 0.1 * np.eye(k)
            elif family == "near_singular":
                a = rng.normal(size=(k, 1))
                cov = a @ a.T + 1e-8 * np.eye(k)
            elif family == "equal_vol":
                corr = rng.uniform(-0.3, 0.9)
                cov = np.full((k, k), corr) + (1.0 - corr) * np.eye(k)
            elif family == "one_dominant":
                a = rng.normal(size=(k, k))
                cov = a @ a.T + np.eye(k)
                cov[0, 0] *= 1e4
            else:
                q, _ = np.linalg.qr(rng.normal(size=(k, k)))
                cov = q @ np.diag(np.logspace(-4, 4, k)) @ q.T
            weights = risk_parity_weights(cov)
            contributions = weights * (cov @ weights) / (weights @ cov @ weights)
            np.testing.assert_allclose(contributions, 1.0 / k, atol=1e-6)


class TestUnconstrainedMeanVarianceWeights:
    def test_invalid_inputs_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07, 0.03]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=2.0,
            )

    def test_invalid_inputs_non_square_cov_matrix(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                risk_aversion=2.0,
            )

    def test_invalid_inputs_zero_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=0.0,
            )

    def test_invalid_inputs_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=-1.0,
            )

    def test_matches_analytical_solution_for_uncorrelated_assets(self) -> None:
        # For diagonal Sigma, the closed-form solution is
        # w_i = mu_i / (risk_aversion * sigma_i^2).
        expected_returns = np.array([0.05, 0.07])
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        risk_aversion = 3.0

        expected = np.array([0.05 / (3.0 * 0.04), 0.07 / (3.0 * 0.01)])
        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)

        assert weights == pytest.approx(expected, abs=1e-10)

    def test_satisfies_first_order_condition(self) -> None:
        expected_returns = np.array([0.05, 0.07, 0.03])
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.02]])
        risk_aversion = 2.0

        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)

        assert cov @ weights * risk_aversion == pytest.approx(expected_returns, abs=1e-8)

    def test_matches_explicit_inverse_on_ill_conditioned_matrix(self) -> None:
        # Near-singular but still invertible: a strong cross-correlation
        # pushes the condition number high without breaking invertibility.
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        expected_returns = np.array([0.05, 0.06])
        risk_aversion = 2.0

        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)
        reference = np.linalg.inv(cov) @ expected_returns / risk_aversion

        assert weights == pytest.approx(reference, abs=1e-6)


class TestSolverFailureIsReported:
    """No optimiser returns a point it could not certify as optimal.

    `min_variance_weights` raises when its quadratic program has no finite
    optimum or the active-set iteration runs out of its iteration budget;
    `risk_parity_weights` has the same contract for its Newton iteration.
    """

    def test_min_variance_weights_raises_at_iteration_cap(self, monkeypatch) -> None:
        monkeypatch.setattr(opt_module, "_QP_MAX_ITER_PER_CONSTRAINT", 0)
        with pytest.raises(RuntimeError, match="did not converge"):
            min_variance_weights(np.eye(2))

    def test_risk_parity_weights_raises_on_non_convergence(self, monkeypatch) -> None:
        # risk_parity_weights uses its own Newton iteration rather than
        # SLSQP; forcing an iteration budget of zero with no fallback
        # tolerance must raise rather than return the starting point.
        import quantcore.portfolio.optimization as opt_module

        monkeypatch.setattr(opt_module, "_RISK_PARITY_MAX_ITER", 0)
        monkeypatch.setattr(opt_module, "_RISK_PARITY_STALL_TOL", 0.0)
        with pytest.raises(RuntimeError):
            risk_parity_weights(np.array([[1.0, 0.5], [0.5, 2.0]]))


FloatArray = npt.NDArray[np.float64]


def _random_problem(seed: int, k: int = 6) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Expected returns, a positive definite covariance and a long-only portfolio."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    return rng.normal(0.02, 0.05, k), a @ a.T / k + 0.05 * np.eye(k), rng.dirichlet(np.ones(k))


def _near_singular_cov(seed: int, k: int) -> FloatArray:
    """Covariance with eigenvalues in [0.5, 2] except one of 1e-6."""
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(k, k)))
    eigenvalues = rng.uniform(0.5, 2.0, k)
    eigenvalues[0] = 1e-6
    return (orthogonal * eigenvalues) @ orthogonal.T


def _one_factor_cov(seed: int, k: int) -> FloatArray:
    """Covariance of assets that load on one common factor with betas in [0.5, 1.5].

    The unconstrained minimum-variance portfolio shorts the high-beta assets
    to hedge the factor, so a long-only constraint binds.
    """
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.5, 1.5, k)
    return 0.04 * np.outer(betas, betas) + np.diag(rng.uniform(0.005, 0.02, k))


def _gradient_and_scale(
    weights: FloatArray, hessian: FloatArray, linear: FloatArray
) -> tuple[FloatArray, float]:
    """Gradient of `0.5*w'Hw + linear'w` and the size of the terms it is summed from.

    Round-off in the gradient is proportional to that size, not to the
    gradient itself, which can be orders of magnitude smaller through
    cancellation; optimality conditions are therefore judged relative to it.
    """
    scale = max(float(np.max(np.abs(linear))), float(np.max(np.abs(hessian) @ np.abs(weights))))
    return hessian @ weights + linear, scale


def _budget_kkt_violation(
    weights: FloatArray, hessian: FloatArray, linear: FloatArray, long_only: bool
) -> float:
    """Violation of the optimality conditions of `min 0.5*w'Hw + linear'w, sum(w) = 1, [w >= 0]`.

    At an optimum every asset with a positive weight (every asset, when
    shorting is allowed) has the same gradient entry, the budget multiplier,
    and an asset at zero has one no lower than it. Returns the amount by
    which that fails, relative to the size of the terms the gradient is
    summed from.
    """
    gradient, scale = _gradient_and_scale(weights, hessian, linear)
    held = weights > 0.0 if long_only else np.ones(weights.size, dtype=bool)
    multiplier = float(np.mean(gradient[held]))
    spread = float(np.max(np.abs(gradient[held] - multiplier)))
    undercut = float(np.max(multiplier - gradient[~held], initial=0.0))
    return max(spread, undercut) / scale


class TestMinVarianceWeightsExactness:
    def test_matches_closed_form_when_shorting_is_allowed(self) -> None:
        # Equality-constrained problem: w = Sigma^-1 1 / (1' Sigma^-1 1).
        _, cov, _ = _random_problem(11)
        inverse_ones = np.linalg.solve(cov, np.ones(cov.shape[0]))
        np.testing.assert_allclose(
            min_variance_weights(cov, allow_short=True),
            inverse_ones / inverse_ones.sum(),
            atol=1e-12,
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_long_only_solution_satisfies_kkt_conditions(self, seed: int) -> None:
        cov = _one_factor_cov(seed, k=8)
        w = min_variance_weights(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w.min() == 0.0
        assert _budget_kkt_violation(w, 2.0 * cov, np.zeros(8), long_only=True) < 1e-12

    def test_asset_excluded_by_the_long_only_constraint_has_weight_exactly_zero(self) -> None:
        # Asset 2 is asset 0 plus noise at three times the scale: any positive
        # holding of it raises the variance, so only a short position would
        # be used to hedge.
        cov = np.array([[0.04, 0.0, 0.12], [0.0, 0.09, 0.0], [0.12, 0.0, 0.45]])
        w = min_variance_weights(cov)
        assert w[2] == 0.0
        np.testing.assert_allclose(w[:2], np.array([0.09, 0.04]) / 0.13, atol=1e-12)
        assert min_variance_weights(cov, allow_short=True)[2] < 0.0

    @pytest.mark.parametrize("allow_short", [False, True])
    @pytest.mark.parametrize("seed", range(5))
    def test_near_singular_covariance_satisfies_kkt_conditions(
        self, seed: int, allow_short: bool
    ) -> None:
        cov = _near_singular_cov(seed, k=8)
        w = min_variance_weights(cov, allow_short)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert _budget_kkt_violation(w, 2.0 * cov, np.zeros(8), not allow_short) < 1e-12

    @pytest.mark.parametrize("scale", [1e-8, 1e-4, 1e4, 1e8])
    def test_invariant_to_the_units_of_the_covariance(self, scale: float) -> None:
        _, cov, _ = _random_problem(12, k=8)
        for allow_short in (False, True):
            np.testing.assert_allclose(
                min_variance_weights(scale * cov, allow_short),
                min_variance_weights(cov, allow_short),
                atol=1e-12,
            )

    def test_rank_deficient_covariance_is_solved(self) -> None:
        # Three assets driven by two factors: a combination with zero
        # variance exists, and that is the minimum.
        factors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cov = factors @ factors.T
        w = min_variance_weights(cov, allow_short=True)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w @ cov @ w == pytest.approx(0.0, abs=1e-24)

    def test_single_asset(self) -> None:
        np.testing.assert_array_equal(min_variance_weights(np.array([[0.04]])), [1.0])


def _no_rows(n: int) -> tuple[FloatArray, FloatArray]:
    return np.zeros((0, n)), np.zeros(0)


def _unbounded(n: int) -> tuple[FloatArray, FloatArray]:
    return np.full(n, -np.inf), np.full(n, np.inf)


def _qp_kkt_violations(
    x: FloatArray,
    hessian: FloatArray,
    linear: FloatArray,
    eq_matrix: FloatArray,
    eq_rhs: FloatArray,
    ineq_matrix: FloatArray,
    ineq_rhs: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[float, float]:
    """Check the Karush-Kuhn-Tucker conditions of a convex program at `x` from scratch.

    The program is `min 0.5*x'Hx + linear'x` subject to `eq_matrix @ x =
    eq_rhs`, `ineq_matrix @ x >= ineq_rhs` and `lower <= x <= upper`.

    Returns `(primal, dual)`. `primal` is the largest constraint violation.
    `dual` is how far the gradient is from a combination of constraint
    normals in which only constraints holding with equality take part
    (complementary slackness) and inequality constraints have non-negative
    multipliers (dual feasibility), relative to the size of the terms the
    gradient is summed from; zero means stationarity holds with admissible
    multipliers. The multipliers are fitted here by non-negative least
    squares, independently of the solver.
    """
    n = x.size
    gradient = hessian @ x + linear
    scale = max(float(np.max(np.abs(linear))), float(np.max(np.abs(hessian) @ np.abs(x))), 1e-300)
    primal = max(
        float(np.max(np.abs(eq_matrix @ x - eq_rhs), initial=0.0)),
        float(np.max(ineq_rhs - ineq_matrix @ x, initial=0.0)),
        float(np.max(lower - x)),
        float(np.max(x - upper)),
        0.0,
    )
    active_rows = ineq_matrix @ x - ineq_rhs <= 1e-9
    identity = np.eye(n)
    normals = np.column_stack(
        [
            eq_matrix.T,
            -eq_matrix.T,
            ineq_matrix[active_rows].T,
            identity[:, x - lower <= 1e-9],
            -identity[:, upper - x <= 1e-9],
        ]
    )
    multipliers, _ = nnls(normals, gradient, maxiter=50 * normals.shape[1] + 50)
    dual = float(np.max(np.abs(normals @ multipliers - gradient))) / scale
    return primal, dual


def _random_qp(seed: int, n: int, n_eq: int, n_ineq: int, rank: int) -> dict[str, FloatArray]:
    """A convex program with a non-empty, bounded feasible set and a Hessian of rank `rank`.

    Every constraint is built around one random point, which is strictly
    inside the inequality constraints and the bounds, so the program is
    feasible; the finite bounds keep it bounded when `rank < n`.
    """
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, rank))
    interior = rng.normal(size=n)
    eq_matrix = rng.normal(size=(n_eq, n))
    ineq_matrix = rng.normal(size=(n_ineq, n))
    return {
        "hessian": factors @ factors.T,
        "linear": rng.normal(size=n),
        "eq_matrix": eq_matrix,
        "eq_rhs": eq_matrix @ interior,
        "ineq_matrix": ineq_matrix,
        "ineq_rhs": ineq_matrix @ interior - rng.uniform(0.05, 1.0, n_ineq),
        "lower": interior - rng.uniform(0.05, 2.0, n),
        "upper": interior + rng.uniform(0.05, 2.0, n),
        "start": 3.0 * rng.normal(size=n),
    }


class TestConvexQpKernel:
    """`_solve_convex_qp`, the active-set solver behind the three quadratic programs."""

    @pytest.mark.parametrize("rank_deficit", [0, 2, 5])
    @pytest.mark.parametrize("seed", range(25))
    def test_kkt_conditions_hold_on_random_problems(self, seed: int, rank_deficit: int) -> None:
        n = 4 + seed % 6
        problem = _random_qp(
            seed, n, n_eq=seed % 3, n_ineq=1 + seed % 4, rank=max(n - rank_deficit, 0)
        )
        x = opt_module._solve_convex_qp(**problem, func_name="test")
        constraints = {key: value for key, value in problem.items() if key != "start"}
        primal, dual = _qp_kkt_violations(x, **constraints)
        assert primal < 1e-12
        assert dual < 1e-10
        assert np.all(x >= problem["lower"]) and np.all(x <= problem["upper"])

    def test_matches_closed_form_of_an_equality_constrained_problem(self) -> None:
        rng = np.random.default_rng(0)
        n, m = 7, 3
        a = rng.normal(size=(n, n))
        hessian, linear = a @ a.T + np.eye(n), rng.normal(size=n)
        eq_matrix, eq_rhs = rng.normal(size=(m, n)), rng.normal(size=m)
        kkt = np.block([[hessian, -eq_matrix.T], [eq_matrix, np.zeros((m, m))]])
        expected = np.linalg.solve(kkt, np.concatenate([-linear, eq_rhs]))[:n]
        x = opt_module._solve_convex_qp(
            hessian, linear, eq_matrix, eq_rhs, *_no_rows(n), *_unbounded(n), np.zeros(n), "test"
        )
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_unconstrained_minimum(self) -> None:
        hessian = np.array([[2.0, 0.5], [0.5, 1.0]])
        linear = np.array([-1.0, 3.0])
        x = opt_module._solve_convex_qp(
            hessian, linear, *_no_rows(2), *_no_rows(2), *_unbounded(2), np.zeros(2), "test"
        )
        np.testing.assert_allclose(x, np.linalg.solve(hessian, -linear), atol=1e-13)

    def test_variable_stopped_by_a_bound_equals_the_bound_exactly(self) -> None:
        # Unconstrained minimum at (2, -3); the box cuts both coordinates.
        hessian, linear = np.eye(2), np.array([-2.0, 3.0])
        x = opt_module._solve_convex_qp(
            hessian,
            linear,
            *_no_rows(2),
            *_no_rows(2),
            np.array([0.0, -1.0]),
            np.array([1.5, 4.0]),
            np.array([0.3, 0.3]),
            "test",
        )
        np.testing.assert_array_equal(x, [1.5, -1.0])

    def test_inequality_constraint_binds(self) -> None:
        # Closest point to (2, 2) with x0 + x1 <= 1, i.e. -x0 - x1 >= -1.
        x = opt_module._solve_convex_qp(
            np.eye(2),
            np.array([-2.0, -2.0]),
            *_no_rows(2),
            np.array([[-1.0, -1.0]]),
            np.array([-1.0]),
            *_unbounded(2),
            np.zeros(2),
            "test",
        )
        np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-14)

    @pytest.mark.parametrize("scale", [1e-8, 1e-4, 1e4, 1e8])
    def test_scaling_the_objective_leaves_the_minimiser_unchanged(self, scale: float) -> None:
        problem = _random_qp(3, n=8, n_eq=2, n_ineq=3, rank=8)
        reference = opt_module._solve_convex_qp(**problem, func_name="test")
        scaled = dict(problem, hessian=scale * problem["hessian"], linear=scale * problem["linear"])
        np.testing.assert_allclose(
            opt_module._solve_convex_qp(**scaled, func_name="test"), reference, atol=1e-12
        )

    def test_scaling_a_constraint_row_leaves_the_minimiser_unchanged(self) -> None:
        problem = _random_qp(4, n=6, n_eq=1, n_ineq=3, rank=6)
        reference = opt_module._solve_convex_qp(**problem, func_name="test")
        scaled = dict(
            problem,
            eq_matrix=1e6 * problem["eq_matrix"],
            eq_rhs=1e6 * problem["eq_rhs"],
            ineq_matrix=1e-6 * problem["ineq_matrix"],
            ineq_rhs=1e-6 * problem["ineq_rhs"],
        )
        np.testing.assert_allclose(
            opt_module._solve_convex_qp(**scaled, func_name="test"), reference, atol=1e-12
        )

    def test_start_does_not_change_a_unique_minimiser(self) -> None:
        problem = _random_qp(5, n=7, n_eq=1, n_ineq=4, rank=7)
        reference = opt_module._solve_convex_qp(**problem, func_name="test")
        rng = np.random.default_rng(99)
        for _ in range(5):
            moved = dict(problem, start=10.0 * rng.normal(size=7))
            np.testing.assert_allclose(
                opt_module._solve_convex_qp(**moved, func_name="test"), reference, atol=1e-11
            )

    def test_linear_program_reaches_the_optimal_vertex(self) -> None:
        # No curvature at all: minimise a linear cost over the unit simplex.
        cost = np.array([0.3, -0.2, 0.5, -0.7, 0.1])
        x = opt_module._solve_convex_qp(
            np.zeros((5, 5)),
            cost,
            np.ones((1, 5)),
            np.array([1.0]),
            *_no_rows(5),
            np.zeros(5),
            np.full(5, np.inf),
            np.full(5, 0.2),
            "test",
        )
        np.testing.assert_array_equal(x, [0.0, 0.0, 0.0, 1.0, 0.0])

    def test_flat_direction_with_no_slope_is_left_alone(self) -> None:
        # The objective ignores x1 and x2 entirely; they keep their start.
        hessian = np.diag([1.0, 0.0, 0.0])
        linear = np.array([-1.0, 0.0, 0.0])
        x = opt_module._solve_convex_qp(
            hessian,
            linear,
            *_no_rows(3),
            *_no_rows(3),
            *_unbounded(3),
            np.array([5.0, 0.25, -0.75]),
            "test",
        )
        np.testing.assert_array_equal(x, [1.0, 0.25, -0.75])

    def test_redundant_equality_rows_are_tolerated(self) -> None:
        problem = _random_qp(6, n=6, n_eq=2, n_ineq=2, rank=6)
        reference = opt_module._solve_convex_qp(**problem, func_name="test")
        eq_matrix, eq_rhs = problem["eq_matrix"], problem["eq_rhs"]
        repeated = dict(
            problem,
            eq_matrix=np.vstack([eq_matrix, eq_matrix[0] + 2.0 * eq_matrix[1], eq_matrix]),
            eq_rhs=np.concatenate([eq_rhs, [eq_rhs[0] + 2.0 * eq_rhs[1]], eq_rhs]),
        )
        np.testing.assert_allclose(
            opt_module._solve_convex_qp(**repeated, func_name="test"), reference, atol=1e-11
        )

    def test_repeated_inequality_rows_are_tolerated(self) -> None:
        problem = _random_qp(7, n=6, n_eq=1, n_ineq=3, rank=6)
        reference = opt_module._solve_convex_qp(**problem, func_name="test")
        repeated = dict(
            problem,
            ineq_matrix=np.vstack([problem["ineq_matrix"]] * 3),
            ineq_rhs=np.concatenate([problem["ineq_rhs"]] * 3),
        )
        np.testing.assert_allclose(
            opt_module._solve_convex_qp(**repeated, func_name="test"), reference, atol=1e-11
        )

    def test_more_constraints_active_at_the_optimum_than_dimensions(self) -> None:
        # Closest point of the unit square to (2, 2) is its corner (1, 1),
        # where two bounds and three more constraints all hold with equality.
        ineq_matrix = np.array([[-1.0, -1.0], [-2.0, -1.0], [-1.0, -3.0]])
        ineq_rhs = np.array([-2.0, -3.0, -4.0])
        for start in ([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.3, 0.9]):
            x = opt_module._solve_convex_qp(
                np.eye(2),
                np.array([-2.0, -2.0]),
                *_no_rows(2),
                ineq_matrix,
                ineq_rhs,
                np.zeros(2),
                np.ones(2),
                np.array(start),
                "test",
            )
            np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-14)

    def test_degenerate_vertex_start_is_left_when_it_is_not_optimal(self) -> None:
        # Start at the origin, where both bounds and both inequalities are
        # active; the optimum is the opposite corner of the feasible region.
        ineq_matrix = np.array([[1.0, 1.0], [1.0, 2.0]])
        x = opt_module._solve_convex_qp(
            np.zeros((2, 2)),
            np.array([-1.0, -1.0]),
            *_no_rows(2),
            ineq_matrix,
            np.zeros(2),
            np.zeros(2),
            np.array([2.0, 3.0]),
            np.zeros(2),
            "test",
        )
        np.testing.assert_array_equal(x, [2.0, 3.0])

    def test_infeasible_start_is_moved_onto_the_feasible_set(self) -> None:
        # Budget of one with weights capped at 0.4: the start violates both.
        x = opt_module._solve_convex_qp(
            np.eye(3),
            np.zeros(3),
            np.ones((1, 3)),
            np.array([1.0]),
            *_no_rows(3),
            np.zeros(3),
            np.full(3, 0.4),
            np.array([5.0, -2.0, 0.0]),
            "test",
        )
        np.testing.assert_allclose(x, np.full(3, 1.0 / 3.0), atol=1e-14)

    @pytest.mark.parametrize(
        "case", ["crossing_bounds", "inconsistent_equalities", "budget_above_caps", "opposed_rows"]
    )
    def test_infeasible_constraints_raise(self, case: str) -> None:
        n = 3
        eq_matrix, eq_rhs = _no_rows(n)
        ineq_matrix, ineq_rhs = _no_rows(n)
        lower, upper = _unbounded(n)
        if case == "crossing_bounds":
            lower, upper = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.5, 1.0])
        elif case == "inconsistent_equalities":
            eq_matrix, eq_rhs = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), np.array([1.0, 3.0])
        elif case == "budget_above_caps":
            eq_matrix, eq_rhs = np.ones((1, n)), np.array([1.0])
            lower, upper = np.zeros(n), np.full(n, 0.3)
        else:
            ineq_matrix = np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
            ineq_rhs = np.array([1.0, 0.5])
        with pytest.raises(RuntimeError, match="infeasible"):
            opt_module._solve_convex_qp(
                np.eye(n),
                np.zeros(n),
                eq_matrix,
                eq_rhs,
                ineq_matrix,
                ineq_rhs,
                lower,
                upper,
                np.zeros(n),
                "test",
            )

    def test_linear_objective_without_bounds_raises_unbounded(self) -> None:
        with pytest.raises(RuntimeError, match="unbounded"):
            opt_module._solve_convex_qp(
                np.zeros((2, 2)),
                np.array([1.0, -1.0]),
                np.ones((1, 2)),
                np.array([1.0]),
                *_no_rows(2),
                *_unbounded(2),
                np.array([0.5, 0.5]),
                "test",
            )

    def test_singular_hessian_with_a_slope_along_its_null_space_raises_unbounded(self) -> None:
        # Curvature only along (1, 1); the linear term slopes along (1, -1).
        with pytest.raises(RuntimeError, match="unbounded"):
            opt_module._solve_convex_qp(
                np.ones((2, 2)),
                np.array([1.0, -1.0]),
                *_no_rows(2),
                *_no_rows(2),
                *_unbounded(2),
                np.zeros(2),
                "test",
            )

    def test_bound_stops_a_slope_along_the_null_space(self) -> None:
        # f = 0.5*(x0 + x1)^2 + x0 - x1 falls without limit along (-1, 1)
        # until x1 reaches its upper bound of 2; with x1 = 2 it is minimised
        # at x0 = -3.
        x = opt_module._solve_convex_qp(
            np.ones((2, 2)),
            np.array([1.0, -1.0]),
            *_no_rows(2),
            *_no_rows(2),
            np.array([-10.0, -np.inf]),
            np.array([np.inf, 2.0]),
            np.zeros(2),
            "test",
        )
        np.testing.assert_allclose(x, [-3.0, 2.0], atol=1e-14)

    def test_negative_curvature_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not convex"):
            opt_module._solve_convex_qp(
                np.diag([1.0, -1.0]),
                np.array([0.0, 0.1]),
                *_no_rows(2),
                *_no_rows(2),
                np.full(2, -1.0),
                np.full(2, 1.0),
                np.zeros(2),
                "test",
            )

    @pytest.mark.parametrize("seed", range(6))
    def test_minimiser_does_not_depend_on_how_the_step_is_computed(
        self, seed: int, monkeypatch
    ) -> None:
        # A positive definite Hessian is handled through a Cholesky factor of
        # its free block; an infinite pivot requirement rules that out, which
        # leaves the null-space computation used for singular blocks.
        problem = _random_qp(seed, n=12, n_eq=seed % 3, n_ineq=3, rank=12)
        with_factor = opt_module._solve_convex_qp(**problem, func_name="test")
        monkeypatch.setattr(opt_module, "_QP_CHOLESKY_RTOL", np.inf)
        without_factor = opt_module._solve_convex_qp(**problem, func_name="test")
        np.testing.assert_allclose(with_factor, without_factor, atol=1e-11)

    def test_iteration_cap_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(opt_module, "_QP_MAX_ITER_PER_CONSTRAINT", 0)
        problem = _random_qp(8, n=5, n_eq=1, n_ineq=2, rank=5)
        with pytest.raises(RuntimeError, match="did not converge"):
            opt_module._solve_convex_qp(**problem, func_name="test")

    def test_error_message_names_the_calling_function(self) -> None:
        with pytest.raises(RuntimeError, match=r"^my_optimiser: "):
            opt_module._solve_convex_qp(
                np.zeros((1, 1)),
                np.array([1.0]),
                *_no_rows(1),
                *_no_rows(1),
                *_unbounded(1),
                np.zeros(1),
                "my_optimiser",
            )


_FINITE_COV = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])
_FINITE_MU = np.array([0.05, 0.03, 0.02])
_FINITE_PREVIOUS = np.array([0.2, 0.3, 0.5])


def _with_entry(array: np.ndarray, index: tuple[int, ...], value: float) -> np.ndarray:
    changed = array.copy()
    changed[index] = value
    return changed


class TestNonFiniteInputsAreRejected:
    """A NaN or infinite input raises `ValueError` instead of producing weights."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda cov: min_variance_weights(cov),
            lambda cov: unconstrained_mean_variance_weights(_FINITE_MU, cov, 3.0),
            lambda cov: risk_parity_weights(cov),
        ],
        ids=["min_variance", "unconstrained", "risk_parity"],
    )
    def test_non_finite_covariance_entry(self, solve, bad):
        cov = _with_entry(_FINITE_COV, (0, 1), bad)
        cov[1, 0] = bad
        with pytest.raises(ValueError, match="cov_matrix must contain only finite values"):
            solve(cov)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda mu: unconstrained_mean_variance_weights(mu, _FINITE_COV, 3.0),
        ],
        ids=["unconstrained"],
    )
    def test_non_finite_expected_return(self, solve, bad):
        with pytest.raises(ValueError, match="expected_returns must contain only finite values"):
            solve(_with_entry(_FINITE_MU, (1,), bad))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda gamma: unconstrained_mean_variance_weights(_FINITE_MU, _FINITE_COV, gamma),
        ],
        ids=["unconstrained"],
    )
    def test_non_finite_risk_aversion(self, solve, bad):
        with pytest.raises(ValueError, match="risk_aversion must be finite"):
            solve(bad)


_RISKY_BLOCK = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])


def _with_cash(risky_cov: np.ndarray) -> np.ndarray:
    """`risky_cov` with a first asset of zero variance, uncorrelated with the rest."""
    cov = np.zeros((risky_cov.shape[0] + 1, risky_cov.shape[0] + 1))
    cov[1:, 1:] = risky_cov
    return cov


class TestZeroVarianceAsset:
    """An asset with zero variance (cash) is a valid input and a legitimate optimum.

    With everything in cash the gradient of the variance is zero, and so is
    every term it is summed from; the optimum must still be recognised.
    """

    @pytest.mark.parametrize("allow_short", [False, True])
    def test_min_variance_puts_everything_in_cash(self, allow_short: bool) -> None:
        w = min_variance_weights(_with_cash(_RISKY_BLOCK), allow_short)
        np.testing.assert_allclose(w, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("allow_short", [False, True])
    def test_min_variance_on_a_diagonal_covariance_with_cash(self, allow_short: bool) -> None:
        w = min_variance_weights(np.diag([0.0, 0.04, 0.04, 0.09]), allow_short)
        np.testing.assert_allclose(w, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("allow_short", [False, True])
    def test_min_variance_with_two_cash_assets_is_bounded(self, allow_short: bool) -> None:
        # Any split between the two cash assets has zero variance; the
        # minimum is not unique but it exists.
        cov = np.diag([0.0, 4.0, 0.0, 4.0])
        w = min_variance_weights(cov, allow_short)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(w[[1, 3]], 0.0, atol=1e-12)
        assert w @ cov @ w == pytest.approx(0.0, abs=1e-24)

    def test_min_variance_resolves_weights_far_below_the_round_off_of_the_largest(self) -> None:
        # "Cash" with a variance of 1e-30: the risky weights are 1e-30 *
        # Sigma^-1 1, thirteen orders of magnitude below one unit of
        # round-off of the cash weight.
        cov = _with_cash(_RISKY_BLOCK)
        cov[0, 0] = 1e-30
        w = min_variance_weights(cov, allow_short=True)
        assert w[0] == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            w[1:], 1e-30 * np.linalg.solve(_RISKY_BLOCK, np.ones(3)), rtol=1e-9
        )


_INDEFINITE_COVARIANCES = {
    # Eigenvalues -5, 1 and 15: (0, 0.5, 0.5) has "variance" -2.5.
    "three_assets": np.array([[1.0, 2.0, 2.0], [2.0, 5.0, -10.0], [2.0, -10.0, 5.0]]),
    # Correlation of 2 between two unit-variance assets.
    "two_assets": np.array([[1.0, 2.0], [2.0, 1.0]]),
    "negative_definite": -np.eye(3),
    # Positive semi-definite up to one eigenvalue of -1e-6 times the largest.
    "slightly_indefinite": np.diag([1.0, 1.0, -1e-6]),
}


class TestCovarianceMustBePositiveSemiDefinite:
    """The three quadratic programs are convex only for a positive semi-definite covariance."""

    @pytest.mark.parametrize("allow_short", [False, True])
    @pytest.mark.parametrize("name", list(_INDEFINITE_COVARIANCES))
    def test_min_variance_rejects_an_indefinite_covariance(
        self, name: str, allow_short: bool
    ) -> None:
        with pytest.raises(ValueError, match="positive semi-definite"):
            min_variance_weights(_INDEFINITE_COVARIANCES[name], allow_short)

    def test_asymmetric_matrix_is_judged_by_its_symmetric_part(self) -> None:
        # w'Aw only involves the symmetric part of A, which is positive
        # definite here although A itself is not symmetric.
        cov = np.array([[1.0, 0.9], [0.1, 2.0]])
        w = min_variance_weights(cov, allow_short=True)
        np.testing.assert_allclose(w, [0.75, 0.25], atol=1e-12)

    @pytest.mark.parametrize("k", [40, 300])
    def test_rank_deficient_covariance_is_accepted(self, k: int) -> None:
        # Three factors and no specific risk: k - 3 eigenvalues are zero in
        # exact arithmetic and of either sign, at round-off size, as computed.
        rng = np.random.default_rng(k)
        factors = rng.normal(size=(k, 3))
        cov = factors @ factors.T

        for allow_short in (False, True):
            w = min_variance_weights(cov, allow_short)
            assert w.sum() == pytest.approx(1.0, abs=1e-10)
            assert _budget_kkt_violation(w, 2.0 * cov, np.zeros(k), not allow_short) < 1e-10

    def test_zero_covariance_is_accepted(self) -> None:
        w = min_variance_weights(np.zeros((3, 3)))
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w.min() >= 0.0

    def test_single_asset_with_zero_variance_is_accepted(self) -> None:
        np.testing.assert_array_equal(min_variance_weights(np.array([[0.0]])), [1.0])


class TestExtremeScales:
    """Inputs too large or too small for double precision raise instead of returning garbage."""

    def test_min_variance_raises_when_the_covariance_overflows(self) -> None:
        with pytest.raises(RuntimeError, match="not finite"):
            min_variance_weights(np.eye(3) * 1e308)


class TestLargeProblems:
    """Hundreds of assets, most of them held: the optimum is still certified exactly."""

    @pytest.mark.parametrize("k", [150, 400])
    def test_long_only_solutions_satisfy_kkt_conditions(self, k: int) -> None:
        rng = np.random.default_rng(k)
        cov = np.cov((rng.standard_normal((2 * k, k)) * 0.01).T)

        w = min_variance_weights(cov)
        assert np.count_nonzero(w) > k // 2
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w.min() >= 0.0
        assert _budget_kkt_violation(w, 2.0 * cov, np.zeros(k), long_only=True) < 1e-12


class TestLongOnlyStart:
    """`_long_only_start` gives a feasible guess; the solver's answer does not depend on it."""

    @staticmethod
    def _factor_covariance(k: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        loadings = rng.normal(size=(k, 3)) * 0.01
        return loadings @ loadings.T + np.diag(rng.uniform(1e-6, 1e-4, k))

    @pytest.mark.parametrize("seed", range(5))
    def test_is_non_negative_and_fully_invested(self, seed: int) -> None:
        cov = self._factor_covariance(30, seed)
        mu = np.random.default_rng(seed).normal(0.0, 0.001, 30)
        fallback = np.eye(30)[0]
        start = opt_module._long_only_start(2.0 * cov, -mu, fallback)
        assert start.min() >= 0.0
        assert start.sum() == pytest.approx(1.0, abs=1e-14)

    def test_returns_the_fallback_for_a_singular_hessian(self) -> None:
        loadings = np.random.default_rng(0).normal(size=(6, 2))
        fallback = np.eye(6)[3]
        start = opt_module._long_only_start(loadings @ loadings.T, np.zeros(6), fallback)
        np.testing.assert_array_equal(start, fallback)

    def test_returns_the_fallback_for_a_zero_hessian(self) -> None:
        fallback = np.eye(4)[1]
        start = opt_module._long_only_start(np.zeros((4, 4)), -np.arange(4.0), fallback)
        np.testing.assert_array_equal(start, fallback)

    @pytest.mark.parametrize("seed", range(5))
    def test_solution_is_the_same_from_a_single_asset_start(self, seed: int) -> None:
        cov = self._factor_covariance(25, seed)
        hessian = 2.0 * cov
        vertex = np.zeros(25)
        vertex[int(np.argmin(np.diag(hessian)))] = 1.0
        from_vertex = opt_module._fully_invested_minimiser(
            hessian, np.zeros(25), vertex, False, "test"
        )
        np.testing.assert_allclose(min_variance_weights(cov), from_vertex, rtol=0.0, atol=1e-12)

    def test_many_held_assets_need_far_fewer_iterations_than_a_single_asset_start(self) -> None:
        # Every asset is held at the optimum here, so a single-asset start
        # frees them one iteration at a time. Both solves are timed in the
        # same test so that machine load cancels out of the comparison.
        cov = self._factor_covariance(400, 1) + np.diag(np.full(400, 1e-4))
        hessian = 2.0 * cov
        vertex = np.zeros(400)
        vertex[int(np.argmin(np.diag(hessian)))] = 1.0

        started = time.perf_counter()
        from_vertex = opt_module._fully_invested_minimiser(
            hessian, np.zeros(400), vertex, False, "test"
        )
        vertex_seconds = time.perf_counter() - started
        started = time.perf_counter()
        weights = min_variance_weights(cov)
        seconds = time.perf_counter() - started

        assert int((weights > 0.0).sum()) > 300
        np.testing.assert_allclose(weights, from_vertex, rtol=0.0, atol=1e-12)
        assert seconds < vertex_seconds / 3.0
