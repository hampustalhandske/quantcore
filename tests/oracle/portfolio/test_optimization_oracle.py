"""Oracle tests for the portfolio optimizers' formulas vs `cvxpy`.

Covers minimum-variance and risk-parity weights vs `cvxpy` formulations
(constraints satisfied, objective within tolerance).

`min_variance_weights` is a convex quadratic program. It is
compared with the identical program solved by cvxpy's CLARABEL
interior-point solver, in two ways:

- on the objective both solvers optimise, evaluated at each solver's own
  weights. CLARABEL stops once its duality gap and constraint residuals are
  below 1e-8 (absolute below 1 in magnitude, relative above), so the two
  objectives are required to agree to `1e-8 + 1e-6 * |objective|`. The
  inputs include near-singular and wide-spectrum covariances, on which the
  objective is nearly flat along some directions, and returns and
  covariances in daily units, where the objective itself is of order 1e-4.
- on the weights, for a well-conditioned covariance and with CLARABEL's
  three stopping tolerances tightened to 1e-12. A duality gap of `gap` on an
  objective with curvature at least `m` leaves the weights within
  `sqrt(2 * gap / m)` of the optimum; `m >= 0.2` for the matrices used
  here, which gives 3e-6, and the weights are required to agree to 1e-5.

For `risk_parity_weights` the cvxpy problem is Spinu (2013)'s convex
reformulation (minimize `0.5*w'*Sigma*w - sum(b_i*log(w_i))` over `w > 0`,
then normalize to sum to 1) solved by an independent interior-point solver,
against quantcore's own damped-Newton solution of the same problem;
agreement to cvxpy's own accuracy confirms quantcore lands on the unique
risk-parity portfolio.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxpy")

import cvxpy as cp

from quantcore.portfolio.optimization import (
    min_variance_weights,
    risk_parity_weights,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921

# CLARABEL's default duality-gap and feasibility tolerance, and the relative
# margin allowed on top of it when two attained objectives are compared.
OBJECTIVE_ATOL = 1e-8
OBJECTIVE_RTOL = 1e-6

# Stopping tolerances for the solves whose weights are compared, and the
# weight tolerance they justify (see the module docstring).
TIGHT_CLARABEL = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}
WEIGHTS_ATOL = 1e-5


def _random_cov(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    return a @ a.T + np.eye(k) * 0.1


def _spectrum_cov(family: str, k: int, seed: int) -> np.ndarray:
    """Covariance `Q diag(eigenvalues) Q'` with a random orthogonal `Q`.

    `near_singular` draws eigenvalues in [0.5, 2] and sets one to 1e-6;
    `high_condition` draws them log-uniformly between 1e-3 and 1e3.
    """
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(0.0, 1.0, (k, k)))
    if family == "near_singular":
        eigenvalues = rng.uniform(0.5, 2.0, k)
        eigenvalues[0] = 1e-6
    elif family == "high_condition":
        eigenvalues = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), k))
    else:
        raise ValueError(f"unknown covariance family {family!r}")
    return (orthogonal * eigenvalues) @ orthogonal.T


def _daily_scale_problem(seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Expected returns and covariance in daily units: volatilities of 0.8% to 2.5% a day."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.025, k)
    a = rng.normal(size=(k, k))
    corr = a @ a.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)
    return rng.uniform(-0.0002, 0.0008, k), corr * np.outer(vols, vols)


def _variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def _solve_fully_invested(
    w: cp.Variable, objective: cp.Minimize | cp.Maximize, allow_short: bool, **solver_options: float
) -> np.ndarray:
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, **solver_options)
    assert problem.status == cp.OPTIMAL
    return np.asarray(w.value, dtype=np.float64)


def _cvxpy_min_variance(cov: np.ndarray, allow_short: bool, **solver_options: float) -> np.ndarray:
    w = cp.Variable(cov.shape[0])
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    return _solve_fully_invested(w, objective, allow_short, **solver_options)


class TestMinVarianceWeightsMatchesCvxpy:
    @pytest.mark.parametrize("allow_short", [False, True])
    def test_matches(self, allow_short: bool) -> None:
        cov = _random_cov(4, RNG_SEED)
        cvxpy_w = _cvxpy_min_variance(cov, allow_short, **TIGHT_CLARABEL)

        quantcore_w = min_variance_weights(cov, allow_short)
        np.testing.assert_allclose(quantcore_w, cvxpy_w, atol=WEIGHTS_ATOL)

    @pytest.mark.parametrize(
        ("family", "k", "seed", "allow_short"),
        [
            ("near_singular", 9, 60084, True),
            ("near_singular", 6, 8, True),
            ("near_singular", 7, 3, False),
            ("high_condition", 8, 5691, True),
            ("high_condition", 10, 10, False),
        ],
    )
    def test_attains_the_minimum_variance_on_ill_conditioned_covariance(
        self, family: str, k: int, seed: int, allow_short: bool
    ) -> None:
        cov = _spectrum_cov(family, k, seed)
        cvxpy_variance = _variance(_cvxpy_min_variance(cov, allow_short), cov)

        quantcore_w = min_variance_weights(cov, allow_short)
        assert quantcore_w.sum() == pytest.approx(1.0, abs=1e-12)
        if not allow_short:
            assert quantcore_w.min() >= 0.0
        assert _variance(quantcore_w, cov) == pytest.approx(
            cvxpy_variance, rel=OBJECTIVE_RTOL, abs=OBJECTIVE_ATOL
        )


class TestRiskParityWeightsMatchesSpinuConvexReformulation:
    def test_matches(self) -> None:
        cov = _random_cov(4, RNG_SEED)
        k = cov.shape[0]
        b = np.full(k, 1.0 / k)

        w = cp.Variable(k, pos=True)
        objective = 0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - b @ cp.log(w)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve()
        cvxpy_w = np.asarray(w.value) / np.sum(w.value)

        quantcore_w = risk_parity_weights(cov)
        np.testing.assert_allclose(quantcore_w, cvxpy_w, atol=1e-4)

    def test_risk_contributions_equal_at_the_cvxpy_solution_too(self) -> None:
        # Independent confirmation that Spinu's reformulation really does
        # solve the same "equal risk contribution" problem quantcore's own
        # objective targets.
        cov = _random_cov(5, RNG_SEED + 1)
        k = cov.shape[0]
        b = np.full(k, 1.0 / k)
        w = cp.Variable(k, pos=True)
        objective = 0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - b @ cp.log(w)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve()
        cvxpy_w = np.asarray(w.value) / np.sum(w.value)

        marginal = cov @ cvxpy_w
        contributions = cvxpy_w * marginal
        assert contributions.max() - contributions.min() < 1e-4
