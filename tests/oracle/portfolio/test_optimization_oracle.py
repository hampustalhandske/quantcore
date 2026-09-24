"""Oracle tests for the portfolio optimizers' formulas vs `cvxpy`.

Covers Workstream D: "mean-variance, min-variance, risk parity, L1-turnover
weights vs `cvxpy` formulations (constraints satisfied, objective within
tolerance)."

Each quantcore SLSQP-based optimizer is checked against an independent
convex reformulation solved by `cvxpy`. For `risk_parity_weights` the
cvxpy problem is Spinu (2013)'s convex reformulation (minimize
`0.5*w'*Sigma*w - sum(b_i*log(w_i))` over `w > 0`, then normalize to sum
to 1) solved by an independent interior-point solver, against quantcore's
own damped-Newton solution of the same problem; agreement to cvxpy's own
accuracy confirms quantcore lands on the unique risk-parity portfolio.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxpy")

import cvxpy as cp

from quantcore.portfolio.optimization import (
    l1_turnover_penalized_weights,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_cov(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    return a @ a.T + np.eye(k) * 0.1


class TestMinVarianceWeightsMatchesCvxpy:
    def test_matches(self) -> None:
        cov = _random_cov(4, RNG_SEED)
        w = cp.Variable(4)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0])
        prob.solve()

        quantcore_w = min_variance_weights(cov)
        np.testing.assert_allclose(quantcore_w, w.value, atol=1e-4)


class TestMeanVarianceWeightsMatchesCvxpy:
    def test_matches(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        cov = _random_cov(4, RNG_SEED)
        mu = rng.normal(scale=0.05, size=4)
        risk_aversion = 3.0

        w = cp.Variable(4)
        objective = w @ mu - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
        prob = cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0])
        prob.solve()

        quantcore_w = mean_variance_weights(mu, cov, risk_aversion)
        np.testing.assert_allclose(quantcore_w, w.value, atol=1e-4)


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


class TestL1TurnoverPenalizedWeightsMatchesCvxpy:
    def test_matches(self) -> None:
        # Objective value (the principled optimizer-vs-optimizer check per
        # tests/oracle/README.md's tolerance policy) is checked tightly;
        # the individual weights get a looser bound, since the L1 penalty
        # makes the objective comparatively flat in some directions near
        # the optimum -- two solvers can land on visibly different points
        # with essentially identical objective values there.
        rng = np.random.default_rng(RNG_SEED)
        cov = _random_cov(4, RNG_SEED)
        mu = rng.normal(scale=0.05, size=4)
        risk_aversion = 2.0
        previous_weights = np.full(4, 0.25)
        cost_bps = 20.0

        w = cp.Variable(4)
        objective = (
            w @ mu
            - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            - (cost_bps / 10000.0) * cp.norm1(w - previous_weights)
        )
        prob = cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0])
        prob.solve()
        cvxpy_objective = float(objective.value)

        quantcore_w = l1_turnover_penalized_weights(
            mu, cov, previous_weights, cost_bps, risk_aversion
        )
        quantcore_objective = (
            quantcore_w @ mu
            - 0.5 * risk_aversion * quantcore_w @ cov @ quantcore_w
            - (cost_bps / 10000.0) * np.sum(np.abs(quantcore_w - previous_weights))
        )

        assert quantcore_objective == pytest.approx(cvxpy_objective, abs=1e-6)
        np.testing.assert_allclose(quantcore_w, w.value, atol=1e-3)
