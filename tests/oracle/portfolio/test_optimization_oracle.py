"""Oracle tests for the portfolio optimizers' formulas vs `cvxpy`.

Covers risk-parity weights vs a `cvxpy` formulation (constraints satisfied, objective
within tolerance).

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
    risk_parity_weights,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_cov(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    return a @ a.T + np.eye(k) * 0.1


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
