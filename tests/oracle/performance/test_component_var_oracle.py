"""Oracle test for `component_var`'s own formula vs an independent hand derivation.

Covers Workstream D: "`component_var`'s own formula (its sum-to-
portfolio-VaR identity is verified elsewhere -- the Gaussian-parametric
formula itself vs. an independent source is not)."

`component_var` implements the standard parametric (Gaussian) Component
VaR decomposition (Garman 1996, "Improving on VaR"; Jorion, P. (2006),
*Value at Risk*, 3rd ed., Chapter 7): each asset's contribution is its
marginal contribution to portfolio risk, scaled into VaR units --

    ComponentVaR_i = w_i * (Sigma @ w)_i / portfolio_std * z_alpha

No third-party library exposes this exact decomposition as a public
function to compare against directly (it is standard textbook material,
not something e.g. `empyrical` or `QuantLib` implements as a named
routine), so this test computes the expected values from scratch in plain
NumPy/SciPy -- independent of `quantcore`'s own implementation -- for a
concrete numeric case, and pins the exact result.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quantcore.performance.metrics import component_var

pytestmark = pytest.mark.oracle


class TestComponentVarMatchesHandDerivedGaussianDecomposition:
    def test_two_asset_case(self) -> None:
        weights = np.array([0.6, 0.4])
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
        confidence_level = 0.95

        z_alpha = norm.ppf(confidence_level)
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        marginal = cov_matrix @ weights
        expected = weights * marginal / portfolio_std * z_alpha

        actual = component_var(weights, cov_matrix, confidence_level)

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(actual, np.array([0.15075333, 0.15075333]), atol=1e-8)

    def test_three_asset_case_sums_to_independently_computed_portfolio_var(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        cov_matrix = np.array(
            [
                [0.02, 0.003, -0.001],
                [0.003, 0.05, 0.004],
                [-0.001, 0.004, 0.01],
            ]
        )
        confidence_level = 0.99

        z_alpha = norm.ppf(confidence_level)
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        expected_portfolio_var = z_alpha * portfolio_std

        actual = component_var(weights, cov_matrix, confidence_level)

        assert float(np.sum(actual)) == pytest.approx(expected_portfolio_var, rel=1e-10)
