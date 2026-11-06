"""Oracle tests for `ols` / `factor_loadings` / `newey_west_optimal_lags` vs statsmodels.

Covers finding-list items (owner follow-up message, 2a):
- `ols`: uncentered R^2 when `x` has no intercept column; NaN (not 1.0) R^2
  for a constant dependent variable.
- `factor_loadings`: standard errors / p-values / R^2 / residuals now
  returned; `n_lags_nw=None` selects a sample-size-based lag count
  automatically rather than always defaulting to 4.
- `newey_west_optimal_lags`: documentation/citation only in this pass (see
  its docstring); the computation itself is unchanged and still matches the
  closed-form rule.

Oracle: `statsmodels.api.OLS` (with and without a constant column) for R^2
conventions; `statsmodels.regression.linear_model.OLS(...).fit(cov_type="HAC", ...)`
for HAC standard errors / t-stats / p-values.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

import statsmodels.api as sm

from quantcore.statistics.regression import factor_loadings, newey_west_cov, ols

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestOlsMatchesStatsmodels:
    def test_with_intercept_uses_centered_r_squared(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        x = rng.normal(size=(200, 2))
        design = sm.add_constant(x)
        y = design @ np.array([0.5, 1.2, -0.7]) + rng.normal(scale=0.5, size=200)

        coefficients, residuals, r_squared = ols(y, design)
        sm_result = sm.OLS(y, design).fit()

        np.testing.assert_allclose(coefficients, sm_result.params, rtol=1e-8)
        np.testing.assert_allclose(residuals, sm_result.resid, rtol=1e-6, atol=1e-10)
        assert r_squared == pytest.approx(sm_result.rsquared, rel=1e-8)

    def test_without_intercept_uses_uncentered_r_squared(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        x = rng.normal(loc=3.0, size=(200, 2))  # no constant column
        y = x @ np.array([1.2, -0.7]) + rng.normal(scale=0.5, size=200)

        _, _, r_squared = ols(y, x)
        sm_result = sm.OLS(y, x).fit()  # statsmodels auto-detects no constant

        assert r_squared == pytest.approx(sm_result.rsquared, rel=1e-8)

    def test_constant_y_r_squared_is_undefined_not_one(self) -> None:
        x = sm.add_constant(np.arange(20, dtype=np.float64))
        y = np.full(20, 5.0)
        _, _, r_squared = ols(y, x)
        assert np.isnan(r_squared)


class TestFactorLoadingsMatchesStatsmodelsHac:
    def test_loadings_and_hac_inference_match(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_obs = 400
        factors = rng.normal(scale=0.01, size=(n_obs, 3))
        true_betas = np.array([1.1, -0.4, 0.2])
        returns = factors @ true_betas + rng.normal(scale=0.005, size=n_obs)

        result = factor_loadings(returns, factors, n_lags_nw=5)

        design = sm.add_constant(factors)
        sm_result = sm.OLS(returns, design).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        np.testing.assert_allclose(result.loadings, sm_result.params, rtol=1e-6)
        np.testing.assert_allclose(result.residuals, sm_result.resid, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(result.standard_errors, sm_result.bse, rtol=1e-4)
        np.testing.assert_allclose(result.t_stats, sm_result.tvalues, rtol=1e-4)
        np.testing.assert_allclose(result.p_values, sm_result.pvalues, atol=1e-4)
        assert result.r_squared == pytest.approx(sm_result.rsquared, rel=1e-6)

    def test_auto_n_lags_nw_matches_the_rule_applied_manually(self) -> None:
        from quantcore.statistics.regression import newey_west_optimal_lags

        rng = np.random.default_rng(RNG_SEED)
        n_obs = 250
        factors = rng.normal(scale=0.01, size=(n_obs, 2))
        returns = factors @ np.array([1.0, -0.3]) + rng.normal(scale=0.01, size=n_obs)

        auto_result = factor_loadings(returns, factors)
        expected_lags = newey_west_optimal_lags(np.zeros(n_obs))
        manual_result = factor_loadings(returns, factors, n_lags_nw=expected_lags)

        assert auto_result.n_lags_nw == expected_lags
        np.testing.assert_allclose(auto_result.standard_errors, manual_result.standard_errors)


class TestNeweyWestSmallSampleCorrectionMatchesStatsmodels:
    def test_matches_use_correction_true(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_obs = 200
        design = sm.add_constant(rng.normal(size=(n_obs, 2)))
        y = design @ np.array([0.5, 1.2, -0.7]) + rng.normal(scale=0.5, size=n_obs)
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients

        quantcore_corrected = newey_west_cov(
            design, residuals, n_lags=4, small_sample_correction=True
        )
        sm_result = sm.OLS(y, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}
        )
        np.testing.assert_allclose(quantcore_corrected, sm_result.cov_params(), rtol=1e-6)
