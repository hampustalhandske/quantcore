"""Oracle tests for `var_fit`/`impulse_response`/`granger_causality_test` vs statsmodels.

Covers Workstream D: "ARIMA, VAR, IRF, Granger vs statsmodels."

Found and fixed a real defect along the way: `var_fit`'s `sigma_u` used
`np.cov(residuals, ddof=1)` -- dividing by `T_eff - 1`, the correction
appropriate for a 1-parameter (mean-only) estimate. A VAR(p) equation
estimates `k*p + 1` parameters, so Lutkepohl (2005)'s (and statsmodels')
unbiased estimator divides by `T_eff - (k*p + 1)` instead. Fixed; now
matches `statsmodels.tsa.api.VAR` exactly.

Found and fixed a second defect: `select_var_lag_order` computed each
candidate lag order's information criterion on `n_obs - n_lags`
observations -- a *different* effective sample size per candidate.
Information criteria are only comparable across models fit on the same
effective sample size (Lutkepohl 2005, pp. 146-150); `statsmodels`
enforces this by fitting every candidate on the same trailing window of
`n_obs - max_lags` observations (`VAR.select_order`'s `offset=maxlags-p`).
quantcore's varying-sample-size version could select a genuinely
different lag order in small samples (confirmed: on one 40-observation,
2-variable synthetic series, AIC selected 1 instead of the fixed-sample
answer of 2). Fixed to hold `n_obs - max_lags` fixed across all
candidates and to use the maximum-likelihood (not `var_fit`'s bias-
corrected) residual covariance in the criterion, matching statsmodels'
`sigma_u_mle`-based formula exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from quantcore.time_series.var_model import (
    granger_causality_test,
    impulse_response,
    select_var_lag_order,
    var_fit,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _simulate_var1(n: int, a1: np.ndarray, c: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = a1.shape[0]
    data = np.zeros((n, k))
    for t in range(1, n):
        data[t] = c + a1 @ data[t - 1] + rng.normal(scale=0.1, size=k)
    return data


A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
C = np.array([0.1, -0.05])


class TestVarFitMatchesStatsmodels:
    def test_coefficients_and_sigma_u_match(self) -> None:
        data = _simulate_var1(300, A1, C, seed=RNG_SEED)

        quantcore_coef, quantcore_sigma_u = var_fit(data, n_lags=1)

        model = VAR(data)
        result = model.fit(1, trend="c")

        np.testing.assert_allclose(quantcore_coef[:, :2], result.params[1:].T, rtol=1e-8)
        np.testing.assert_allclose(quantcore_coef[:, -1], result.params[0], rtol=1e-8)
        np.testing.assert_allclose(quantcore_sigma_u, result.sigma_u, rtol=1e-8)


class TestImpulseResponseMatchesStatsmodels:
    def test_orthogonalized_irf_matches(self) -> None:
        data = _simulate_var1(300, A1, C, seed=RNG_SEED)
        coef_matrix, sigma_u = var_fit(data, n_lags=1)
        quantcore_irf = impulse_response(coef_matrix, sigma_u, n_lags=1, n_periods=6)

        model = VAR(data)
        result = model.fit(1, trend="c")
        sm_irf = result.irf(5).orth_irfs

        np.testing.assert_allclose(quantcore_irf, sm_irf, rtol=1e-6, atol=1e-10)


class TestGrangerCausalityTestMatchesStatsmodels:
    def test_f_statistic_and_pvalue_match(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 300
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.normal(scale=0.1)
            y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal(scale=0.1)

        quantcore_f, quantcore_p = granger_causality_test(y, x, n_lags=2)

        data = np.column_stack([y, x])
        sm_result = grangercausalitytests(data, maxlag=[2])
        sm_f, sm_p, _, _ = sm_result[2][0]["ssr_ftest"]

        assert quantcore_f == pytest.approx(sm_f, rel=1e-8)
        assert quantcore_p == pytest.approx(sm_p, rel=1e-6)


class TestSelectVarLagOrderMatchesStatsmodels:
    def test_selected_lag_matches_on_a_larger_sample(self) -> None:
        data = _simulate_var1(300, A1, C, seed=RNG_SEED)

        quantcore_aic = select_var_lag_order(data, max_lags=8, criterion="aic")
        quantcore_bic = select_var_lag_order(data, max_lags=8, criterion="bic")

        result = VAR(data).select_order(8)

        assert quantcore_aic == result.aic
        assert quantcore_bic == result.bic

    def test_selected_lag_matches_on_a_small_sample_where_the_old_bug_disagreed(self) -> None:
        # Regression test for the fixed-sample-size defect: on this small
        # sample, the old varying-n_eff implementation selected a
        # different lag order (AIC=1) than the fixed-sample-size answer
        # (AIC=2, matching statsmodels).
        rng = np.random.default_rng(7)
        n = 40
        a1 = np.array([[0.3, 0.0], [0.0, 0.3]])
        data = np.zeros((n, 2))
        for t in range(1, n):
            data[t] = a1 @ data[t - 1] + rng.normal(scale=0.1, size=2)

        quantcore_aic = select_var_lag_order(data, max_lags=10, criterion="aic")
        result = VAR(data).select_order(10)

        assert quantcore_aic == result.aic == 2

    def test_criterion_values_match_statsmodels_numerically(self) -> None:
        # Beyond just the argmin, the underlying AIC/BIC scores themselves
        # (up to an additive constant that cancels in the argmin but not
        # in a direct value comparison) should be exactly reproducible
        # from statsmodels' own per-order info_criteria.
        data = _simulate_var1(300, A1, C, seed=RNG_SEED)
        n_obs, k = data.shape
        max_lags = 5

        model = VAR(data)
        for n_lags in range(1, max_lags + 1):
            sm_result = model._estimate_var(n_lags, offset=max_lags - n_lags, trend="c")
            sm_aic = sm_result.info_criteria["aic"]
            sm_bic = sm_result.info_criteria["bic"]

            n_eff = n_obs - max_lags
            x_window = data[max_lags - n_lags :]
            from quantcore.time_series.var_model import _build_lagged_design

            x = _build_lagged_design(x_window, n_lags)
            y = data[max_lags:]
            coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            residuals = y - x @ coefs
            sigma_u_mle = (residuals.T @ residuals) / n_eff
            _, log_det = np.linalg.slogdet(sigma_u_mle)
            free_params = n_lags * k**2 + k

            quantcore_aic = log_det + (2.0 / n_eff) * free_params
            quantcore_bic = log_det + (np.log(n_eff) / n_eff) * free_params

            assert quantcore_aic == pytest.approx(sm_aic, rel=1e-8)
            assert quantcore_bic == pytest.approx(sm_bic, rel=1e-8)
