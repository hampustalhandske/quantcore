"""Tests for Ljung-Box autocorrelation testing and ARIMA(p,d,q) fitting/forecasting."""

from __future__ import annotations

import time

import numpy as np
import pytest

from quantcore.time_series.arima import (
    _css_objective,
    _css_residuals_numba_kernel,
    _difference,
    arima_fit,
    arima_forecast,
    arima_residuals,
    ljung_box_test,
    select_arima_order,
)

RNG_SEED = 7


def _simulate_ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, size=n)
    series = np.empty(n, dtype=np.float64)
    series[0] = innovations[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + innovations[t]
    return series


class TestLjungBoxTest:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ljung_box_test(np.array([]), n_lags=5)

    def test_invalid_n_lags_non_positive(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=0)

    def test_invalid_n_lags_too_large(self) -> None:
        residuals = np.arange(10, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=10)

    def test_white_noise_not_rejected(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=500)
        _, p_value = ljung_box_test(residuals, n_lags=10)
        assert p_value > 0.05

    def test_autocorrelated_series_rejected(self) -> None:
        series = _simulate_ar1(phi=0.9, n=500, seed=RNG_SEED)
        _, p_value = ljung_box_test(series, n_lags=10)
        assert p_value < 0.05

    def test_invalid_negative_model_df_raises(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=5, model_df=-1)

    def test_invalid_model_df_at_least_n_lags_raises(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=5, model_df=5)

    def test_default_model_df_is_zero(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=200)
        stat_default, p_default = ljung_box_test(residuals, n_lags=10)
        stat_explicit, p_explicit = ljung_box_test(residuals, n_lags=10, model_df=0)
        assert stat_default == pytest.approx(stat_explicit)
        assert p_default == pytest.approx(p_explicit)

    def test_model_df_reduces_degrees_of_freedom_and_pvalue(self) -> None:
        # Same Q statistic, fewer df -> the survival function is evaluated
        # further into the chi-squared tail -> a smaller (not larger)
        # p-value for model_df > 0, all else equal.
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=200)
        q_no_adjustment, p_no_adjustment = ljung_box_test(residuals, n_lags=10, model_df=0)
        q_adjusted, p_adjusted = ljung_box_test(residuals, n_lags=10, model_df=3)
        assert q_adjusted == pytest.approx(q_no_adjustment)
        assert p_adjusted < p_no_adjustment


class TestArimaFit:
    def test_invalid_inputs_empty_series(self) -> None:
        with pytest.raises(ValueError):
            arima_fit(np.array([]), p=1, d=0, q=0)

    def test_invalid_negative_orders(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=-1, d=0, q=0)

    def test_invalid_d_out_of_range(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=1, d=3, q=0)

    def test_invalid_p_and_q_both_zero(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=0, d=0, q=0)

    def test_ar1_recovers_known_coefficient(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=0)
        assert ar_coefs.shape == (1,)
        assert ma_coefs.shape == (0,)
        assert ar_coefs[0] == pytest.approx(0.6, abs=0.15)
        assert sigma2 > 0.0


class TestSelectArimaOrder:
    def test_invalid_criterion_raises(self) -> None:
        series = _simulate_ar1(phi=0.6, n=100, seed=RNG_SEED)
        with pytest.raises(ValueError):
            select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="hqic")

    def test_does_not_crash_on_infeasible_small_orders(self) -> None:
        # A short series makes most (p, d, q) candidates infeasible for
        # arima_fit; select_arima_order must silently skip them rather than
        # propagate the ValueError, as long as at least one candidate fits.
        series = _simulate_ar1(phi=0.5, n=6, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=3, max_d=2, max_q=3, criterion="aic")
        assert p >= 0 and d >= 0 and q >= 0

    def test_ar1_series_selects_low_order_with_no_differencing(self) -> None:
        # Order selection via CSS-based AIC/BIC is noisy on a single finite
        # sample: the criterion can occasionally prefer p=2 over the true
        # p=1 (an extra near-zero AR coefficient barely changes sigma2_hat
        # but the criterion's data-dependent penalty can still favor it), so
        # we assert the recovered order is small and undifferenced rather
        # than pinning the exact true order.
        series = _simulate_ar1(phi=0.7, n=400, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=3, max_d=1, max_q=2, criterion="bic")
        assert p >= 1
        assert d == 0
        assert q <= 1


class TestArimaResiduals:
    def test_invalid_empty_series(self) -> None:
        with pytest.raises(ValueError):
            arima_residuals(np.array([]), np.array([0.5]), np.array([]), d=0)

    def test_invalid_d_out_of_range(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5]), np.array([]), d=3)

    def test_invalid_ar_coefs_not_1d(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([[0.5]]), np.array([]), d=0)

    def test_invalid_ma_coefs_not_1d(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5]), np.array([[0.1]]), d=0)

    def test_invalid_series_too_short(self) -> None:
        series = np.arange(2, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5, 0.2, 0.1]), np.array([]), d=0)

    def test_bit_identical_to_arima_fit_internal_residuals(self) -> None:
        # arima_residuals must reproduce _arima_fit's internal residual
        # computation exactly, since both route through the same CSS kernel
        # call with the winning parameters.
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        p, d, q = 2, 0, 1

        ar_coefs, ma_coefs, _ = arima_fit(series, p=p, d=d, q=q)

        y = _difference(series, d)
        params = np.concatenate([ar_coefs, ma_coefs])
        internal_sum_sq = _css_objective(params, y, p, q)
        internal_residuals = _css_residuals_numba_kernel(params[:p], params[p : p + q], y)

        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)

        assert np.array_equal(residuals, internal_residuals)
        assert float(np.sum(residuals**2)) == internal_sum_sq

    def test_residuals_feed_ljung_box_no_rejection_on_well_specified_fit(self) -> None:
        series = _simulate_ar1(phi=0.6, n=500, seed=RNG_SEED)
        ar_coefs, ma_coefs, _ = arima_fit(series, p=1, d=0, q=0)
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d=0)
        _, p_value = ljung_box_test(residuals, n_lags=10)
        assert p_value > 0.05

    def test_output_length_matches_css_slicing_convention(self) -> None:
        series = _simulate_ar1(phi=0.5, n=200, seed=RNG_SEED)
        p, d, q = 2, 1, 1
        ar_coefs, ma_coefs, _ = arima_fit(series, p=p, d=d, q=q)
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)
        differenced = _difference(series, d)
        assert residuals.shape == (differenced.size - max(p, q),)


class TestCssRefactorRegression:
    def test_ar1_fit_matches_expected_coefficient_after_numba_refactor(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=0)
        assert ar_coefs[0] == pytest.approx(0.6, abs=0.15)
        assert ma_coefs.shape == (0,)
        assert sigma2 > 0.0

    def test_arma_fit_is_self_consistent_across_repeated_calls(self) -> None:
        series = _simulate_ar1(phi=0.4, n=250, seed=RNG_SEED + 1)
        first = arima_fit(series, p=1, d=0, q=1)
        second = arima_fit(series, p=1, d=0, q=1)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_short_series_grid_still_recovers_a_feasible_order(self) -> None:
        series = _simulate_ar1(phi=0.5, n=60, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="aic")
        assert p >= 0 and d >= 0 and q >= 0


class TestSelectArimaOrderBenchmark:
    def test_select_arima_order_completes_within_a_generous_ceiling(self) -> None:
        # Informational timing guard, not a tight assertion: warms up the
        # Numba JIT kernel first (a fresh process pays that compile cost
        # once) so the measured call only reflects steady-state performance,
        # matching how a long-lived daily refit process would behave after
        # its first asset.
        series = _simulate_ar1(phi=0.6, n=500, seed=RNG_SEED)
        select_arima_order(series[:20], max_p=1, max_d=0, max_q=1, criterion="aic")

        start = time.perf_counter()
        select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="aic")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0


class TestArimaForecast:
    def test_ar1_forecast_matches_expected_next_step(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, _ = arima_fit(series, p=1, d=0, q=0)
        forecast = arima_forecast(series, ar_coefs=ar_coefs, ma_coefs=ma_coefs, d=0, h=5)
        assert forecast.shape == (5,)
        assert np.all(np.isfinite(forecast))
        expected_first_step = ar_coefs[0] * series[-1]
        assert forecast[0] == pytest.approx(expected_first_step, abs=0.5)
