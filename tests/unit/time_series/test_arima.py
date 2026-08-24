"""Tests for Ljung-Box autocorrelation testing and ARIMA(p,d,q) fitting/forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.time_series.arima import arima_fit, arima_forecast, ljung_box_test

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


class TestArimaForecast:
    def test_ar1_forecast_matches_expected_next_step(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, _ = arima_fit(series, p=1, d=0, q=0)
        forecast = arima_forecast(series, ar_coefs=ar_coefs, ma_coefs=ma_coefs, d=0, h=5)
        assert forecast.shape == (5,)
        assert np.all(np.isfinite(forecast))
        expected_first_step = ar_coefs[0] * series[-1]
        assert forecast[0] == pytest.approx(expected_first_step, abs=0.5)
