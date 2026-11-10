"""Tests for VAR(p) estimation, forecasting, impulse response, and Granger causality."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore import UndefinedResultError
from quantcore.time_series.var_model import (
    granger_causality_test,
    impulse_response,
    select_var_lag_order,
    var_fit,
    var_forecast,
)

RNG_SEED = 7


def _simulate_var1(a1: np.ndarray, n: int, seed: int) -> np.ndarray:
    k = a1.shape[0]
    rng = np.random.default_rng(seed)
    data = np.zeros((n, k), dtype=np.float64)
    for t in range(1, n):
        data[t] = a1 @ data[t - 1] + rng.normal(0.0, 0.1, size=k)
    return data


class TestVarFit:
    def test_invalid_n_lags(self) -> None:
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 50, RNG_SEED)
        with pytest.raises(ValueError):
            var_fit(data, n_lags=0)

    def test_invalid_too_few_rows(self) -> None:
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 3, RNG_SEED)
        with pytest.raises(ValueError):
            var_fit(data, n_lags=2)

    def test_recovers_known_var1_coefficients(self) -> None:
        a1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        data = _simulate_var1(a1, 500, RNG_SEED)
        coef_matrix, sigma_u = var_fit(data, n_lags=1)
        assert coef_matrix.shape == (2, 3)
        assert sigma_u.shape == (2, 2)
        assert np.max(np.abs(coef_matrix[:, :2] - a1)) < 0.1

    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    @pytest.mark.parametrize("column", [0, 1])
    def test_zero_range_column_raises_naming_it(self, value: float, column: int) -> None:
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 80, RNG_SEED)
        data[:, column] = value
        with pytest.raises(UndefinedResultError, match=rf"column\(s\) \[{column}\]") as raised:
            var_fit(data, n_lags=1)
        assert isinstance(raised.value, ValueError)

    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    def test_all_constant_system_raises(self, value: float) -> None:
        with pytest.raises(UndefinedResultError, match=r"\[0, 1, 2\]"):
            var_fit(np.full((60, 3), value), n_lags=1)


class TestSelectVarLagOrder:
    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    @pytest.mark.parametrize("column", [0, 1])
    def test_zero_range_column_raises_naming_it(self, value: float, column: int) -> None:
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 80, RNG_SEED)
        data[:, column] = value
        with pytest.raises(UndefinedResultError, match=rf"column\(s\) \[{column}\]") as raised:
            select_var_lag_order(data, max_lags=3)
        assert isinstance(raised.value, ValueError)

    def test_invalid_criterion_raises(self) -> None:
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 100, RNG_SEED)
        with pytest.raises(ValueError):
            select_var_lag_order(data, max_lags=3, criterion="hqic")

    def test_a_single_series_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 series"):
            var_fit(np.arange(40.0).reshape(-1, 1), 1)
        with pytest.raises(ValueError, match="at least 2 series"):
            select_var_lag_order(np.arange(40.0).reshape(-1, 1), 3)

    def test_skips_lag_orders_that_are_not_estimable_on_the_common_window(self) -> None:
        # Every candidate is fit on the same trailing window of
        # n_obs - max_lags observations; with 10 observations of 2 series
        # only n_lags=1 leaves spare degrees of freedom for max_lags=5.
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 10, RNG_SEED)
        assert select_var_lag_order(data, max_lags=5, criterion="aic") == 1

    def test_raises_when_max_lags_leaves_no_feasible_candidate(self) -> None:
        # With too few observations for even n_lags=1 to leave a usable
        # fixed window (n_obs - max_lags), no candidate is feasible.
        data = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 8, RNG_SEED)
        with pytest.raises(ValueError, match="no feasible n_lags"):
            select_var_lag_order(data, max_lags=5, criterion="aic")

    def test_var1_series_selects_low_lag_order(self) -> None:
        # Lag-order selection is noisy in finite samples; assert the
        # recovered lag order is small (consistent with the true VAR(1)
        # generating process) rather than pinning it to exactly 1.
        a1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        data = _simulate_var1(a1, 400, RNG_SEED)
        n_lags = select_var_lag_order(data, max_lags=5, criterion="bic")
        assert 1 <= n_lags <= 2


class TestVarForecast:
    def _fit(self) -> tuple[np.ndarray, np.ndarray]:
        a1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        data = _simulate_var1(a1, 200, RNG_SEED)
        coef_matrix, _ = var_fit(data, n_lags=1)
        return data, coef_matrix

    def test_invalid_n_lags(self) -> None:
        data, coef_matrix = self._fit()
        with pytest.raises(ValueError):
            var_forecast(data, coef_matrix, n_lags=0, h=5)

    def test_invalid_horizon(self) -> None:
        data, coef_matrix = self._fit()
        with pytest.raises(ValueError):
            var_forecast(data, coef_matrix, n_lags=1, h=0)

    def test_invalid_coef_matrix_shape(self) -> None:
        data, _ = self._fit()
        bad_coef = np.zeros((2, 5))
        with pytest.raises(ValueError):
            var_forecast(data, bad_coef, n_lags=1, h=5)

    def test_output_shape(self) -> None:
        data, coef_matrix = self._fit()
        forecast = var_forecast(data, coef_matrix, n_lags=1, h=4)
        assert forecast.shape == (4, 2)


class TestImpulseResponse:
    def _setup(self) -> tuple[np.ndarray, np.ndarray]:
        coef_matrix = np.array([[0.5, 0.1, 0.0], [0.0, 0.3, 0.0]])
        rng = np.random.default_rng(RNG_SEED)
        m = rng.normal(size=(2, 2))
        sigma_u = m @ m.T + np.eye(2) * 0.1
        return coef_matrix, sigma_u

    def test_invalid_n_lags(self) -> None:
        coef_matrix, sigma_u = self._setup()
        with pytest.raises(ValueError):
            impulse_response(coef_matrix, sigma_u, n_lags=0, n_periods=5)

    def test_invalid_n_periods(self) -> None:
        coef_matrix, sigma_u = self._setup()
        with pytest.raises(ValueError):
            impulse_response(coef_matrix, sigma_u, n_lags=1, n_periods=0)

    def test_invalid_shape_mismatch(self) -> None:
        coef_matrix, _sigma_u = self._setup()
        bad_sigma = np.eye(3)
        with pytest.raises(ValueError):
            impulse_response(coef_matrix, bad_sigma, n_lags=1, n_periods=5)

    def test_horizon_zero_equals_cholesky_of_sigma_u(self) -> None:
        coef_matrix, sigma_u = self._setup()
        irf = impulse_response(coef_matrix, sigma_u, n_lags=1, n_periods=6)
        expected = np.linalg.cholesky(sigma_u)
        assert irf.shape == (6, 2, 2)
        assert np.max(np.abs(irf[0] - expected)) < 1e-9


class TestGrangerCausalityTest:
    def test_invalid_n_lags(self) -> None:
        y = np.zeros(50)
        x = np.zeros(50)
        with pytest.raises(ValueError):
            granger_causality_test(y, x, n_lags=0)

    def test_invalid_mismatched_lengths(self) -> None:
        y = np.zeros(50)
        x = np.zeros(40)
        with pytest.raises(ValueError):
            granger_causality_test(y, x, n_lags=1)

    def test_invalid_too_few_observations(self) -> None:
        y = np.zeros(3)
        x = np.zeros(3)
        with pytest.raises(ValueError):
            granger_causality_test(y, x, n_lags=2)

    def test_rejects_null_when_x_truly_causes_y(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 500
        x = rng.normal(0.0, 1.0, size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + rng.normal(0.0, 0.1)
        _, p_value = granger_causality_test(y, x, n_lags=1)
        assert p_value < 0.05

    def test_fails_to_reject_null_for_independent_series(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 500
        x = rng.normal(0.0, 1.0, size=n)
        y = rng.normal(0.0, 1.0, size=n)
        _, p_value = granger_causality_test(y, x, n_lags=1)
        assert p_value > 0.05


class TestStatsmodelsConventions:
    """The statsmodels calls receive every option explicitly."""

    def test_var_fit_uses_ols_a_constant_and_exactly_n_lags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}
        from statsmodels.tsa.vector_ar.var_model import VAR

        class Spy(VAR):  # type: ignore[valid-type, misc]
            def fit(self, **kwargs: object):  # type: ignore[no-untyped-def]
                seen.update(kwargs)
                return super().fit(**kwargs)

        monkeypatch.setattr("statsmodels.tsa.vector_ar.var_model.VAR", Spy)
        var_fit(_simulate_var1(np.array([[0.5, 0.0], [0.0, 0.3]]), 80, RNG_SEED), 2)
        assert seen == {"maxlags": 2, "method": "ols", "ic": None, "trend": "c", "verbose": False}

    def test_granger_tests_the_second_column_at_exactly_the_requested_lag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def spy(data: np.ndarray, **kwargs: object) -> dict[int, tuple[dict[str, tuple], None]]:
            seen.update(kwargs)
            seen["data"] = data
            return {3: ({"ssr_ftest": (2.5, 0.05, 90.0, 3)}, None)}

        monkeypatch.setattr("statsmodels.tsa.stattools.grangercausalitytests", spy)
        y, x = np.arange(20.0), np.arange(20.0)[::-1] * 2.0
        assert granger_causality_test(y, x, 3) == (2.5, 0.05)
        assert seen["maxlag"] == [3] and seen["addconst"] is True
        np.testing.assert_array_equal(seen["data"], np.column_stack([y, x]))


class TestGrangerUndefined:
    @pytest.mark.parametrize("constant", ["y", "x"])
    @pytest.mark.parametrize("level", [0.0, 0.01])
    def test_zero_range_series_raises_undefined(self, constant: str, level: float) -> None:
        varying = np.random.default_rng(3).normal(size=60)
        flat = np.full(60, level)
        y, x = (flat, varying) if constant == "y" else (varying, flat)
        with pytest.raises(UndefinedResultError, match=f"^{constant} has zero range") as raised:
            granger_causality_test(y, x, n_lags=2)
        assert isinstance(raised.value, ValueError)

    def test_lag_constant_over_the_regression_rows_raises_undefined(self) -> None:
        # x changes only at its last observation, so both lags of x are constant on rows 2..59.
        x = np.r_[np.full(59, 0.01), 0.02]
        y = np.random.default_rng(3).normal(size=60)
        with pytest.raises(UndefinedResultError, match="lag 1 of x is constant"):
            granger_causality_test(y, x, 2)

    def test_one_value_off_a_constant_is_defined(self) -> None:
        x = np.full(60, 0.01)
        x[30] = 0.02
        f_stat, p_value = granger_causality_test(np.random.default_rng(3).normal(size=60), x, 2)
        assert np.isfinite(f_stat) and 0.0 <= p_value <= 1.0
