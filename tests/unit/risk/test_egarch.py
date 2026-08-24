"""Tests for EGARCH(1,1), GJR-GARCH(1,1), and EWMA conditional variance."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.egarch import egarch_11_variance, ewma_variance, gjr_garch_11_variance
from quantcore.risk.volatility import garch_11_variance

RETURNS = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012])


class TestEgarch11Variance:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            egarch_11_variance(np.array([]), omega=0.01, alpha=0.1, gamma=-0.05, beta=0.9)

    @pytest.mark.parametrize("beta", [1.0, -1.0, 1.5, -1.5])
    def test_invalid_beta_outside_unit_interval_raises(self, beta: float) -> None:
        with pytest.raises(ValueError):
            egarch_11_variance(RETURNS, omega=0.01, alpha=0.1, gamma=-0.05, beta=beta)

    def test_output_same_length_as_input(self) -> None:
        result = egarch_11_variance(RETURNS, omega=0.01, alpha=0.1, gamma=-0.05, beta=0.9)
        assert result.shape == RETURNS.shape

    def test_output_finite_for_valid_parameters(self) -> None:
        result = egarch_11_variance(RETURNS, omega=0.01, alpha=0.1, gamma=-0.05, beta=0.9)
        assert np.all(np.isfinite(result))

    def test_first_value_seeded_at_stationary_log_variance(self) -> None:
        omega, beta = 0.02, 0.9
        result = egarch_11_variance(RETURNS, omega=omega, alpha=0.1, gamma=-0.05, beta=beta)
        assert result[0] == pytest.approx(omega / (1.0 - beta), abs=1e-9)


class TestGjrGarch11Variance:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_11_variance(np.array([]), omega=0.01, alpha=0.05, gamma=0.1, beta=0.85)

    def test_invalid_omega_non_positive_raises(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_11_variance(RETURNS, omega=0.0, alpha=0.05, gamma=0.1, beta=0.85)

    def test_invalid_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_11_variance(RETURNS, omega=0.01, alpha=-0.05, gamma=0.1, beta=0.85)

    def test_invalid_alpha_plus_gamma_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_11_variance(RETURNS, omega=0.01, alpha=0.05, gamma=-0.2, beta=0.85)

    def test_invalid_non_stationary_raises(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_11_variance(RETURNS, omega=0.01, alpha=0.5, gamma=0.3, beta=0.85)

    def test_output_same_length_as_input(self) -> None:
        result = gjr_garch_11_variance(RETURNS, omega=0.01, alpha=0.05, gamma=0.1, beta=0.8)
        assert result.shape == RETURNS.shape

    def test_output_positive_for_valid_parameters(self) -> None:
        result = gjr_garch_11_variance(RETURNS, omega=0.01, alpha=0.05, gamma=0.1, beta=0.8)
        assert np.all(result > 0.0)

    def test_zero_gamma_matches_standard_garch_11(self) -> None:
        omega, alpha, beta = 0.01, 0.05, 0.85
        gjr = gjr_garch_11_variance(RETURNS, omega=omega, alpha=alpha, gamma=0.0, beta=beta)
        garch = garch_11_variance(RETURNS, omega=omega, alpha=alpha, beta=beta)
        assert np.max(np.abs(gjr - garch)) < 1e-10


class TestEwmaVariance:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ewma_variance(np.array([]), lambda_=0.94)

    @pytest.mark.parametrize("lambda_", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_lambda_outside_open_unit_interval_raises(self, lambda_: float) -> None:
        with pytest.raises(ValueError):
            ewma_variance(RETURNS, lambda_=lambda_)

    def test_output_same_length_as_input(self) -> None:
        result = ewma_variance(RETURNS, lambda_=0.94)
        assert result.shape == RETURNS.shape

    def test_first_value_equals_first_squared_return(self) -> None:
        result = ewma_variance(RETURNS, lambda_=0.94)
        assert result[0] == pytest.approx(RETURNS[0] ** 2, abs=1e-12)

    def test_recursion_matches_manual_computation(self) -> None:
        lambda_ = 0.9
        result = ewma_variance(RETURNS, lambda_=lambda_)
        expected = np.empty_like(RETURNS)
        expected[0] = RETURNS[0] ** 2
        for t in range(1, len(RETURNS)):
            expected[t] = lambda_ * expected[t - 1] + (1.0 - lambda_) * RETURNS[t - 1] ** 2
        assert np.max(np.abs(result - expected)) < 1e-10

    def test_variance_decays_toward_zero_after_large_shock_then_calm_returns(self) -> None:
        returns = np.array([0.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        result = ewma_variance(returns, lambda_=0.9)
        assert result[1] > result[-1]
