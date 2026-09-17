"""Tests for EGARCH(1,1), GJR-GARCH(1,1), and EWMA conditional variance.

The `egarch_fit` / `gjr_garch_fit` sections below follow the same house
convention as `tests/unit/risk/test_fit_garch_11.py`: since we cannot run
these optimizers ourselves ahead of time to hand-calibrate exact expected
numbers, the tests are structured around invariants that must hold for ANY
correct MLE implementation of these models (feasibility of output,
determinism, recovering known parameters from synthetic data within loose
but principled tolerances, being closer to the truth than a deliberately bad
guess) rather than pinned magic-number assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.egarch import (
    EGARCHParams,
    GJRGARCHParams,
    egarch_11_variance,
    egarch_fit,
    ewma_variance,
    gjr_garch_11_variance,
    gjr_garch_fit,
)
from quantcore.risk.volatility import garch_11_variance

RETURNS = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012])
RNG_SEED = 2024


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


# ---------------------------------------------------------------------------
# Synthetic-data generators for the MLE fitters, mirroring
# `test_fit_garch_11.py::_simulate_garch_11_process`: each new residual
# depends on the just-computed conditional variance, and the next variance
# depends on the just-drawn residual, so these co-generate both series by
# hand rather than reusing the public `*_variance` functions (which consume
# an already-realized residual series).
# ---------------------------------------------------------------------------


def _simulate_egarch_11_process(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int,
) -> np.ndarray:
    """Simulate a synthetic EGARCH(1,1) residual series from known parameters."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)

    e_abs_z = np.sqrt(2.0 / np.pi)
    log_variance = np.empty(n, dtype=np.float64)
    epsilon = np.empty(n, dtype=np.float64)

    log_variance[0] = omega / (1.0 - beta)
    epsilon[0] = np.sqrt(np.exp(log_variance[0])) * z[0]

    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_variance[t - 1]))
        z_prev = epsilon[t - 1] / sigma_prev
        log_variance[t] = (
            omega + beta * log_variance[t - 1] + alpha * (np.abs(z_prev) - e_abs_z) + gamma * z_prev
        )
        epsilon[t] = np.sqrt(np.exp(log_variance[t])) * z[t]

    return epsilon


def _simulate_gjr_garch_11_process(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int,
) -> np.ndarray:
    """Simulate a synthetic GJR-GARCH(1,1) residual series from known parameters."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)

    variance = np.empty(n, dtype=np.float64)
    epsilon = np.empty(n, dtype=np.float64)

    variance[0] = omega / (1.0 - alpha - gamma / 2.0 - beta)
    epsilon[0] = np.sqrt(variance[0]) * z[0]

    for t in range(1, n):
        bad_news = 1.0 if epsilon[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * epsilon[t - 1] ** 2
            + gamma * epsilon[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )
        epsilon[t] = np.sqrt(variance[t]) * z[t]

    return epsilon


class TestEgarchFit:
    # ---- Shape / type contract. ----

    def test_returns_egarch_params_with_float_fields(self) -> None:
        epsilon = _simulate_egarch_11_process(500, 0.0, 0.1, -0.05, 0.9, seed=RNG_SEED)
        result = egarch_fit(epsilon)

        assert isinstance(result, EGARCHParams)
        for field in (result.omega, result.alpha, result.gamma, result.beta):
            assert isinstance(field, float)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.converged, bool)

    # ---- Constraint satisfaction across several synthetic series. ----

    @pytest.mark.parametrize(
        "omega_true, alpha_true, gamma_true, beta_true, seed",
        [
            (0.0, 0.10, -0.05, 0.90, 1),
            (-0.05, 0.05, -0.10, 0.85, 2),
            (0.02, 0.15, 0.00, 0.70, 3),
            (0.01, 0.20, -0.05, 0.60, 4),
        ],
    )
    def test_output_satisfies_stationarity_constraint(
        self, omega_true: float, alpha_true: float, gamma_true: float, beta_true: float, seed: int
    ) -> None:
        epsilon = _simulate_egarch_11_process(
            800, omega_true, alpha_true, gamma_true, beta_true, seed=seed
        )
        result = egarch_fit(epsilon)

        assert -1.0 < result.beta < 1.0

    # ---- Core correctness: recover known parameters on synthetic data. ----

    def test_recovers_known_parameters_on_synthetic_data(self) -> None:
        omega_true, alpha_true, gamma_true, beta_true = 0.0, 0.1, -0.05, 0.9
        epsilon = _simulate_egarch_11_process(
            5000, omega_true, alpha_true, gamma_true, beta_true, seed=RNG_SEED
        )

        result = egarch_fit(epsilon)

        # Loose tolerance for the same reason as `fit_garch_11`'s equivalent
        # test: finite-sample GARCH-family MLE is noisy, and we cannot run
        # the optimizer ourselves ahead of time to calibrate a tighter bound.
        assert result.beta == pytest.approx(beta_true, rel=0.3, abs=0.1)
        assert -1.0 < result.beta < 1.0

        # Closer to the truth than a deliberately poor, arbitrary guess.
        bad_guess_beta = 0.2
        assert abs(result.beta - beta_true) < abs(bad_guess_beta - beta_true)

    # ---- Degenerate input validation. ----

    def test_empty_returns_raises(self) -> None:
        with pytest.raises(ValueError):
            egarch_fit(np.array([]))

    # ---- Determinism. ----

    def test_deterministic_given_fixed_input(self) -> None:
        epsilon = _simulate_egarch_11_process(400, 0.0, 0.1, -0.05, 0.9, seed=RNG_SEED)

        result_1 = egarch_fit(epsilon)
        result_2 = egarch_fit(epsilon)

        assert result_1 == result_2

    # ---- `converged` diagnostic behavior. ----

    def test_converged_flag_is_consistent_with_feasibility(self) -> None:
        # Whatever the optimizer reports, a converged result must, by
        # definition, be feasible -- the fitter should never claim
        # convergence while returning a beta outside the stationarity
        # region.
        epsilon = _simulate_egarch_11_process(600, 0.0, 0.1, -0.05, 0.9, seed=RNG_SEED)
        result = egarch_fit(epsilon)

        if result.converged:
            assert -1.0 < result.beta < 1.0


class TestGjrGarchFit:
    # ---- Shape / type contract. ----

    def test_returns_gjr_garch_params_with_float_fields(self) -> None:
        epsilon = _simulate_gjr_garch_11_process(500, 1e-5, 0.05, 0.10, 0.80, seed=RNG_SEED)
        result = gjr_garch_fit(epsilon)

        assert isinstance(result, GJRGARCHParams)
        for field in (result.omega, result.alpha, result.gamma, result.beta):
            assert isinstance(field, float)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.converged, bool)

    # ---- Constraint satisfaction across several synthetic series. ----

    @pytest.mark.parametrize(
        "omega_true, alpha_true, gamma_true, beta_true, seed",
        [
            (1e-5, 0.05, 0.10, 0.80, 1),
            (5e-6, 0.02, 0.15, 0.75, 2),
            (2e-5, 0.10, -0.05, 0.70, 3),
            (1e-4, 0.05, 0.05, 0.85, 4),
        ],
    )
    def test_output_satisfies_stationarity_constraints(
        self, omega_true: float, alpha_true: float, gamma_true: float, beta_true: float, seed: int
    ) -> None:
        epsilon = _simulate_gjr_garch_11_process(
            800, omega_true, alpha_true, gamma_true, beta_true, seed=seed
        )
        result = gjr_garch_fit(epsilon)

        assert result.omega > 0.0
        assert result.alpha >= 0.0
        assert result.alpha + result.gamma >= 0.0
        assert result.alpha + result.gamma / 2.0 + result.beta < 1.0

    # ---- Core correctness: recover known parameters on synthetic data. ----

    def test_recovers_known_parameters_on_synthetic_data(self) -> None:
        omega_true, alpha_true, gamma_true, beta_true = 1e-5, 0.05, 0.10, 0.80
        epsilon = _simulate_gjr_garch_11_process(
            5000, omega_true, alpha_true, gamma_true, beta_true, seed=RNG_SEED
        )

        result = gjr_garch_fit(epsilon)

        # Loose tolerance for the same reason as `fit_garch_11`'s equivalent
        # test: finite-sample GARCH-family MLE is noisy, and we cannot run
        # the optimizer ourselves ahead of time to calibrate a tighter bound.
        assert result.omega == pytest.approx(omega_true, rel=0.75, abs=1e-5)
        assert result.alpha == pytest.approx(alpha_true, rel=0.5, abs=0.05)
        assert result.gamma == pytest.approx(gamma_true, rel=0.75, abs=0.1)
        assert result.beta == pytest.approx(beta_true, rel=0.3, abs=0.1)

        # Broad plausibility range -- catches gross errors (wrong sign,
        # wrong scale, constraint violation) independent of the exact
        # tolerance tuning above.
        assert 0.0 < result.omega < 1e-3
        assert 0.0 <= result.alpha < 1.0
        assert -1.0 < result.gamma < 1.0
        assert -1.0 < result.beta < 1.0

        # Closer to the truth than a deliberately poor, arbitrary guess.
        bad_guess = (1e-3, 0.5, -0.5, 0.2)
        assert abs(result.omega - omega_true) < abs(bad_guess[0] - omega_true)
        assert abs(result.alpha - alpha_true) < abs(bad_guess[1] - alpha_true)
        assert abs(result.gamma - gamma_true) < abs(bad_guess[2] - gamma_true)
        assert abs(result.beta - beta_true) < abs(bad_guess[3] - beta_true)

    # ---- Degenerate input validation. ----

    def test_empty_returns_raises(self) -> None:
        with pytest.raises(ValueError):
            gjr_garch_fit(np.array([]))

    # ---- Determinism. ----

    def test_deterministic_given_fixed_input(self) -> None:
        epsilon = _simulate_gjr_garch_11_process(400, 1e-5, 0.05, 0.10, 0.80, seed=RNG_SEED)

        result_1 = gjr_garch_fit(epsilon)
        result_2 = gjr_garch_fit(epsilon)

        assert result_1 == result_2

    # ---- `converged` diagnostic behavior. ----

    def test_converged_flag_is_consistent_with_feasibility(self) -> None:
        # Whatever the optimizer reports, a converged result must, by
        # definition, be feasible -- the fitter should never claim
        # convergence while returning parameters that violate the
        # constraints `gjr_garch_11_variance` itself enforces.
        epsilon = _simulate_gjr_garch_11_process(600, 1e-5, 0.05, 0.10, 0.80, seed=RNG_SEED)
        result = gjr_garch_fit(epsilon)

        if result.converged:
            assert result.omega > 0.0
            assert result.alpha >= 0.0
            assert result.alpha + result.gamma >= 0.0
            assert result.alpha + result.gamma / 2.0 + result.beta < 1.0
