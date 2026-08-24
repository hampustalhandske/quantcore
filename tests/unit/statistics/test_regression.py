"""Tests for OLS, Newey-West HAC covariance, factor loadings, and Fama-MacBeth regression."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.statistics.regression import (
    factor_loadings,
    fama_macbeth_regression,
    newey_west_cov,
    ols,
)

RNG_SEED = 7


class TestOls:
    def test_invalid_shape_mismatch_raises(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            ols(y, X)

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            ols(np.array([]), np.empty((0, 2)))

    def test_invalid_underdetermined_raises(self) -> None:
        y = np.array([1.0, 2.0])
        X = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        with pytest.raises(ValueError):
            ols(y, X)

    def test_recovers_known_two_variable_solution(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 + 3.0 * x
        X = np.column_stack([np.ones_like(x), x])
        coefficients, residuals, r_squared = ols(y, X)
        assert coefficients[0] == pytest.approx(2.0, abs=1e-8)
        assert coefficients[1] == pytest.approx(3.0, abs=1e-8)
        assert np.max(np.abs(residuals)) < 1e-8
        assert r_squared == pytest.approx(1.0, abs=1e-8)


class TestNeweyWestCov:
    def test_invalid_shape_mismatch_raises(self) -> None:
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        residuals = np.array([0.1, -0.1])
        with pytest.raises(ValueError):
            newey_west_cov(X, residuals, n_lags=1)

    def test_invalid_negative_lags_raises(self) -> None:
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        residuals = np.array([0.1, -0.1, 0.05])
        with pytest.raises(ValueError):
            newey_west_cov(X, residuals, n_lags=-1)

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            newey_west_cov(np.empty((0, 2)), np.array([]), n_lags=1)

    def test_result_is_symmetric_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 200
        x = rng.normal(size=T)
        X = np.column_stack([np.ones(T), x])
        residuals = rng.normal(scale=0.1, size=T)
        cov = newey_west_cov(X, residuals, n_lags=4)
        assert np.max(np.abs(cov - cov.T)) < 1e-8
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-8)
        assert np.all(np.diag(cov) >= 0.0)


class TestFactorLoadings:
    def test_invalid_shape_mismatch_raises(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        factors = np.array([[0.01, 0.02], [0.02, 0.01]])
        with pytest.raises(ValueError):
            factor_loadings(returns, factors)

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            factor_loadings(np.array([]), np.empty((0, 2)))

    def test_alpha_close_to_zero_when_returns_generated_from_factors(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 500
        true_betas = np.array([1.2, -0.5])
        factors = rng.normal(scale=0.01, size=(T, 2))
        returns = factors @ true_betas + rng.normal(scale=1e-5, size=T)

        loadings, t_stats = factor_loadings(returns, factors)
        assert loadings.shape == (3,)
        assert t_stats.shape == (3,)
        assert loadings[0] == pytest.approx(0.0, abs=1e-3)
        assert loadings[1] == pytest.approx(true_betas[0], abs=1e-2)
        assert loadings[2] == pytest.approx(true_betas[1], abs=1e-2)


class TestFamaMacbethRegression:
    def test_invalid_asset_count_mismatch_raises(self) -> None:
        returns = np.zeros((10, 3))
        betas = np.zeros((4, 2))
        with pytest.raises(ValueError):
            fama_macbeth_regression(returns, betas)

    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            fama_macbeth_regression(np.empty((0, 3)), np.zeros((3, 2)))

    def test_recovers_known_risk_premia(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_periods, n_assets, n_factors = 300, 20, 2
        true_lambda = np.array([0.02, -0.01])
        betas = rng.normal(scale=1.0, size=(n_assets, n_factors))

        returns = np.empty((n_periods, n_assets))
        for t in range(n_periods):
            lambda_t = true_lambda + rng.normal(scale=0.005, size=n_factors)
            returns[t, :] = betas @ lambda_t + rng.normal(scale=0.001, size=n_assets)

        mean_risk_premia, t_stats = fama_macbeth_regression(returns, betas)
        assert mean_risk_premia.shape == (n_factors,)
        assert t_stats.shape == (n_factors,)
        assert mean_risk_premia[0] == pytest.approx(true_lambda[0], abs=5e-3)
        assert mean_risk_premia[1] == pytest.approx(true_lambda[1], abs=5e-3)
