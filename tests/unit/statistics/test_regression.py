"""Tests for OLS, Newey-West HAC covariance, factor loadings, and Fama-MacBeth regression."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore import UndefinedResultError
from quantcore.statistics.regression import (
    factor_loadings,
    fama_macbeth_regression,
    newey_west_cov,
    newey_west_optimal_lags,
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

    def test_constant_y_gives_nan_r_squared_not_one(self) -> None:
        # Regression test: a constant y is not "perfectly explained" by
        # definition — R^2 is undefined (0/0), not 1.0. This is a
        # documented value (see `ols`'s Returns docstring), not an error.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.full(5, 7.0)
        X = np.column_stack([np.ones_like(x), x])
        _, _, r_squared = ols(y, X)
        assert np.isnan(r_squared)

    def test_all_zero_y_without_intercept_gives_nan_r_squared(self) -> None:
        # Same documented NaN-on-zero-denominator behavior, in the
        # uncentered (no intercept column) case: y all-zero makes
        # sum(y^2) == 0.
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.zeros(3)
        _, _, r_squared = ols(y, x)
        assert np.isnan(r_squared)

    @pytest.mark.parametrize("value", [0.0, 0.01, 0.002, 3.7, -146.7])
    def test_zero_range_y_with_intercept_gives_nan_r_squared(self, value: float) -> None:
        # 0.01 and 0.002 leave rounding residue after demeaning; the range
        # test still sees them as constant.
        x = np.linspace(-1.0, 2.0, 50)
        X = np.column_stack([np.ones_like(x), x])
        coefficients, _, r_squared = ols(np.full(50, value), X)
        assert np.isnan(r_squared)
        assert coefficients[0] == pytest.approx(value, abs=1e-12)

    def test_constant_non_zero_y_without_intercept_is_defined(self) -> None:
        # Uncentered R^2 measures fit against the origin: a non-zero
        # constant has a positive denominator.
        x = np.column_stack([np.linspace(1.0, 2.0, 20), np.linspace(-1.0, 3.0, 20) ** 2])
        _, _, r_squared = ols(np.full(20, 0.01), x)
        assert np.isfinite(r_squared) and 0.0 <= r_squared <= 1.0

    def test_one_value_off_a_constant_y_is_defined(self) -> None:
        x = np.linspace(-1.0, 2.0, 50)
        X = np.column_stack([np.ones_like(x), x])
        y = np.full(50, 0.01)
        y[10] = 0.011
        _, _, r_squared = ols(y, X)
        assert np.isfinite(r_squared)

    def test_uncentered_r_squared_without_intercept_column(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        x = rng.normal(loc=5.0, size=(100, 2))  # no constant column, nonzero mean
        y = x @ np.array([1.5, -0.5]) + rng.normal(scale=0.1, size=100)
        _, residuals, r_squared = ols(y, x)
        expected = 1.0 - np.sum(residuals**2) / np.sum(y**2)
        assert r_squared == pytest.approx(expected, rel=1e-9)
        # The centered formula gives a numerically different value (both
        # are close to 1 here since the fit is good, but the two
        # denominators, sum(y^2) vs. sum((y-mean(y))^2), are not equal)
        # -- confirms the two conventions really differ.
        centered = 1.0 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
        assert r_squared != pytest.approx(centered, rel=1e-6)


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

    def test_default_matches_no_correction(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 200
        x = rng.normal(size=T)
        X = np.column_stack([np.ones(T), x])
        residuals = rng.normal(scale=0.1, size=T)
        default = newey_west_cov(X, residuals, n_lags=4)
        explicit = newey_west_cov(X, residuals, n_lags=4, small_sample_correction=False)
        np.testing.assert_allclose(default, explicit)

    def test_small_sample_correction_scales_by_t_over_t_minus_k(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 200
        x = rng.normal(size=T)
        X = np.column_stack([np.ones(T), x])
        residuals = rng.normal(scale=0.1, size=T)
        uncorrected = newey_west_cov(X, residuals, n_lags=4)
        corrected = newey_west_cov(X, residuals, n_lags=4, small_sample_correction=True)
        expected_factor = T / (T - X.shape[1])
        np.testing.assert_allclose(corrected, uncorrected * expected_factor)


class TestNeweyWestOptimalLags:
    def test_invalid_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            newey_west_optimal_lags(np.array([]))

    @pytest.mark.parametrize(
        ("n", "expected_lags"),
        [(100, 4), (200, 4), (50, 3), (1000, 6), (1, 1)],
    )
    def test_matches_closed_form_plug_in_rule(self, n: int, expected_lags: int) -> None:
        residuals = np.zeros(n)
        assert newey_west_optimal_lags(residuals) == expected_lags


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

        result = factor_loadings(returns, factors)
        assert result.loadings.shape == (3,)
        assert result.t_stats.shape == (3,)
        assert result.standard_errors.shape == (3,)
        assert result.p_values.shape == (3,)
        assert result.residuals.shape == (T,)
        assert result.loadings[0] == pytest.approx(0.0, abs=1e-3)
        assert result.loadings[1] == pytest.approx(true_betas[0], abs=1e-2)
        assert result.loadings[2] == pytest.approx(true_betas[1], abs=1e-2)
        assert result.r_squared > 0.99
        # Near-exact fit (tiny noise): loadings are highly significant,
        # alpha is not (it's ~0 by construction).
        assert result.p_values[1] < 0.01
        assert result.p_values[2] < 0.01
        assert result.p_values[0] > 0.05

    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    def test_zero_range_returns_give_nan_r_squared_t_stats_and_p_values(self, value: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        factors = rng.normal(scale=0.01, size=(120, 2))
        result = factor_loadings(np.full(120, value), factors)
        assert np.isnan(result.r_squared)
        assert np.all(np.isnan(result.t_stats))
        assert np.all(np.isnan(result.p_values))
        assert result.loadings[0] == pytest.approx(value, abs=1e-12)
        np.testing.assert_allclose(result.loadings[1:], 0.0, atol=1e-10)
        assert np.all(np.isfinite(result.standard_errors))

    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    def test_zero_range_factor_column_raises_naming_it(self, value: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        factors = rng.normal(scale=0.01, size=(120, 3))
        factors[:, 1] = value
        returns = rng.normal(scale=0.01, size=120)
        with pytest.raises(UndefinedResultError, match=r"factor column\(s\) \[1\]") as excinfo:
            factor_loadings(returns, factors)
        assert isinstance(excinfo.value, ValueError)

    def test_zero_range_one_dimensional_factor_raises(self) -> None:
        returns = np.random.default_rng(RNG_SEED).normal(size=50)
        with pytest.raises(UndefinedResultError, match=r"\[0\]"):
            factor_loadings(returns, np.full(50, 0.01))

    def test_default_n_lags_nw_matches_newey_west_optimal_lags(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 300
        factors = rng.normal(scale=0.01, size=(T, 2))
        returns = factors @ np.array([1.0, -0.3]) + rng.normal(scale=0.01, size=T)
        result = factor_loadings(returns, factors)
        # newey_west_optimal_lags is a function of sample size alone, so
        # this is exact regardless of the actual residuals.
        assert result.n_lags_nw == newey_west_optimal_lags(np.zeros(T))

    def test_explicit_n_lags_nw_is_respected(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        T = 300
        factors = rng.normal(scale=0.01, size=(T, 2))
        returns = factors @ np.array([1.0, -0.3]) + rng.normal(scale=0.01, size=T)
        result = factor_loadings(returns, factors, n_lags_nw=2)
        assert result.n_lags_nw == 2


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

    def test_invalid_negative_n_lags_nw_raises(self) -> None:
        returns = np.zeros((20, 3))
        betas = np.ones((3, 2))
        with pytest.raises(ValueError):
            fama_macbeth_regression(returns, betas, n_lags_nw=-1)

    def test_n_lags_nw_leaves_mean_risk_premia_unchanged(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_periods, n_assets, n_factors = 100, 15, 2
        betas = rng.normal(scale=1.0, size=(n_assets, n_factors))
        returns = rng.normal(scale=0.01, size=(n_periods, n_assets))

        mean_plain, _ = fama_macbeth_regression(returns, betas)
        mean_nw, _ = fama_macbeth_regression(returns, betas, n_lags_nw=3)
        np.testing.assert_allclose(mean_plain, mean_nw)

    def test_n_lags_nw_zero_matches_plain_for_uncorrelated_premia(self) -> None:
        # With n_lags_nw=0, Newey-West reduces to the plain (heteroskedasticity-
        # only, no autocorrelation terms) variance of the mean, which for
        # i.i.d.-ish lambda_t should be close to the plain ddof=1 time-series SE.
        rng = np.random.default_rng(RNG_SEED)
        n_periods, n_assets, n_factors = 300, 20, 2
        betas = rng.normal(scale=1.0, size=(n_assets, n_factors))
        returns = rng.normal(scale=0.01, size=(n_periods, n_assets))

        _, t_plain = fama_macbeth_regression(returns, betas)
        _, t_nw0 = fama_macbeth_regression(returns, betas, n_lags_nw=0)
        np.testing.assert_allclose(t_plain, t_nw0, rtol=0.05)

    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    def test_all_zero_beta_column_raises_naming_it(self, value: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        betas = rng.normal(size=(15, 3))
        betas[:, 2] = 0.0
        returns = np.full((40, 15), value) + rng.normal(scale=0.01, size=(40, 15))
        with pytest.raises(UndefinedResultError, match=r"beta column\(s\) \[2\]") as excinfo:
            fama_macbeth_regression(returns, betas)
        assert isinstance(excinfo.value, ValueError)

    @pytest.mark.parametrize("n_lags_nw", [None, 2])
    @pytest.mark.parametrize("value", [0.0, 0.01, 3.7])
    def test_constant_returns_give_nan_t_stats(self, value: float, n_lags_nw: int | None) -> None:
        # Every period has the same cross-section, so lambda_t is the same
        # in every period: the mean is defined, its t-statistic is not.
        betas = np.random.default_rng(RNG_SEED).normal(size=(15, 2))
        mean_risk_premia, t_stats = fama_macbeth_regression(
            np.full((40, 15), value), betas, n_lags_nw=n_lags_nw
        )
        assert np.all(np.isfinite(mean_risk_premia))
        assert np.all(np.isnan(t_stats))

    def test_repeated_cross_section_gives_nan_t_stats(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        betas = rng.normal(size=(15, 2))
        returns = np.tile(rng.normal(scale=0.01, size=15), (30, 1))
        mean_risk_premia, t_stats = fama_macbeth_regression(returns, betas)
        assert np.all(np.isfinite(mean_risk_premia))
        assert np.all(np.isnan(t_stats))

    def test_constant_cross_section_each_period_is_defined(self) -> None:
        # A constant non-zero beta column acts as the intercept; a period
        # whose returns are equal across assets loads only on it.
        rng = np.random.default_rng(RNG_SEED)
        betas = np.column_stack([np.ones(15), rng.normal(size=15)])
        returns = rng.normal(scale=0.01, size=(40, 1)) * np.ones((1, 15))
        returns += rng.normal(scale=0.001, size=(40, 15))
        mean_risk_premia, t_stats = fama_macbeth_regression(returns, betas)
        assert np.all(np.isfinite(mean_risk_premia)) and np.all(np.isfinite(t_stats))
