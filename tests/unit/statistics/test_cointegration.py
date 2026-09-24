"""Tests for cointegration and mean-reversion statistics."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from quantcore.statistics.cointegration import (
    adf_test,
    engle_granger_test,
    johansen_max_eigenvalue_test,
    johansen_trace_test,
    ou_half_life,
    spread_zscore,
)

RNG_SEED = 12345


def _random_walk(n: int, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 1.0, size=n))


def _stationary_ar1(n: int, phi: float, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=n)
    series = np.empty(n)
    series[0] = noise[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    return series


class TestAdfTest:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            adf_test(np.array([]))
        with pytest.raises(ValueError):
            adf_test(np.array([1.0, 2.0, 3.0]), max_lags=5)

    def test_invalid_trend_raises(self) -> None:
        with pytest.raises(ValueError):
            adf_test(_random_walk(100), trend="bogus")

    def test_invalid_autolag_raises(self) -> None:
        with pytest.raises(ValueError):
            adf_test(_random_walk(100), max_lags=5, autolag="bogus")

    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    def test_default_autolag_none_uses_max_lags_exactly(self, trend: str) -> None:
        # autolag=None (default) must be the exact same computation as
        # before trend/autolag existed: max_lags lags, no search.
        series = _stationary_ar1(300, phi=0.5)
        stat_default, p_default = adf_test(series, max_lags=2, trend=trend)
        stat_explicit, p_explicit = adf_test(series, max_lags=2, trend=trend, autolag=None)
        assert stat_default == pytest.approx(stat_explicit)
        assert p_default == pytest.approx(p_explicit)

    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("autolag", ["aic", "bic", "t-stat"])
    def test_autolag_selects_lag_in_range_and_returns_finite_stat(
        self, trend: str, autolag: str
    ) -> None:
        series = _stationary_ar1(300, phi=0.5)
        stat, p_value = adf_test(series, max_lags=8, trend=trend, autolag=autolag)
        assert np.isfinite(stat)
        assert 0.0 <= p_value <= 1.0

    def test_random_walk_does_not_strongly_reject(self) -> None:
        series = _random_walk(500)
        _, p_value = adf_test(series, max_lags=1)
        assert p_value > 0.01

    def test_stationary_ar1_rejects_unit_root(self) -> None:
        series = _stationary_ar1(500, phi=0.5)
        _, p_value = adf_test(series, max_lags=1)
        assert p_value < 0.05

    def test_returns_tuple_of_floats(self) -> None:
        series = _stationary_ar1(200, phi=0.5)
        stat, p_value = adf_test(series)
        assert isinstance(stat, float)
        assert isinstance(p_value, float)

    @pytest.mark.parametrize(
        ("t_stat", "expected_p", "tol"),
        [
            (-2.86, 0.05, 0.02),
            (-3.43, 0.01, 0.02),
            (-2.57, 0.10, 0.02),
        ],
    )
    def test_mackinnon_pvalue_matches_known_critical_values(
        self, t_stat: float, expected_p: float, tol: float
    ) -> None:
        # Import lazily to keep this internal helper's usage localized to the
        # test that specifically validates its response-surface calibration.
        from quantcore.statistics.cointegration import _mackinnon_pvalue

        assert _mackinnon_pvalue(t_stat, n_series=1) == pytest.approx(expected_p, abs=tol)

    def test_mackinnon_pvalue_monotonically_decreases_as_stat_falls(self) -> None:
        from quantcore.statistics.cointegration import _mackinnon_pvalue

        t_stats = np.linspace(-10.0, 0.9, 200)
        p_values = [_mackinnon_pvalue(t, n_series=1) for t in t_stats]
        assert all(a <= b for a, b in pairwise(p_values))

    def test_mackinnon_pvalue_clamped_within_unit_interval_at_extremes(self) -> None:
        from quantcore.statistics.cointegration import _mackinnon_pvalue

        very_negative = _mackinnon_pvalue(-100.0, n_series=1)
        very_positive = _mackinnon_pvalue(100.0, n_series=1)
        assert 0.0 <= very_negative <= 1.0
        assert 0.0 <= very_positive <= 1.0
        assert not np.isnan(very_negative)
        assert not np.isnan(very_positive)
        assert very_negative == pytest.approx(0.0001)
        assert very_positive == pytest.approx(0.9999)

    def test_mackinnon_pvalue_rejects_n_series_outside_tabulated_range(self) -> None:
        from quantcore.statistics.cointegration import _mackinnon_pvalue

        with pytest.raises(ValueError):
            _mackinnon_pvalue(-3.0, n_series=0)
        with pytest.raises(ValueError):
            _mackinnon_pvalue(-3.0, n_series=7)

    def test_mackinnon_pvalue_shifts_left_as_n_series_grows(self) -> None:
        # The defect this guards against: reusing the N=1 (plain
        # Dickey-Fuller) distribution regardless of N understates how far
        # left the null distribution of an N-variable residual-based test
        # actually lies, and so overstates significance (a smaller p-value
        # than is warranted). See engle_granger_test's docstring.
        from quantcore.statistics.cointegration import _mackinnon_pvalue

        assert _mackinnon_pvalue(-3.0, n_series=1) == pytest.approx(0.0349, abs=1e-3)
        assert _mackinnon_pvalue(-3.0, n_series=2) == pytest.approx(0.1102, abs=1e-3)
        assert _mackinnon_pvalue(-2.86, n_series=2) == pytest.approx(0.1473, abs=1e-3)

    def test_random_walk_reports_large_pvalue_not_old_overconfident_value(self) -> None:
        # Regression test for the production bug: the prior Student-t
        # approximation reported p ~= 0.03 for genuine unit-root series,
        # falsely signaling stationarity to downstream leg-selection filters.
        for seed in range(5):
            series = _random_walk(500, seed=seed)
            _, p_value = adf_test(series, max_lags=1)
            assert p_value > 0.3

    def test_strongly_mean_reverting_series_reports_small_pvalue(self) -> None:
        series = _stationary_ar1(500, phi=0.2)
        _, p_value = adf_test(series, max_lags=1)
        assert p_value < 0.05


class TestEngleGrangerTest:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            engle_granger_test(np.array([]), np.array([]))
        with pytest.raises(ValueError):
            engle_granger_test(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_cointegrated_series_rejects_no_cointegration(self) -> None:
        x = _random_walk(500, seed=RNG_SEED)
        spread = _stationary_ar1(500, phi=0.5, seed=RNG_SEED + 1)
        y = 2.0 * x + spread
        beta, _, p_value = engle_granger_test(y, x)
        assert beta == pytest.approx(2.0, abs=0.5)
        assert p_value < 0.05

    def test_uncorrelated_random_walks_do_not_strongly_reject(self) -> None:
        x = _random_walk(500, seed=RNG_SEED)
        y = _random_walk(500, seed=RNG_SEED + 7)
        _, _, p_value = engle_granger_test(y, x)
        assert p_value > 0.01

    def test_statistic_matches_no_constant_adf_on_same_residuals_but_pvalue_does_not(
        self,
    ) -> None:
        # The step-2 regression has no constant (the step-1 residuals are
        # mean-zero by construction), so the statistic equals `adf_test`'s
        # with `trend="n"` on the same residuals -- and differs from the
        # constant-including `trend="c"` fit. The p-value must still use
        # the N=2 Engle-Granger surface, not the N=1 (plain Dickey-Fuller)
        # surface `adf_test` applies, because the residuals came from an
        # estimated 2-variable cointegrating regression.
        # phi close to 1 keeps the statistic in a range where the N=1 vs
        # N=2 difference is actually visible.
        x = _random_walk(300, seed=RNG_SEED)
        spread = _stationary_ar1(300, phi=0.99, seed=RNG_SEED + 1)
        y = 2.0 * x + spread
        _, eg_adf_stat, eg_p_value = engle_granger_test(y, x)

        x_reg = np.column_stack([np.ones_like(x), x])
        ols_beta, _, _, _ = np.linalg.lstsq(x_reg, y, rcond=None)
        residuals = y - x_reg @ ols_beta
        stat_no_constant, p_n1 = adf_test(residuals, max_lags=1, trend="n")
        stat_with_constant, _ = adf_test(residuals, max_lags=1, trend="c")

        assert eg_adf_stat == pytest.approx(stat_no_constant, rel=1e-12)
        assert eg_adf_stat != pytest.approx(stat_with_constant, rel=1e-6)
        assert eg_p_value > p_n1

    def test_pvalue_is_not_floored_for_strongly_cointegrated_pairs(self) -> None:
        # A near-white-noise spread gives a statistic far past MacKinnon's
        # tabulated minimum; the p-value must saturate at exactly 0.0, not
        # at `adf_test`'s 0.0001 floor.
        x = _random_walk(2000, seed=RNG_SEED)
        spread = _stationary_ar1(2000, phi=0.0, seed=RNG_SEED + 1)
        y = 2.0 * x + spread
        _, _, p_value = engle_granger_test(y, x)
        assert p_value == 0.0


class TestJohansenTraceTest:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            johansen_trace_test(np.zeros((10, 1)))
        with pytest.raises(ValueError):
            johansen_trace_test(np.zeros((3, 2)), n_lags=5)

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(200, 2))
        trace_stats, crit_values, eigenvectors = johansen_trace_test(data, n_lags=1)
        assert trace_stats.shape == (2,)
        assert crit_values.shape == (2,)
        assert eigenvectors.shape == (2, 2)

    def test_trace_statistics_non_negative(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(200, 3))
        trace_stats, _, _ = johansen_trace_test(data, n_lags=1)
        assert np.all(trace_stats >= 0.0)

    def test_default_critical_values_match_published_table_unrestricted_constant(self) -> None:
        # Regression test for the production bug: the old table was the
        # no-deterministic-term case shifted by one index (its 12.3212 was
        # the "c" case's k-r=2 value, not k-r=1), so it required a trace
        # statistic above ~12.32 where the correct 95% threshold is 3.84 —
        # the test almost never rejected. These are MacKinnon, Haug &
        # Michelis (1999) Table 1's unrestricted-constant 95% values,
        # k-r = 1..6, and must never silently shift again.
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(50, 6))
        _, crit_values, _ = johansen_trace_test(data, n_lags=1)
        expected = np.array([95.7542, 69.8189, 47.8545, 29.7961, 15.4943, 3.8415])
        np.testing.assert_allclose(crit_values, expected)

    @pytest.mark.parametrize(
        ("deterministic", "confidence", "expected_k_minus_r_1"),
        [
            ("n", 0.95, 4.1296),
            ("c", 0.95, 3.8415),
            ("ct", 0.95, 3.8415),
            ("c", 0.90, 2.7055),
            ("c", 0.99, 6.6349),
        ],
    )
    def test_critical_value_selects_correct_table_and_column(
        self, deterministic: str, confidence: float, expected_k_minus_r_1: float
    ) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(50, 2))
        _, crit_values, _ = johansen_trace_test(
            data, n_lags=1, deterministic=deterministic, confidence=confidence
        )
        assert crit_values[-1] == pytest.approx(expected_k_minus_r_1)

    def test_invalid_deterministic_or_confidence_raises(self) -> None:
        data = np.random.default_rng(RNG_SEED).normal(size=(50, 2))
        with pytest.raises(ValueError):
            johansen_trace_test(data, deterministic="bogus")
        with pytest.raises(ValueError):
            johansen_trace_test(data, confidence=0.5)

    def test_supports_up_to_twelve_series(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(50, 12))
        trace_stats, crit_values, eigenvectors = johansen_trace_test(data, n_lags=1)
        assert trace_stats.shape == (12,)
        assert crit_values.shape == (12,)
        assert eigenvectors.shape == (12, 12)


class TestJohansenMaxEigenvalueTest:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(200, 3))
        max_eig_stats, crit_values, eigvals, eigenvectors = johansen_max_eigenvalue_test(
            data, n_lags=1
        )
        assert max_eig_stats.shape == (3,)
        assert crit_values.shape == (3,)
        assert eigvals.shape == (3,)
        assert eigenvectors.shape == (3, 3)

    def test_eigenvalues_descending_and_in_unit_interval(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(200, 3))
        _, _, eigvals, _ = johansen_max_eigenvalue_test(data, n_lags=1)
        assert np.all(eigvals >= 0.0) and np.all(eigvals < 1.0)
        assert all(a >= b for a, b in pairwise(eigvals))

    def test_max_eigenvalue_statistics_sum_to_trace_statistic(self) -> None:
        # trace_stat[r] = sum_{i=r}^{k-1} max_eig_stat[i] by construction
        # (both derive from the same eigenvalues); check the r=0 case, which
        # is the full trace statistic.
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(200, 3))
        trace_stats, _, _ = johansen_trace_test(data, n_lags=1)
        max_eig_stats, _, _, _ = johansen_max_eigenvalue_test(data, n_lags=1)
        assert trace_stats[0] == pytest.approx(np.sum(max_eig_stats))

    def test_default_critical_values_match_published_table(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        data = rng.normal(size=(50, 3))
        _, crit_values, _, _ = johansen_max_eigenvalue_test(data, n_lags=1)
        expected = np.array([21.1314, 14.2639, 3.8415])
        np.testing.assert_allclose(crit_values, expected)


class TestOuHalfLife:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ou_half_life(np.array([]))
        diverging = np.cumsum(np.ones(50))
        with pytest.raises(ValueError):
            ou_half_life(diverging)

    def test_fast_mean_reverting_series_gives_short_half_life(self) -> None:
        # spread_t = phi * spread_{t-1} + noise is the level-form AR(1); the
        # level-form coefficient the spec's formula uses is phi itself.
        phi = 0.5
        spread = _stationary_ar1(2000, phi=phi, seed=RNG_SEED)
        expected = -np.log(2.0) / np.log(phi)
        half_life = ou_half_life(spread)
        assert half_life > 0.0
        assert half_life == pytest.approx(expected, rel=0.5)

    # Note: an explicit "random walk / weakly-explosive series raises" test
    # was deliberately omitted here beyond test_invalid_inputs' deterministic
    # ramp. Both a true unit-root random walk and a weakly-explosive noisy
    # AR(1) can produce a *negative* finite-sample OLS point estimate purely
    # from sampling noise (the well-known Dickey-Fuller finite-sample bias),
    # so "raises ValueError" is not a reliable invariant for a single noisy
    # draw of either -- only a deterministic, noise-free diverging series
    # (as in test_invalid_inputs) gives a guaranteed assertion.


class TestSpreadZscore:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            spread_zscore(np.array([1.0, 2.0, 3.0]), window=0)
        with pytest.raises(ValueError):
            spread_zscore(np.array([1.0, 2.0, 3.0]), window=10)

    def test_leading_values_are_nan(self) -> None:
        spread = np.arange(20, dtype=np.float64)
        window = 5
        z = spread_zscore(spread, window)
        assert np.all(np.isnan(z[: window - 1]))
        assert not np.any(np.isnan(z[window - 1 :]))

    def test_stationary_input_has_approx_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        spread = rng.normal(0.0, 1.0, size=2000)
        window = 50
        z = spread_zscore(spread, window)
        tail = z[window - 1 :]
        assert np.mean(tail) == pytest.approx(0.0, abs=0.3)
        assert np.std(tail) == pytest.approx(1.0, abs=0.3)
