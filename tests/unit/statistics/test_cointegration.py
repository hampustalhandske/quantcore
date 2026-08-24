"""Tests for cointegration and mean-reversion statistics."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.statistics.cointegration import (
    adf_test,
    engle_granger_test,
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
