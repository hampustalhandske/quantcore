"""Tests for cointegration and mean-reversion statistics."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.statistics.cointegration import (
    adf_test,
    engle_granger_test,
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

    def test_adf_pvalue_of_strongly_stationary_series_is_below_one_in_ten_thousand(self) -> None:
        series = _stationary_ar1(500, phi=0.2)
        _, p_value = adf_test(series, max_lags=1)
        assert 0.0 <= p_value < 1e-4

    def test_adf_pvalue_of_explosive_series_is_exactly_one(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        series = 1.05 ** np.arange(60) + rng.normal(0.0, 0.01, size=60)
        statistic, p_value = adf_test(series, max_lags=1)
        assert statistic > 2.74
        assert p_value == 1.0

    def test_random_walk_reports_large_pvalue(self) -> None:
        # Genuine unit-root series must not look stationary to downstream
        # leg-selection filters.
        for seed in range(5):
            series = _random_walk(500, seed=seed)
            _, p_value = adf_test(series, max_lags=1)
            assert p_value > 0.3

    def test_strongly_mean_reverting_series_reports_small_pvalue(self) -> None:
        series = _stationary_ar1(500, phi=0.2)
        _, p_value = adf_test(series, max_lags=1)
        assert p_value < 0.05


class TestStatsmodelsConventions:
    """The statsmodels calls receive every option explicitly."""

    @pytest.mark.parametrize("autolag", [None, "aic"])
    def test_adf_passes_regression_lags_and_autolag(
        self, monkeypatch: pytest.MonkeyPatch, autolag: str | None
    ) -> None:
        seen: dict[str, object] = {}

        def spy(series: np.ndarray, **kwargs: object) -> tuple[float, float]:
            seen.update(kwargs)
            return -3.0, 0.04

        monkeypatch.setattr("statsmodels.tsa.stattools.adfuller", spy)
        assert adf_test(_random_walk(50), max_lags=3, trend="ct", autolag=autolag) == (-3.0, 0.04)
        assert seen == {
            "maxlag": 3,
            "regression": "ct",
            "autolag": autolag,
            "store": False,
            "regresults": False,
            "result_object": False,
        }

    def test_engle_granger_passes_the_aeg_conventions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def spy(y: np.ndarray, x: np.ndarray, **kwargs: object) -> tuple[float, float, object]:
            seen.update(kwargs)
            return -3.5, 0.02, None

        monkeypatch.setattr("statsmodels.tsa.stattools.coint", spy)
        x = _random_walk(50)
        beta, statistic, p_value = engle_granger_test(2.0 * x + 1.0, x)
        assert beta == pytest.approx(2.0)
        assert (statistic, p_value) == (-3.5, 0.02)
        assert seen == {
            "trend": "c",
            "method": "aeg",
            "maxlag": 1,
            "autolag": None,
            "return_results": None,
        }

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValueError):
            adf_test(np.ones(50))


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
        # tabulated minimum, where the p-value is exactly 0.0.
        x = _random_walk(2000, seed=RNG_SEED)
        spread = _stationary_ar1(2000, phi=0.0, seed=RNG_SEED + 1)
        y = 2.0 * x + spread
        _, _, p_value = engle_granger_test(y, x)
        assert p_value == 0.0


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
