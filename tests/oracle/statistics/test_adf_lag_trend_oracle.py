"""Oracle tests for `adf_test`'s `trend`/`autolag` options vs statsmodels.

Covers Workstream B's remaining additive item: "Add lag selection
(AIC/BIC/t-stat, as `adfuller(autolag=...)`) and a `trend` option ("n",
"c", "ct"), additively."

Oracle: `statsmodels.tsa.stattools.adfuller`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.stattools import adfuller

from quantcore.statistics.cointegration import adf_test

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def _stationary_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.1, size=n)
    return 0.5 * np.sin(np.linspace(0.0, 20.0, n)) + 3.0 * noise


_SERIES_FNS = [
    lambda: _random_walk(300, RNG_SEED),
    lambda: _stationary_series(300, RNG_SEED),
]


class TestTrendOptionMatchesStatsmodels:
    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("series_fn", _SERIES_FNS)
    def test_statistic_and_pvalue_match_at_fixed_lags(self, trend: str, series_fn) -> None:
        series = series_fn()
        quantcore_stat, quantcore_p = adf_test(series, max_lags=1, trend=trend)
        sm_stat, sm_p, *_ = adfuller(
            series, maxlag=1, regression=trend, autolag=None, result_object=False
        )
        assert quantcore_stat == pytest.approx(sm_stat, rel=1e-6)
        assert quantcore_p == pytest.approx(sm_p, abs=1e-3)


class TestAutolagMatchesStatsmodels:
    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("autolag", ["aic", "bic", "t-stat"])
    @pytest.mark.parametrize("series_fn", _SERIES_FNS)
    def test_statistic_matches_at_selected_lag(self, trend: str, autolag: str, series_fn) -> None:
        series = series_fn()
        quantcore_stat, quantcore_p = adf_test(series, max_lags=10, trend=trend, autolag=autolag)
        sm_stat, sm_p, *_ = adfuller(
            series, maxlag=10, regression=trend, autolag=autolag, result_object=False
        )
        assert quantcore_stat == pytest.approx(sm_stat, rel=1e-6)
        assert quantcore_p == pytest.approx(sm_p, abs=1e-3)
