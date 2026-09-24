"""Oracle tests for `engle_granger_test` / `adf_test` vs statsmodels.

`engle_granger_test` must reproduce `statsmodels.tsa.stattools.coint(y, x,
trend="c", maxlag=1, autolag=None)` -- statistic and p-value -- which
encodes the standard Engle-Granger (1987) conventions: the step-2 ADF
regression on the step-1 residuals carries no constant (`adfuller(...,
regression="n")`), and its p-value comes from MacKinnon's (1994) N=2
response surface for the "c" case (the cointegrating regression's
deterministic term), unclamped (`mackinnonp(stat, "c", N=2)`).

`adf_test` (N=1) is checked against `adfuller` directly. Both draw on the
same MacKinnon (1994) response surface quantcore implements independently
from the paper's published Table II coefficients (reproduced in
`cointegration.py`; cross-checked against `statsmodels` here since it is
the standard implementation, per the ground rule that oracle tests
validate agreement, not that one side is copied from the other).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller, coint

from quantcore.statistics.cointegration import (
    _mackinnon_pvalue,
    adf_test,
    engle_granger_test,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 1.0, size=n))


def _stationary_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=n)
    series = np.empty(n)
    series[0] = noise[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    return series


class TestMackinnonPvalueMatchesStatsmodels:
    """The response-surface table itself, N=1 and N=2, across its domain."""

    @pytest.mark.parametrize("n_series", [1, 2, 3, 4])
    @pytest.mark.parametrize("t_stat", [-6.0, -4.5, -3.5, -3.0, -2.5, -1.5, -0.5, 0.5])
    def test_pvalue_matches_statsmodels_mackinnonp(self, t_stat: float, n_series: int) -> None:
        expected = mackinnonp(t_stat, regression="c", N=n_series)
        actual = _mackinnon_pvalue(t_stat, n_series=n_series)
        # Both sides clamp to [1e-4, 1 - 1e-4] near the tabulated boundary,
        # but statsmodels clamps to exactly [0.0, 1.0] there instead — only
        # compare where neither side is at its own floor/ceiling.
        if 1e-4 < expected < 1 - 1e-4:
            assert actual == pytest.approx(expected, abs=1e-4)


class TestAdfTestMatchesStatsmodelsAdfuller:
    """adf_test (N=1) end to end against statsmodels' adfuller, case='c'."""

    @pytest.mark.parametrize("seed", range(5))
    def test_statistic_and_pvalue_match(self, seed: int) -> None:
        series = _stationary_ar1(400, phi=0.6, seed=seed)
        quantcore_stat, quantcore_p = adf_test(series, max_lags=1)
        sm_stat, sm_p, *_ = adfuller(
            series, maxlag=1, regression="c", autolag=None, result_object=False
        )
        assert quantcore_stat == pytest.approx(sm_stat, rel=1e-6)
        assert quantcore_p == pytest.approx(sm_p, abs=1e-3)


class TestEngleGrangerMatchesStatsmodelsCoint:
    """Statistic and p-value must equal `coint`'s to 1e-10 for cointegrated,
    independent-random-walk and near-unit-root pairs, from 10 to 5,000
    observations."""

    @pytest.mark.parametrize("family", ["cointegrated", "independent", "near_unit_root"])
    @pytest.mark.parametrize("n_obs", [10, 30, 100, 500, 5000])
    def test_statistic_and_pvalue_match_coint(self, family: str, n_obs: int) -> None:
        for seed in range(5):
            x = _random_walk(n_obs, seed=RNG_SEED + seed)
            if family == "cointegrated":
                y = 2.0 * x + _stationary_ar1(n_obs, phi=0.5, seed=RNG_SEED + 100 + seed)
            elif family == "near_unit_root":
                y = 2.0 * x + _stationary_ar1(n_obs, phi=0.99, seed=RNG_SEED + 100 + seed)
            else:
                y = _random_walk(n_obs, seed=RNG_SEED + 100 + seed)

            _, eg_stat, eg_p = engle_granger_test(y, x)
            sm_stat, sm_p, _ = coint(y, x, trend="c", maxlag=1, autolag=None)

            assert eg_stat == pytest.approx(sm_stat, rel=1e-10, abs=1e-10)
            assert eg_p == pytest.approx(sm_p, rel=1e-10, abs=1e-10)

    def test_pvalue_matches_n_equals_two_surface_not_n_equals_one(self) -> None:
        x = _random_walk(300, seed=RNG_SEED)
        spread = _stationary_ar1(300, phi=0.99, seed=RNG_SEED + 1)
        y = 2.0 * x + spread
        _, eg_stat, eg_p = engle_granger_test(y, x)

        expected_n2 = mackinnonp(eg_stat, regression="c", N=2)
        wrong_n1 = mackinnonp(eg_stat, regression="c", N=1)

        assert eg_p == pytest.approx(expected_n2, abs=1e-10)
        assert eg_p != pytest.approx(wrong_n1, abs=1e-3)

    @pytest.mark.parametrize(
        ("statistic", "wrong_n1_p", "correct_n2_p"),
        [
            (-3.0, 0.035, 0.110),
            (-2.86, 0.050, 0.147),
        ],
    )
    def test_brief_numbers_reproduced(
        self, statistic: float, wrong_n1_p: float, correct_n2_p: float
    ) -> None:
        # At these statistics, using N=1 vs. the correct N=2 gives
        # materially different p-values.
        assert _mackinnon_pvalue(statistic, n_series=1) == pytest.approx(wrong_n1_p, abs=2e-3)
        assert _mackinnon_pvalue(statistic, n_series=2) == pytest.approx(correct_n2_p, abs=2e-3)


class TestMonteCarloSize:
    """Under independent random walks (no cointegration), the null is true
    by construction: rejection rate at the 5% level must be close to 5%.

    This is the test the old (N=1-for-everything) implementation would fail:
    reusing the Dickey-Fuller distribution for a residual-based test
    overstates significance, so its rejection rate under this null was far
    above 5% (the exact "expensive direction for pairs trading" failure
    mode described in the finding).
    """

    def test_engle_granger_rejection_rate_near_nominal_five_percent(self) -> None:
        n_trials = 300
        n_obs = 250
        alpha = 0.05
        rng = np.random.default_rng(RNG_SEED + 100)
        rejections = 0
        for _ in range(n_trials):
            x = np.cumsum(rng.normal(0.0, 1.0, size=n_obs))
            y = np.cumsum(rng.normal(0.0, 1.0, size=n_obs))
            _, _, p_value = engle_granger_test(y, x)
            if p_value < alpha:
                rejections += 1
        observed_rate = rejections / n_trials

        # Binomial 99.5% CI half-width for p=0.05, n=300 trials.
        se = np.sqrt(alpha * (1 - alpha) / n_trials)
        margin = 2.81 * se  # z for a two-sided 99.5% interval
        assert abs(observed_rate - alpha) < margin, (
            f"observed rejection rate {observed_rate:.3f} is not within "
            f"{margin:.3f} of the nominal {alpha}; expected ~5% under a true null"
        )
