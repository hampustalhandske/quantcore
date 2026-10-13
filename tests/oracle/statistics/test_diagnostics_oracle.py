"""Oracle tests for `jarque_bera_test`, `arch_lm_test` and `kpss_test`.

Oracles: `scipy.stats.jarque_bera`, `statsmodels.stats.diagnostic.het_arch`
and `statsmodels.tsa.stattools.kpss`. Each quantcore function implements
the same published definition as its oracle, so statistics and p-values
are held to closed-form tolerances rather than to the looser ones a fitted
or simulated quantity would get.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from quantcore.statistics import arch_lm_test, jarque_bera_test, kpss_test

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921
LENGTHS = [8, 30, 120, 500, 5000]


def _series(family: str, n: int, seed: int) -> np.ndarray:
    """One series of the named family; scales are those of daily returns unless stated."""
    rng = np.random.default_rng(seed)
    if family == "normal":
        return rng.normal(0.0005, 0.01, size=n)
    if family == "student_t":
        return rng.standard_t(3, size=n) * 0.01
    if family == "skewed":
        return 0.02 * (rng.lognormal(0.0, 0.5, size=n) - 1.0)
    if family == "near_constant":
        # Mean 4e5 times the spread: the regime where a single centring
        # pass loses digits.
        return np.full(n, 0.0004) + rng.normal(0.0, 1e-9, size=n)
    if family in ("ar1", "ar1_negative", "near_unit_root"):
        phi = {"ar1": 0.6, "ar1_negative": -0.7, "near_unit_root": 0.98}[family]
        noise = rng.normal(0.0, 0.01, size=n)
        values = np.empty(n)
        values[0] = noise[0]
        for t in range(1, n):
            values[t] = phi * values[t - 1] + noise[t]
        return values
    if family == "garch":
        omega, alpha, beta = 1e-6, 0.1, 0.85
        z = rng.normal(0.0, 1.0, size=n)
        values = np.empty(n)
        variance = omega / (1.0 - alpha - beta)
        for t in range(n):
            values[t] = np.sqrt(variance) * z[t]
            variance = omega + alpha * values[t] ** 2 + beta * variance
        return values
    if family == "random_walk":
        return np.cumsum(rng.normal(0.0, 1.0, size=n))
    if family == "random_walk_with_drift":
        return np.cumsum(rng.normal(0.2, 1.0, size=n))
    if family == "trend_stationary":
        return 0.05 * np.arange(n) + rng.normal(0.0, 1.0, size=n)
    raise ValueError(f"unknown family {family!r}")


class TestJarqueBeraMatchesScipy:
    @pytest.mark.parametrize("family", ["normal", "student_t", "skewed", "ar1", "near_constant"])
    @pytest.mark.parametrize("n", [2, 3, *LENGTHS])
    def test_statistic_and_pvalue_match(self, family: str, n: int) -> None:
        for seed in range(5):
            series = _series(family, n, RNG_SEED + seed)
            statistic, p_value = jarque_bera_test(series)
            expected = jarque_bera(series)
            # 1e-9 rather than the closed-form 1e-10: skewness enters
            # squared, so a nearly symmetric sample loses relative precision
            # in the statistic. The absolute term makes p-values below 1e-12
            # equal.
            assert statistic == pytest.approx(float(expected.statistic), rel=1e-9, abs=1e-12)
            assert p_value == pytest.approx(float(expected.pvalue), rel=1e-9, abs=1e-12)

    def test_constant_series_raises_where_scipy_returns_nan(self) -> None:
        series = np.full(50, 0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.isnan(jarque_bera(series).statistic)
        with pytest.raises(ValueError, match="constant"):
            jarque_bera_test(series)


class TestArchLmMatchesStatsmodels:
    @pytest.mark.parametrize("family", ["normal", "student_t", "ar1", "garch"])
    @pytest.mark.parametrize("n", LENGTHS)
    @pytest.mark.parametrize("n_lags", [None, 1, 3])
    def test_statistic_and_pvalue_match(self, family: str, n: int, n_lags: int | None) -> None:
        for seed in range(5):
            residuals = _series(family, n, RNG_SEED + seed)
            statistic, p_value = arch_lm_test(residuals, n_lags)
            sm_statistic, sm_p_value, _, _ = het_arch(residuals, nlags=n_lags, result_object=False)
            # Absolute as well as relative: LM = (n - m) * (1 - SSR/SST)
            # cancels as it approaches zero, and for one lag the p-value
            # moves by dLM / sqrt(2 * pi * LM) there.
            assert statistic == pytest.approx(float(sm_statistic), rel=1e-8, abs=1e-8)
            assert p_value == pytest.approx(float(sm_p_value), rel=1e-8, abs=1e-8)

    def test_garch_series_is_flagged_and_iid_series_is_not(self) -> None:
        # Sanity of the comparison above: the families do differ in what
        # the test should say about them.
        _, p_garch = arch_lm_test(_series("garch", 5000, RNG_SEED), n_lags=5)
        _, p_normal = arch_lm_test(_series("normal", 5000, RNG_SEED), n_lags=5)
        assert p_garch < 1e-6
        assert p_normal > 0.01


class TestArchLmWhereStatsmodelsLosesTheLagColumns:
    """`het_arch` is not scale-free in floating point; `arch_lm_test` is.

    statsmodels fits the auxiliary regression with a pseudo-inverse that
    discards singular values below 1e-15 of the largest. Beside the column
    of ones, squared residuals of order 1e-17 fall under that cutoff, the
    lag columns are dropped, and the statistic comes out as zero for any
    series. These tests record that behaviour so that a disagreement at
    such scales is not read as a quantcore defect.
    """

    def test_statsmodels_statistic_collapses_to_zero_on_a_small_scale_series(self) -> None:
        residuals = _series("garch", 1000, RNG_SEED)
        sm_unit, _, _, _ = het_arch(residuals, nlags=5, result_object=False)
        with warnings.catch_warnings():
            # statsmodels itself reports the design as rank-deficient here.
            warnings.simplefilter("ignore")
            sm_small, sm_small_p, _, _ = het_arch(residuals * 1e-6, nlags=5, result_object=False)
        assert sm_unit > 10.0
        assert abs(sm_small) < 1e-6
        assert sm_small_p == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6, 1e12])
    def test_quantcore_statistic_is_the_unit_scale_one_at_every_scale(self, scale: float) -> None:
        residuals = _series("garch", 1000, RNG_SEED)
        sm_unit, sm_unit_p, _, _ = het_arch(residuals, nlags=5, result_object=False)
        statistic, p_value = arch_lm_test(residuals * scale, n_lags=5)
        assert statistic == pytest.approx(float(sm_unit), rel=1e-8)
        assert p_value == pytest.approx(float(sm_unit_p), rel=1e-7)


def _statsmodels_kpss(
    series: np.ndarray, n_lags: int | None, trend: str
) -> tuple[float, float, bool, int]:
    """statsmodels' (statistic, p-value, at-bound flag, lag count).

    The flag is whether `InterpolationWarning` fired: statsmodels reports a
    p-value outside its table as the nearest tabulated one and says so only
    through that warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, p_value, lags, _ = kpss(
            series,
            regression=trend,
            nlags="auto" if n_lags is None else n_lags,
            result_object=False,
        )
    at_bound = any(issubclass(w.category, InterpolationWarning) for w in caught)
    return float(statistic), float(p_value), at_bound, int(lags)


KPSS_FAMILIES = [
    "normal",
    "student_t",
    "ar1",
    "ar1_negative",
    "garch",
    "near_unit_root",
    "random_walk",
    "random_walk_with_drift",
    "trend_stationary",
]


class TestKpssMatchesStatsmodels:
    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("family", KPSS_FAMILIES)
    @pytest.mark.parametrize("n", LENGTHS)
    def test_automatic_lags_statistic_pvalue_and_flag_match(
        self, trend: str, family: str, n: int
    ) -> None:
        for seed in range(5):
            series = _series(family, n, RNG_SEED + seed)
            statistic, p_value, at_bound = kpss_test(series, trend=trend)
            sm_statistic, sm_p_value, sm_at_bound, sm_lags = _statsmodels_kpss(series, None, trend)

            assert statistic == pytest.approx(sm_statistic, rel=1e-9, abs=0.0)
            assert p_value == pytest.approx(sm_p_value, rel=1e-9, abs=0.0)
            assert at_bound is sm_at_bound
            # The same lag count, not merely a statistic that happens to be
            # close: the automatic result is bit-for-bit the fixed-lag
            # result at statsmodels' count.
            assert (statistic, p_value, at_bound) == kpss_test(series, sm_lags, trend)

    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("family", ["normal", "ar1_negative", "random_walk"])
    @pytest.mark.parametrize("n_lags", [0, 1, 5, 20])
    def test_fixed_lags_statistic_pvalue_and_flag_match(
        self, trend: str, family: str, n_lags: int
    ) -> None:
        for seed in range(5):
            series = _series(family, 250, RNG_SEED + seed)
            statistic, p_value, at_bound = kpss_test(series, n_lags, trend)
            sm_statistic, sm_p_value, sm_at_bound, _ = _statsmodels_kpss(series, n_lags, trend)
            assert statistic == pytest.approx(sm_statistic, rel=1e-9, abs=0.0)
            assert p_value == pytest.approx(sm_p_value, rel=1e-9, abs=0.0)
            assert at_bound is sm_at_bound

    def test_families_cover_both_ends_and_the_interior_of_the_table(self) -> None:
        # The comparison above is only informative if it sees p-values at
        # each bound and between them.
        seen = set()
        for family in KPSS_FAMILIES:
            for n in LENGTHS:
                for seed in range(5):
                    _, p_value, at_bound = kpss_test(_series(family, n, RNG_SEED + seed))
                    seen.add((p_value if at_bound else "interior", at_bound))
        assert seen == {(0.10, True), (0.01, True), ("interior", False)}


class TestKpssWhereStatsmodelsReturnsRoundingResidue:
    """Degenerate series: quantcore raises where statsmodels fails or returns a number.

    The mean of `np.full(252, 0.07)` is not exactly 0.07, so the residuals
    of a constant series are rounding residue, not zeros; statsmodels turns
    that residue into a statistic and a p-value. With an exactly
    representable mean it divides zero by zero instead.
    """

    def test_constant_series_with_inexact_mean(self) -> None:
        series = np.full(252, 0.07)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_statistic, sm_p_value, _, _ = kpss(
                series, regression="c", nlags="auto", result_object=False
            )
        assert np.isfinite(sm_statistic)
        assert sm_p_value == 0.01
        with pytest.raises(ValueError, match="rounding error"):
            kpss_test(series)

    def test_constant_series_with_exact_mean(self) -> None:
        series = np.full(64, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises((ValueError, OverflowError, ZeroDivisionError)):
                kpss(series, regression="c", nlags="auto", result_object=False)
        with pytest.raises(ValueError, match="rounding error"):
            kpss_test(series)

    def test_exactly_linear_series_under_the_trend_null(self) -> None:
        series = np.cumsum(np.full(40, -0.999))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_statistic, _, _, _ = kpss(series, regression="ct", nlags="auto", result_object=False)
        assert np.isfinite(sm_statistic)
        with pytest.raises(ValueError, match="rounding error"):
            kpss_test(series, trend="ct")
