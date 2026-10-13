"""Tests for the residual diagnostics: Jarque-Bera, ARCH-LM and KPSS."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from quantcore.statistics import arch_lm_test, diagnostics, jarque_bera_test, kpss_test
from quantcore.statistics.diagnostics import _KPSS_CRITICAL_VALUES, _KPSS_SIGNIFICANCE_LEVELS

RNG_SEED = 12345


def _white_noise(n: int, seed: int = RNG_SEED) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=n)


def _random_walk(n: int, seed: int = RNG_SEED) -> np.ndarray:
    return np.cumsum(_white_noise(n, seed))


def _arch1(n: int, alpha: float, seed: int = RNG_SEED) -> np.ndarray:
    """ARCH(1) series e_t = z_t * sqrt(0.2 + alpha * e_{t-1}^2), z_t standard normal."""
    z = _white_noise(n, seed)
    series = np.empty(n)
    series[0] = z[0]
    for t in range(1, n):
        series[t] = z[t] * math.sqrt(0.2 + alpha * series[t - 1] ** 2)
    return series


def _exact_jarque_bera(series: np.ndarray) -> float:
    """Jarque-Bera statistic of the given floats in exact rational arithmetic.

    Uses S^2 = m3^2 / m2^3, so no square root is needed and the only
    rounding is the final conversion to float.
    """
    values = [Fraction(float(v)) for v in series]
    n = len(values)
    mean = sum(values, Fraction(0)) / n
    deviations = [v - mean for v in values]
    m2 = sum((d**2 for d in deviations), Fraction(0)) / n
    m3 = sum((d**3 for d in deviations), Fraction(0)) / n
    m4 = sum((d**4 for d in deviations), Fraction(0)) / n
    skewness_squared = m3**2 / m2**3
    excess_kurtosis = m4 / m2**2 - 3
    return float(Fraction(n, 6) * (skewness_squared + excess_kurtosis**2 / 4))


class TestJarqueBeraTest:
    def test_hand_computed_value(self) -> None:
        # Deviations (-1, -1, -1, -1, 4): m2 = 4, m3 = 12, m4 = 52, so
        # S = 12 / 4^1.5 = 3/2 and K = 52/16 - 3 = 1/4, and
        # JB = 5/6 * (9/4 + 1/64) = 725/384.
        series = np.array([2.0, 2.0, 2.0, 2.0, 7.0])
        statistic, p_value = jarque_bera_test(series)
        assert statistic == pytest.approx(725.0 / 384.0, rel=1e-13)
        assert p_value == pytest.approx(math.exp(-725.0 / 768.0), rel=1e-13)

    def test_any_two_distinct_values_give_one_third(self) -> None:
        # With two observations S = 0 and K = -2 whatever the values are.
        statistic, p_value = jarque_bera_test(np.array([0.3, -1.7]))
        assert statistic == pytest.approx(1.0 / 3.0, rel=1e-13)
        assert p_value == pytest.approx(math.exp(-1.0 / 6.0), rel=1e-13)

    def test_p_value_is_the_chi_squared_two_survival_function(self) -> None:
        statistic, p_value = jarque_bera_test(_white_noise(300))
        assert p_value == pytest.approx(math.exp(-statistic / 2.0), rel=1e-12)

    def test_heavy_tailed_sample_rejects_and_normal_sample_does_not(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        _, p_normal = jarque_bera_test(rng.normal(0.0, 1.0, size=2000))
        _, p_heavy = jarque_bera_test(rng.standard_t(3, size=2000))
        assert p_normal > 0.05
        assert p_heavy < 1e-6

    def test_matches_exact_arithmetic_when_the_mean_dwarfs_the_spread(self) -> None:
        # Mean 4e-4 against a spread of 1e-9: the rounding error of the mean
        # is 5e-11 of the spread, enough to cost five digits of the
        # statistic unless the deviations are centred a second time.
        rng = np.random.default_rng(RNG_SEED)
        series = np.full(400, 0.0004) + rng.normal(0.0, 1e-9, size=400)
        statistic, _ = jarque_bera_test(series)
        assert statistic == pytest.approx(_exact_jarque_bera(series), rel=1e-11)

    @pytest.mark.parametrize("scale", [1e-150, 1e-8, 1e8, 1e150])
    def test_statistic_does_not_depend_on_the_scale_of_the_series(self, scale: float) -> None:
        series = _white_noise(200)
        expected, _ = jarque_bera_test(series)
        statistic, p_value = jarque_bera_test(series * scale)
        assert statistic == pytest.approx(expected, rel=1e-11)
        assert math.isfinite(p_value)

    @pytest.mark.parametrize("series", [np.full(3, 0.1), np.full(252, 0.07), np.zeros(10)])
    def test_constant_series_raises(self, series: np.ndarray) -> None:
        # The mean of np.full(252, 0.07) is not exactly 0.07, so its
        # deviations are rounding residue rather than zeros; the series is
        # constant all the same.
        with pytest.raises(ValueError, match="constant"):
            jarque_bera_test(series)

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            jarque_bera_test(np.array([]))
        with pytest.raises(ValueError, match="at least 2"):
            jarque_bera_test(np.array([1.0]))

    def test_two_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            jarque_bera_test(np.ones((4, 2)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_input_raises(self, bad: float) -> None:
        series = _white_noise(20)
        series[7] = bad
        with pytest.raises(ValueError, match="finite"):
            jarque_bera_test(series)

    def test_non_contiguous_view_gives_the_same_result_as_its_copy(self) -> None:
        series = _white_noise(100)
        assert jarque_bera_test(series[::2]) == jarque_bera_test(series[::2].copy())

    def test_returns_python_floats(self) -> None:
        statistic, p_value = jarque_bera_test(_white_noise(50))
        assert type(statistic) is float
        assert type(p_value) is float

    def test_input_is_not_modified(self) -> None:
        series = _white_noise(50)
        original = series.copy()
        jarque_bera_test(series)
        np.testing.assert_array_equal(series, original)


class TestArchLmTest:
    def test_hand_computed_value(self) -> None:
        # Squares 0.01, 0.04, 0.0025, 0.09; regressing the last three on a
        # constant and the first three gives Sxy = -51/32000,
        # Sxx = 63/80000, Syy = 37/9600, so R^2 = 867/1036 and
        # LM = 3 * R^2 = 2601/1036.
        residuals = np.array([0.1, -0.2, 0.05, 0.3])
        statistic, p_value = arch_lm_test(residuals, n_lags=1)
        assert statistic == pytest.approx(2601.0 / 1036.0, rel=1e-12)
        # chi-squared with one degree of freedom: sf(x) = erfc(sqrt(x / 2)).
        assert p_value == pytest.approx(math.erfc(math.sqrt(2601.0 / 2072.0)), rel=1e-12)

    def test_input_is_squared_without_demeaning(self) -> None:
        # A constant shift changes the squares, so it changes the statistic;
        # a test that demeaned its input first would not notice it.
        residuals = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.25, 0.02, -0.3])
        statistic, _ = arch_lm_test(residuals, n_lags=1)
        shifted, _ = arch_lm_test(residuals + 1.0, n_lags=1)
        assert shifted != pytest.approx(statistic, rel=1e-3)

    @pytest.mark.parametrize(
        ("n", "expected_lags"), [(5, 1), (23, 4), (49, 9), (50, 10), (400, 10)]
    )
    def test_default_lag_count_is_min_of_ten_and_a_fifth_of_the_sample(
        self, n: int, expected_lags: int
    ) -> None:
        residuals = _white_noise(n)
        assert arch_lm_test(residuals) == arch_lm_test(residuals, n_lags=expected_lags)

    def test_arch_series_rejects_and_iid_series_does_not(self) -> None:
        _, p_arch = arch_lm_test(_arch1(1000, alpha=0.6), n_lags=5)
        _, p_iid = arch_lm_test(_white_noise(1000), n_lags=5)
        assert p_arch < 1e-6
        assert p_iid > 0.05

    @pytest.mark.parametrize("scale", [1e-150, 1e-12, 1e-6, 1e6, 1e12, 1e150])
    def test_statistic_does_not_depend_on_the_scale_of_the_residuals(self, scale: float) -> None:
        residuals = _arch1(500, alpha=0.6)
        expected_statistic, expected_p = arch_lm_test(residuals, n_lags=5)
        statistic, p_value = arch_lm_test(residuals * scale, n_lags=5)
        assert statistic == pytest.approx(expected_statistic, rel=1e-9)
        assert p_value == pytest.approx(expected_p, rel=1e-7)

    def test_statistic_is_never_negative(self) -> None:
        for seed in range(50):
            statistic, p_value = arch_lm_test(_white_noise(12, seed), n_lags=1)
            assert statistic >= 0.0
            assert 0.0 <= p_value <= 1.0

    @pytest.mark.parametrize(
        "residuals",
        [
            np.zeros(12),
            np.full(12, 0.3),
            np.array([0.2, -0.2] * 6),
            # Varies only in its first observation, which the regression
            # sample for one lag (observations 2 onwards) does not contain.
            np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ],
    )
    def test_constant_squared_residuals_raise(self, residuals: np.ndarray) -> None:
        with pytest.raises(ValueError, match="constant"):
            arch_lm_test(residuals, n_lags=1)

    def test_non_positive_lag_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_lags"):
            arch_lm_test(_white_noise(50), n_lags=0)
        with pytest.raises(ValueError, match="n_lags"):
            arch_lm_test(_white_noise(50), n_lags=-2)

    def test_too_few_observations_for_the_lag_count_raises(self) -> None:
        # Three lags need 2 * 3 + 2 = 8 observations: five regression rows
        # for four coefficients.
        with pytest.raises(ValueError, match="at least 8"):
            arch_lm_test(_white_noise(7), n_lags=3)
        statistic, _ = arch_lm_test(_white_noise(8), n_lags=3)
        assert math.isfinite(statistic)

    def test_too_few_observations_for_the_default_lag_count_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 5"):
            arch_lm_test(_white_noise(4))

    def test_two_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            arch_lm_test(np.ones((10, 2)), n_lags=1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_input_raises(self, bad: float) -> None:
        residuals = _white_noise(30)
        residuals[11] = bad
        with pytest.raises(ValueError, match="finite"):
            arch_lm_test(residuals, n_lags=2)

    def test_non_contiguous_view_gives_the_same_result_as_its_copy(self) -> None:
        residuals = _white_noise(120)
        assert arch_lm_test(residuals[::2], n_lags=3) == arch_lm_test(
            residuals[::2].copy(), n_lags=3
        )

    def test_returns_python_floats(self) -> None:
        statistic, p_value = arch_lm_test(_white_noise(50))
        assert type(statistic) is float
        assert type(p_value) is float

    def test_input_is_not_modified(self) -> None:
        residuals = _white_noise(50)
        original = residuals.copy()
        arch_lm_test(residuals)
        np.testing.assert_array_equal(residuals, original)


def _reference_kpss_statistic(series: np.ndarray, n_lags: int, trend: str) -> float:
    """KPSS statistic written out term by term, for a given lag count.

    Detrends with the centred simple-regression formulas and builds the
    long-run variance from explicit double sums, sharing no code with the
    implementation under test.
    """
    n = series.size
    centred = series - series.mean()
    if trend == "ct":
        time = np.arange(1, n + 1) - (n + 1) / 2.0
        centred = centred - (time @ centred) / (time @ time) * time
    partial_sums = np.cumsum(centred)
    eta = float(partial_sums @ partial_sums) / n**2
    long_run_variance = float(centred @ centred)
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1.0)
        cross = sum(centred[t] * centred[t - lag] for t in range(lag, n))
        long_run_variance += 2.0 * weight * cross
    return eta / (long_run_variance / n)


class TestKpssTest:
    def test_critical_values_are_the_published_table(self) -> None:
        # Kwiatkowski, Phillips, Schmidt & Shin (1992), Table 1: upper-tail
        # critical values at the 10%, 5%, 2.5% and 1% levels.
        np.testing.assert_array_equal(_KPSS_SIGNIFICANCE_LEVELS, [0.10, 0.05, 0.025, 0.01])
        np.testing.assert_array_equal(_KPSS_CRITICAL_VALUES["c"], [0.347, 0.463, 0.574, 0.739])
        np.testing.assert_array_equal(_KPSS_CRITICAL_VALUES["ct"], [0.119, 0.146, 0.176, 0.216])

    def test_hand_computed_value_with_automatic_lags(self) -> None:
        # Residuals (1, -8, 7)/30. The automatic rule uses int(3^(2/9)) = 1
        # autocovariance: s0 = -14/2700, s1 = -128/2700, so the bandwidth is
        # 1.1447 * ((64/7)^2)^(1/3) * 3^(1/3) = 7.2, capped at n - 1 = 2
        # lags. Then eta = 1/162, s^2(2) = 1/81 and the statistic is 1/2,
        # one third of the way from 0.463 (5%) to 0.574 (2.5%).
        statistic, p_value, at_bound = kpss_test(np.array([0.1, -0.2, 0.3]))
        assert statistic == pytest.approx(0.5, rel=1e-12)
        assert p_value == pytest.approx(0.05 - 0.025 / 3.0, rel=1e-12)
        assert at_bound is False

    def test_hand_computed_level_value_inside_the_table(self) -> None:
        # Residuals (-2, 0, -1, 2, 1): partial sums (-2, -2, -3, -1, 0), so
        # eta = 18/25; with no lags s^2 = 10/5 and the statistic is 0.36,
        # between 0.347 (10%) and 0.463 (5%).
        series = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        statistic, p_value, at_bound = kpss_test(series, n_lags=0)
        assert statistic == pytest.approx(0.36, rel=1e-12)
        assert p_value == pytest.approx(0.10 - 0.05 * (0.36 - 0.347) / (0.463 - 0.347), rel=1e-12)
        assert at_bound is False

    def test_hand_computed_trend_values(self) -> None:
        # The same series detrended: slope 8/10, residuals
        # (-0.4, 0.8, -1.0, 1.2, -0.6), partial sums (-0.4, 0.4, -0.6, 0.6, 0),
        # eta = 1.04/25. With no lags s^2 = 3.6/5 and the statistic is
        # 13/225, below the table; with one lag the lag-1 cross products sum
        # to -3.04, s^2 = (3.6 - 3.04)/5 and the statistic is 13/35, above it.
        series = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        statistic, p_value, at_bound = kpss_test(series, n_lags=0, trend="ct")
        assert statistic == pytest.approx(13.0 / 225.0, rel=1e-12)
        assert (p_value, at_bound) == (0.10, True)

        statistic, p_value, at_bound = kpss_test(series, n_lags=1, trend="ct")
        assert statistic == pytest.approx(13.0 / 35.0, rel=1e-12)
        assert (p_value, at_bound) == (0.01, True)

    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("n_lags", [0, 1, 4, 12])
    def test_statistic_matches_the_definition_written_out(self, trend: str, n_lags: int) -> None:
        series = _random_walk(60) * 0.1 + _white_noise(60, RNG_SEED + 1)
        statistic, _, _ = kpss_test(series, n_lags=n_lags, trend=trend)
        assert statistic == pytest.approx(
            _reference_kpss_statistic(series, n_lags, trend), rel=1e-11
        )

    def test_p_value_is_reported_at_the_table_bounds_and_flagged(self) -> None:
        # A random walk is far above the 1% critical value and white noise
        # with many lags far below the 10% one: the p-value is then only
        # known to be at most 0.01, or at least 0.10.
        statistic, p_value, at_bound = kpss_test(_random_walk(500))
        assert statistic > 0.739
        assert (p_value, at_bound) == (0.01, True)

        statistic, p_value, at_bound = kpss_test(_white_noise(500), n_lags=0)
        assert statistic < 0.347
        assert (p_value, at_bound) == (0.10, True)

    @pytest.mark.parametrize(
        ("drift", "lower", "upper", "p_lower", "p_upper"),
        [
            (0.0018, 0.347, 0.463, 0.10, 0.05),
            (0.0022, 0.463, 0.574, 0.05, 0.025),
            (0.0026, 0.574, 0.739, 0.025, 0.01),
        ],
    )
    def test_p_value_is_interpolated_and_unflagged_inside_the_table(
        self, drift: float, lower: float, upper: float, p_lower: float, p_upper: float
    ) -> None:
        # Each drift puts the statistic of this noise series in a different
        # segment of the table, where the p-value is linear between the two
        # tabulated significance levels.
        series = _white_noise(200) + drift * np.arange(200)
        statistic, p_value, at_bound = kpss_test(series, n_lags=4)
        assert lower < statistic < upper
        expected = p_lower + (p_upper - p_lower) * (statistic - lower) / (upper - lower)
        assert p_value == pytest.approx(expected, rel=1e-12)
        assert at_bound is False

    def test_trend_stationary_series_is_rejected_as_level_but_not_as_trend_stationary(
        self,
    ) -> None:
        series = 0.05 * np.arange(400) + _white_noise(400)
        _, p_level, _ = kpss_test(series, trend="c")
        _, p_trend, _ = kpss_test(series, trend="ct")
        assert p_level == 0.01
        assert p_trend == 0.10

    def test_automatic_lags_are_capped_below_the_sample_size(self) -> None:
        # The automatic bandwidth of this series is 7.2 (see the hand
        # computation above); it is cut to n - 1 = 2, the largest lag with
        # any cross products.
        series = np.array([0.1, -0.2, 0.3])
        assert kpss_test(series) == kpss_test(series, n_lags=2)
        assert kpss_test(series) != kpss_test(series, n_lags=1)

    def test_automatic_lags_grow_with_persistence(self) -> None:
        # The automatic statistic equals the fixed-lag statistic for exactly
        # one lag count; a persistent series needs a longer window than
        # white noise of the same length.
        def selected_lags(series: np.ndarray) -> int:
            automatic = kpss_test(series)[0]
            matches = [k for k in range(40) if kpss_test(series, n_lags=k)[0] == automatic]
            assert len(matches) == 1
            return matches[0]

        noise = _white_noise(300)
        persistent = np.empty(300)
        persistent[0] = noise[0]
        for t in range(1, 300):
            persistent[t] = 0.9 * persistent[t - 1] + noise[t]
        assert selected_lags(persistent) > selected_lags(noise)

    def test_statistic_does_not_depend_on_level_or_scale(self) -> None:
        series = _random_walk(80) * 0.25 + _white_noise(80, RNG_SEED + 2)
        statistic, p_value, at_bound = kpss_test(series, n_lags=3)
        shifted = kpss_test(series * 1024.0 + 4096.0, n_lags=3)
        assert shifted[0] == pytest.approx(statistic, rel=1e-10)
        assert shifted[1] == pytest.approx(p_value, rel=1e-10)
        assert shifted[2] is at_bound

    @pytest.mark.parametrize("series", [np.full(4, 0.1), np.full(252, 0.07), np.zeros(10)])
    @pytest.mark.parametrize("trend", ["c", "ct"])
    def test_constant_series_raises(self, series: np.ndarray, trend: str) -> None:
        # The mean of np.full(252, 0.07) is not exactly 0.07, so its
        # residuals are rounding residue rather than zeros.
        with pytest.raises(ValueError, match="rounding error"):
            kpss_test(series, trend=trend)

    @pytest.mark.parametrize(
        "series",
        [
            np.arange(10.0),
            3.0 - 0.25 * np.arange(40),
            # Accumulating a constant step leaves rounding residue of up to
            # n/2 units in the last place around the straight line.
            np.cumsum(np.full(5000, -0.999)),
        ],
    )
    def test_exactly_linear_series_raises_for_trend_stationarity(self, series: np.ndarray) -> None:
        with pytest.raises(ValueError, match="rounding error"):
            kpss_test(series, trend="ct")
        # About a level it is an ordinary trending series with a statistic.
        statistic, _, _ = kpss_test(series, trend="c")
        assert statistic > 0.0

    def test_series_with_undefined_automatic_bandwidth_raises(self) -> None:
        # Residuals (0, -1, 1): the variance 2/3 and twice the lag-1
        # autocovariance -1/3 cancel, so the bandwidth is a division by zero.
        series = np.array([2.0, 1.0, 3.0])
        with pytest.raises(ValueError, match="n_lags"):
            kpss_test(series)
        statistic, _, _ = kpss_test(series, n_lags=0)
        assert statistic == pytest.approx((1.0 / 9.0) / (2.0 / 3.0), rel=1e-12)

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            kpss_test(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="at least 4"):
            kpss_test(np.array([1.0, 3.0, 2.0]), trend="ct")

    def test_lag_count_out_of_range_raises(self) -> None:
        series = _white_noise(20)
        with pytest.raises(ValueError, match="n_lags"):
            kpss_test(series, n_lags=-1)
        with pytest.raises(ValueError, match="n_lags"):
            kpss_test(series, n_lags=20)
        statistic, _, _ = kpss_test(series, n_lags=19)
        assert math.isfinite(statistic)

    @pytest.mark.parametrize("trend", ["n", "ctt", "C", ""])
    def test_unknown_trend_raises(self, trend: str) -> None:
        with pytest.raises(ValueError, match="trend"):
            kpss_test(_white_noise(20), trend=trend)

    def test_two_dimensional_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            kpss_test(np.ones((10, 2)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_input_raises(self, bad: float) -> None:
        series = _white_noise(30)
        series[4] = bad
        with pytest.raises(ValueError, match="finite"):
            kpss_test(series)

    def test_non_contiguous_view_gives_the_same_result_as_its_copy(self) -> None:
        series = _random_walk(120)
        assert kpss_test(series[::2]) == kpss_test(series[::2].copy())

    def test_non_positive_long_run_variance_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Bartlett estimate is positive for any non-zero residuals in
        # exact arithmetic; a zero is forced here to reach the guard that
        # stands in front of the division.
        monkeypatch.setattr(
            diagnostics, "_bartlett_long_run_variance", lambda residuals, n_lags: 0.0
        )
        with pytest.raises(ValueError, match="long-run variance"):
            kpss_test(_white_noise(30), n_lags=2)

    def test_returns_python_floats_and_bool(self) -> None:
        statistic, p_value, at_bound = kpss_test(_white_noise(50))
        assert type(statistic) is float
        assert type(p_value) is float
        assert type(at_bound) is bool

    @pytest.mark.parametrize("trend", ["c", "ct"])
    def test_input_is_not_modified(self, trend: str) -> None:
        series = _random_walk(50)
        original = series.copy()
        kpss_test(series, trend=trend)
        np.testing.assert_array_equal(series, original)


class TestScaleInvarianceAtExtremeMagnitudes:
    """The statistics are ratios, so the units of the input cannot change them.

    The scales reach the ends of float64, where a naive sum of squares, or
    of the values themselves, overflows or underflows to zero.
    """

    @staticmethod
    def _walk(n: int = 300, seed: int = 11) -> np.ndarray:
        return np.cumsum(np.random.default_rng(seed).standard_normal(n))

    def test_jarque_bera_with_a_level_near_the_float_maximum(self) -> None:
        x = np.random.default_rng(5).standard_t(df=5, size=200)
        expected = jarque_bera_test(x)
        statistic, p_value = jarque_bera_test(x * 1e300 + 1e306)
        assert statistic == pytest.approx(expected[0], rel=1e-9)
        assert p_value == pytest.approx(expected[1], rel=1e-9)

    @pytest.mark.parametrize("scale", [1e160, 1e200, 1e300, 1e-170, 1e-200, 1e-300])
    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("n_lags", [None, 3])
    def test_kpss_statistic_does_not_depend_on_the_scale(
        self, scale: float, trend: str, n_lags: int | None
    ) -> None:
        series = self._walk()
        expected = kpss_test(series, n_lags, trend)
        statistic, p_value, at_bound = kpss_test(series * scale, n_lags, trend)
        assert statistic == pytest.approx(expected[0], rel=1e-9)
        assert p_value == pytest.approx(expected[1], rel=1e-9)
        assert at_bound == expected[2]


class TestBartlettLongRunVariance:
    def test_matches_the_definition_term_by_term(self) -> None:
        e = np.array([0.5, -1.0, 0.25, 2.0, -0.75])
        n, lags = e.size, 2
        gamma = [sum(e[t] * e[t - j] for t in range(j, n)) / n for j in range(lags + 1)]
        expected = gamma[0] + 2.0 * ((1.0 - 1.0 / 3.0) * gamma[1] + (1.0 - 2.0 / 3.0) * gamma[2])
        assert diagnostics._bartlett_long_run_variance(e, lags) == pytest.approx(
            expected, rel=1e-14
        )

    def test_zero_lags_is_the_mean_square(self) -> None:
        e = _white_noise(40)
        assert diagnostics._bartlett_long_run_variance(e, 0) == pytest.approx(
            float(np.mean(e**2)), rel=1e-14
        )
