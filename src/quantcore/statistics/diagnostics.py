"""Diagnostic tests for a return or residual series: normality, ARCH effects, stationarity.

Jarque-Bera normality test (Jarque & Bera 1980, 1987):
    JB = n/6 * (S^2 + K^2/4)
    S = m3 / m2^(3/2),  K = m4 / m2^2 - 3,  m_k = (1/n)*sum_t (x_t - mean(x))^k
    Under normality JB is asymptotically chi-squared with 2 degrees of
    freedom, whose survival function is exp(-JB/2).

ARCH-LM test for conditional heteroskedasticity (Engle 1982):
    u_t = e_t^2
    regress u_t on [1, u_{t-1}, ..., u_{t-m}] for t = m+1..n
    LM = (n - m) * R^2,  R^2 the centred R-squared of that regression
    Under no ARCH effects LM is asymptotically chi-squared with m degrees of
    freedom.

KPSS stationarity test (Kwiatkowski, Phillips, Schmidt & Shin 1992):
    e_t = residual of x_t regressed on a constant (level null) or on a
          constant and a linear trend (trend null)
    S_t = sum_{s<=t} e_s
    statistic = (1/n^2)*sum_t S_t^2 / s^2(l)
    s^2(l) = (1/n)*[sum_t e_t^2 + 2*sum_{j=1}^l w_j*sum_{t>j} e_t*e_{t-j}]
    w_j = 1 - j/(l+1)  (Bartlett kernel)
    The null hypothesis is stationarity, so a small p-value rejects it. The
    statistic's null distribution is not a standard one; p-values are
    interpolated in the paper's table of four upper-tail critical values.

References:
    Jarque, C.M. and Bera, A.K. (1980). "Efficient Tests for Normality,
    Homoscedasticity and Serial Independence of Regression Residuals."
    Economics Letters, 6(3), 255-259.
    Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of
    Observations and Regression Residuals." International Statistical
    Review, 55(2), 163-172.
    Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
    Estimates of the Variance of United Kingdom Inflation." Econometrica,
    50(4), 987-1007.
    Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992).
    "Testing the Null Hypothesis of Stationarity against the Alternative of
    a Unit Root." Journal of Econometrics, 54(1-3), 159-178.
    Hobijn, B., Franses, P.H. and Ooms, M. (2004). "Generalizations of the
    KPSS-test for Stationarity." Statistica Neerlandica, 58(4), 483-502.
    Newey, W.K. and West, K.D. (1987, 1994). See docs/REFERENCES.md.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from quantcore.statistics.regression import _ols

# ---------------------------------------------------------------------------
# Shared input checks
# ---------------------------------------------------------------------------


def _power_of_two_scale(magnitude: float) -> float:
    """Largest power of two not above `magnitude`, a positive finite float.

    Dividing by a power of two changes only the exponent of a float, so a
    series rescaled this way keeps every digit and a scale-free statistic
    computed from it is the same number as from the original units.
    """
    return math.ldexp(1.0, math.frexp(magnitude)[1] - 1)


def _validate_series(values: npt.NDArray[np.float64], name: str, min_observations: int) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if values.size < min_observations:
        raise ValueError(f"{name} must have at least {min_observations} observations")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")


# ---------------------------------------------------------------------------
# Jarque-Bera normality test
# ---------------------------------------------------------------------------


def _validate_jarque_bera_inputs(series: npt.NDArray[np.float64]) -> None:
    _validate_series(series, "series", min_observations=2)
    # A range test, not a variance test: the mean of a constant array is
    # often not exactly that constant, which leaves a tiny non-zero variance.
    if np.ptp(series) == 0.0:
        raise ValueError("series is constant; the Jarque-Bera statistic is undefined")


def jarque_bera_test(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Jarque-Bera test of the null hypothesis that a sample is normally distributed.

    The statistic is `n/6 * (S^2 + K^2/4)` with `S` the sample skewness and
    `K` the sample excess kurtosis, both from population (divide-by-n)
    central moments with no small-sample correction. It does not depend on
    the location or scale of the series. Same definition as
    `scipy.stats.jarque_bera`.

    Args:
        series: The sample to test, shape (n,), n >= 2, all values finite.

    Returns:
        Tuple (statistic, p_value). `p_value` is the chi-squared(2)
        survival function of the statistic, `exp(-statistic / 2)`; a small
        value rejects normality. The chi-squared distribution is the
        large-sample limit: with two observations the statistic is 1/3
        whatever their values, and for a few dozen the test is known to
        under-reject.

    Raises:
        ValueError: `series` is not 1-D, has fewer than 2 observations,
            contains NaN or infinite values, or is constant.

    References:
        Jarque, C.M. and Bera, A.K. (1980). "Efficient Tests for Normality,
        Homoscedasticity and Serial Independence of Regression Residuals."
        *Economics Letters*, 6(3), 255-259. Jarque, C.M. and Bera, A.K.
        (1987). "A Test for Normality of Observations and Regression
        Residuals." *International Statistical Review*, 55(2), 163-172.
        See docs/REFERENCES.md.
    """
    _validate_jarque_bera_inputs(series)
    return _jarque_bera_test(series)


def _jarque_bera_test(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    n = series.size
    # The statistic is scale-free. Bringing the values to order one before the
    # mean is taken keeps their sum from overflowing; unit maximum of the
    # deviations keeps their fourth powers from overflowing or underflowing.
    scaled = series / _power_of_two_scale(float(np.max(np.abs(series))))
    deviations = scaled - scaled.mean()
    # The mean of the deviations is the rounding error of the first mean.
    # Removing it matters when the mean is large against the spread: at a
    # ratio of 1e5 the third moment otherwise loses five digits.
    deviations = deviations - deviations.mean()
    deviations = deviations / np.max(np.abs(deviations))

    m2 = float(np.mean(deviations**2))
    skewness = float(np.mean(deviations**3)) / m2**1.5
    excess_kurtosis = float(np.mean(deviations**4)) / m2**2 - 3.0
    statistic = n / 6.0 * (skewness**2 + excess_kurtosis**2 / 4.0)
    return statistic, float(chi2.sf(statistic, df=2))


# ---------------------------------------------------------------------------
# ARCH-LM test
# ---------------------------------------------------------------------------

# Default lag count min(10, n // 5): at most ten lags, and at least five
# observations per lag.
_ARCH_LM_MAX_DEFAULT_LAGS = 10
_ARCH_LM_OBSERVATIONS_PER_DEFAULT_LAG = 5


def _validate_arch_lm_inputs(residuals: npt.NDArray[np.float64], n_lags: int | None) -> None:
    if n_lags is None:
        # With fewer observations the default lag count is zero.
        min_observations = _ARCH_LM_OBSERVATIONS_PER_DEFAULT_LAG
    else:
        if n_lags < 1:
            raise ValueError("n_lags must be a positive integer")
        # More regression rows (n - n_lags) than coefficients (n_lags + 1):
        # with as many rows as coefficients the fit is exact and R^2 is 1
        # whatever the data.
        min_observations = 2 * n_lags + 2
    _validate_series(residuals, "residuals", min_observations)


def arch_lm_test(
    residuals: npt.NDArray[np.float64], n_lags: int | None = None
) -> tuple[float, float]:
    """Engle's Lagrange-multiplier test for autoregressive conditional heteroskedasticity.

    Regresses the squared series on a constant and its own first `n_lags`
    lags and returns `(n - n_lags) * R^2`, `n - n_lags` being the number of
    rows in that regression. The null hypothesis is no ARCH effect: the
    squared values are not predictable from their past.

    The input is squared as given. Nothing is demeaned first, so pass
    residuals (or returns whose mean is negligible beside their spread);
    the constant in the regression centres the squares, not the series.
    The result does not depend on the scale of `residuals`. Same definition
    as `statsmodels.stats.diagnostic.het_arch`.

    Args:
        residuals: The series to test, shape (n,), all values finite.
        n_lags: Number of lagged squares in the regression, at least 1,
            with `n >= 2 * n_lags + 2` so the regression has more rows
            than coefficients. `None` (default) uses `min(10, n // 5)`,
            the default of `het_arch`, and needs `n >= 5`.

    Returns:
        Tuple (statistic, p_value). `statistic` is non-negative; `p_value`
        is the chi-squared(`n_lags`) survival function of the statistic,
        and a small value rejects the null of no ARCH effect.

    Raises:
        ValueError: `residuals` is not 1-D, is too short for the lag count,
            or contains NaN or infinite values; `n_lags` is below 1; or the
            squared residuals are constant over the regression sample
            (observations `n_lags + 1` onwards), which leaves nothing to
            explain. A series of equal magnitudes and alternating signs is
            such a case.

    References:
        Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity
        with Estimates of the Variance of United Kingdom Inflation."
        *Econometrica*, 50(4), 987-1007. See docs/REFERENCES.md.
    """
    _validate_arch_lm_inputs(residuals, n_lags)
    if n_lags is None:
        n_lags = min(
            _ARCH_LM_MAX_DEFAULT_LAGS, residuals.size // _ARCH_LM_OBSERVATIONS_PER_DEFAULT_LAG
        )
    return _arch_lm_test(residuals, n_lags)


def _arch_lm_test(residuals: npt.NDArray[np.float64], n_lags: int) -> tuple[float, float]:
    n = residuals.size
    n_rows = n - n_lags
    constant_message = (
        "squared residuals are constant over the regression sample; "
        "the ARCH-LM statistic is undefined"
    )

    # R^2 is unchanged by rescaling the residuals, but the least-squares
    # solver is not: beside a column of ones, squares of order 1e-16 or
    # 1e+16 are treated as rank-deficient and dropped. Unit maximum keeps
    # every column within [0, 1].
    scale = float(np.max(np.abs(residuals)))
    if scale == 0.0:
        raise ValueError(constant_message)
    squared = (residuals / scale) ** 2
    dependent = squared[n_lags:]
    if np.ptp(dependent) == 0.0:
        raise ValueError(constant_message)

    design = np.empty((n_rows, n_lags + 1), dtype=np.float64)
    design[:, 0] = 1.0
    for lag in range(1, n_lags + 1):
        design[:, lag] = squared[n_lags - lag : n - lag]
    _, _, r_squared = _ols(dependent, design)

    # R^2 = 1 - SSR/SST can round a hair below zero when the lags explain
    # nothing.
    statistic = max(n_rows * r_squared, 0.0)
    return statistic, float(chi2.sf(statistic, df=n_lags))


# ---------------------------------------------------------------------------
# KPSS stationarity test
# ---------------------------------------------------------------------------

# Kwiatkowski, Phillips, Schmidt & Shin (1992), Table 1: upper-tail critical
# values of the statistic at the significance levels below, for the level
# ("c") and the trend ("ct") null hypothesis.
_KPSS_SIGNIFICANCE_LEVELS: npt.NDArray[np.float64] = np.array([0.10, 0.05, 0.025, 0.01])
_KPSS_CRITICAL_VALUES: dict[str, npt.NDArray[np.float64]] = {
    "c": np.array([0.347, 0.463, 0.574, 0.739]),
    "ct": np.array([0.119, 0.146, 0.176, 0.216]),
}
# Deterministic terms removed under each null; the series needs two more
# observations than that for the statistic to vary with the data at all.
_KPSS_DETERMINISTIC_TERMS: dict[str, int] = {"c": 1, "ct": 2}
# Bartlett-kernel constant of the Newey & West (1994) plug-in bandwidth.
_KPSS_BANDWIDTH_CONSTANT = 1.1447


def _validate_kpss_inputs(series: npt.NDArray[np.float64], n_lags: int | None, trend: str) -> None:
    if trend not in _KPSS_CRITICAL_VALUES:
        raise ValueError(f"trend must be one of {tuple(_KPSS_CRITICAL_VALUES)}")
    _validate_series(series, "series", _KPSS_DETERMINISTIC_TERMS[trend] + 2)
    if n_lags is not None and not 0 <= n_lags < series.size:
        raise ValueError("n_lags must satisfy 0 <= n_lags < len(series)")


def kpss_test(
    series: npt.NDArray[np.float64], n_lags: int | None = None, trend: str = "c"
) -> tuple[float, float, bool]:
    """KPSS test of the null hypothesis that a series is stationary.

    The roles of the hypotheses are the reverse of `adf_test`'s: here the
    null is stationarity (about a level, or about a linear trend) and the
    alternative is a unit root, so a small p-value is evidence against
    stationarity.

    The statistic is the sum of squared partial sums of the residuals,
    divided by `n^2` and by a Bartlett-kernel estimate of their long-run
    variance (see the module docstring). Statistic, p-value and automatic
    lag count are those of `statsmodels.tsa.stattools.kpss(series,
    regression=trend, nlags="auto")`.

    Args:
        series: The series to test, shape (n,), all values finite; n >= 3
            for `trend="c"` and n >= 4 for `trend="ct"`.
        n_lags: Number of autocovariance lags `l` in the long-run variance,
            `0 <= n_lags < n`. `None` (default) chooses it from the data
            with the Bartlett-kernel plug-in bandwidth of Newey & West
            (1994), as Hobijn, Franses & Ooms (2004) apply it to this test:
            `l = min(n - 1, int(g * n^(1/3)))`,
            `g = 1.1447 * ((s1/s0)^2)^(1/3)`,
            `s0 = c_0 + 2*sum_{j=1}^m c_j`, `s1 = 2*sum_{j=1}^m j*c_j`,
            `c_j = (1/n)*sum_{t>j} e_t*e_{t-j}`, over the first
            `m = int(n^(2/9))` autocovariances. `int` truncates. A
            persistent series gets a long window (dozens of lags for a
            random walk of a few thousand points), white noise a short one.
        trend: The null hypothesis: `"c"` (default), stationary about a
            constant level, tested on `series - mean(series)`; or `"ct"`,
            stationary about a linear trend, tested on the residuals of an
            OLS regression of `series` on a constant and `t = 1..n`.

    Returns:
        Tuple (statistic, p_value, p_value_at_bound).

        `p_value` is interpolated linearly between the four published
        critical values (significance 0.10, 0.05, 0.025, 0.01: statistics
        0.347, 0.463, 0.574, 0.739 for `"c"`; 0.119, 0.146, 0.176, 0.216
        for `"ct"`), so it always lies in [0.01, 0.10].

        `p_value_at_bound` is `True` when the statistic is at or outside an
        end of that table. `p_value` is then exactly 0.10 or 0.01 and means
        "at least 0.10" or "at most 0.01": the true value is not known more
        closely than that.

    Raises:
        ValueError: `series` is not 1-D, is too short, or contains NaN or
            infinite values; `trend` or `n_lags` is out of range; the
            residuals are no larger than rounding error (a constant series,
            or under `"ct"` an exactly linear one, has nothing left to
            test); or `n_lags` is `None` and `s0` above is exactly zero, in
            which case a lag count has to be passed.

    Notes:
        The partial sums accumulate any error in the removed level, so the
        statistic loses accuracy as the level grows against the spread: its
        relative error is of order `sqrt(n) * eps * |mean| / std`, and for
        `"ct"` `n * eps * max|series| / std` of the residuals.

    References:
        Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992).
        "Testing the Null Hypothesis of Stationarity against the
        Alternative of a Unit Root." *Journal of Econometrics*, 54(1-3),
        159-178. Hobijn, B., Franses, P.H. and Ooms, M. (2004).
        "Generalizations of the KPSS-test for Stationarity." *Statistica
        Neerlandica*, 58(4), 483-502. Newey, W.K. and West, K.D. (1994).
        "Automatic Lag Selection in Covariance Matrix Estimation." *Review
        of Economic Studies*, 61(4), 631-653. See docs/REFERENCES.md.
    """
    _validate_kpss_inputs(series, n_lags, trend)
    return _kpss_test(series, n_lags, trend)


def _kpss_residuals(series: npt.NDArray[np.float64], trend: str) -> npt.NDArray[np.float64]:
    """Residuals of `series` about its mean ("c") or its OLS linear trend ("ct").

    The residuals are returned divided by a power of two that brings their
    largest magnitude into [1, 2). The KPSS statistic and the automatic lag
    count are ratios that the units cancel out of, and order-one residuals
    keep the sums of squares and cross products they are built from inside
    float64 whatever the units of `series`.

    Raises:
        ValueError: No residual exceeds the rounding error of the series,
            taken as `4 * n * eps * max|series|` (the worst case for a sum
            of `n` terms): the statistic would be a ratio of rounding
            residue.
    """
    n = series.size
    largest = float(np.max(np.abs(series)))
    if largest == 0.0:
        raise ValueError(
            "series is constant to within rounding error; the KPSS statistic is undefined"
        )
    scale = _power_of_two_scale(largest)
    scaled = series / scale
    if trend == "c":
        residuals: npt.NDArray[np.float64] = scaled - scaled.mean()
    else:
        time = np.arange(1, n + 1, dtype=np.float64)
        _, residuals, _ = _ols(scaled, np.column_stack((np.ones(n), time)))

    largest_residual = float(np.max(np.abs(residuals)))
    resolution = 4.0 * n * float(np.finfo(np.float64).eps) * (largest / scale)
    if largest_residual <= resolution:
        shape = "constant" if trend == "c" else "a linear trend"
        raise ValueError(
            f"series is {shape} to within rounding error; the KPSS statistic is undefined"
        )
    return residuals / _power_of_two_scale(largest_residual)


def _kpss_automatic_lags(residuals: npt.NDArray[np.float64]) -> int:
    """Data-dependent lag count for the Bartlett-kernel long-run variance.

    See `kpss_test`'s `n_lags` for the formula.

    Raises:
        ValueError: `s0` is exactly zero, so the bandwidth is undefined.
    """
    n = residuals.size
    n_autocovariances = int(n ** (2.0 / 9.0))
    s0 = float(np.sum(residuals**2)) / n
    s1 = 0.0
    for j in range(1, n_autocovariances + 1):
        autocovariance = float(np.dot(residuals[j:], residuals[: n - j])) / n
        s0 += 2.0 * autocovariance
        s1 += 2.0 * j * autocovariance
    if s0 == 0.0:
        raise ValueError(
            "the automatic lag count is undefined for this series (its leading "
            "autocovariances sum to zero); pass n_lags explicitly"
        )
    ratio = s1 / s0
    bandwidth = _KPSS_BANDWIDTH_CONSTANT * (ratio * ratio) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    # Lags beyond n - 1 have no cross products.
    return n - 1 if bandwidth >= n - 1 else int(bandwidth)


def _bartlett_long_run_variance(residuals: npt.NDArray[np.float64], n_lags: int) -> float:
    """Bartlett-kernel estimate `s^2(l)` of the long-run variance of `residuals`.

    `s^2(l) = gamma_0 + 2 * sum_{j=1}^{l} (1 - j / (l + 1)) * gamma_j` with the
    autocovariances `gamma_j = (1 / n) * sum_t e_t * e_{t-j}`.
    """
    n = residuals.size
    total = float(np.dot(residuals, residuals))
    for j in range(1, n_lags + 1):
        weight = 1.0 - j / (n_lags + 1.0)
        total += 2.0 * weight * float(np.dot(residuals[j:], residuals[: n - j]))
    return total / n


def _kpss_test(
    series: npt.NDArray[np.float64], n_lags: int | None, trend: str
) -> tuple[float, float, bool]:
    n = series.size
    residuals = _kpss_residuals(series, trend)
    lags = _kpss_automatic_lags(residuals) if n_lags is None else n_lags

    eta = float(np.sum(np.cumsum(residuals) ** 2)) / n**2
    long_run_variance = _bartlett_long_run_variance(residuals, lags)
    if not long_run_variance > 0.0:
        raise ValueError(
            "long-run variance estimate is not positive; the KPSS statistic is undefined"
        )
    statistic = eta / long_run_variance

    critical_values = _KPSS_CRITICAL_VALUES[trend]
    p_value = float(np.interp(statistic, critical_values, _KPSS_SIGNIFICANCE_LEVELS))
    p_value_at_bound = bool(statistic <= critical_values[0] or statistic >= critical_values[-1])
    return statistic, p_value, p_value_at_bound
