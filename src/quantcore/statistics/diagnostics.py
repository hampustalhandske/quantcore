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
    freedom. Computed by `statsmodels.stats.diagnostic.het_arch`.

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
    Computed by `statsmodels.tsa.stattools.kpss`.

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
import warnings

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

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
    from statsmodels.stats.diagnostic import het_arch

    constant_message = (
        "squared residuals are constant over the regression sample; "
        "the ARCH-LM statistic is undefined"
    )

    # The statistic does not depend on the scale of the residuals. Unit
    # maximum keeps their squares within [0, 1], clear of underflow and
    # overflow.
    scale = float(np.max(np.abs(residuals)))
    if scale == 0.0:
        raise ValueError(constant_message)
    scaled = residuals / scale
    if np.ptp(scaled[n_lags:] ** 2) == 0.0:
        raise ValueError(constant_message)

    statistic, p_value, _, _ = het_arch(
        scaled, nlags=n_lags, store=False, ddof=0, result_object=False
    )
    return max(float(statistic), 0.0), float(p_value)


# ---------------------------------------------------------------------------
# KPSS stationarity test
# ---------------------------------------------------------------------------

_KPSS_TRENDS = ("c", "ct")
# Deterministic terms removed under each null; the series needs two more
# observations than that for the statistic to vary with the data at all.
_KPSS_DETERMINISTIC_TERMS: dict[str, int] = {"c": 1, "ct": 2}
# Relative size below which the sum of the leading autocovariances counts as
# zero when the automatic lag count is chosen.
_KPSS_BANDWIDTH_TOLERANCE = 1e-12


def _validate_kpss_inputs(series: npt.NDArray[np.float64], n_lags: int | None, trend: str) -> None:
    if trend not in _KPSS_TRENDS:
        raise ValueError(f"trend must be one of {_KPSS_TRENDS}")
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


def _check_kpss_is_defined(scaled: npt.NDArray[np.float64], n_lags: int | None, trend: str) -> None:
    """Raise `ValueError` for the series on which the KPSS statistic is undefined.

    `scaled` is the series divided by a power of two. The statistic is undefined
    when no residual about the null's deterministic terms exceeds the rounding
    error of the series (a constant series, or under `"ct"` an exactly linear
    one), and, when `n_lags` is `None`, when the automatic bandwidth is a
    division by zero. These are checked here so the contract does not depend on
    what the installed statsmodels does with them.
    """
    n = scaled.size
    largest = float(np.max(np.abs(scaled)))
    if largest == 0.0:
        raise ValueError(
            "series is constant to within rounding error; the KPSS statistic is undefined"
        )
    if trend == "c":
        residuals = scaled - scaled.mean()
    else:
        design = np.column_stack((np.ones(n), np.arange(1, n + 1, dtype=np.float64)))
        residuals = scaled - design @ np.linalg.lstsq(design, scaled, rcond=None)[0]
    largest_residual = float(np.max(np.abs(residuals)))
    # Worst-case rounding error of a sum of n terms.
    if largest_residual <= 4.0 * n * float(np.finfo(np.float64).eps) * largest:
        shape = "constant" if trend == "c" else "a linear trend"
        raise ValueError(
            f"series is {shape} to within rounding error; the KPSS statistic is undefined"
        )
    if n_lags is None:
        e = residuals / _power_of_two_scale(largest_residual)
        gamma0 = float(np.sum(e**2)) / n
        s0 = gamma0
        for j in range(1, int(n ** (2.0 / 9.0)) + 1):
            s0 += 2.0 * float(np.dot(e[j:], e[: n - j])) / n
        # s0 is a sum of terms of size gamma0; a result within rounding of
        # zero relative to gamma0 is zero, and the bandwidth ratio s1/s0 would
        # be rounding residue.
        if abs(s0) <= _KPSS_BANDWIDTH_TOLERANCE * gamma0:
            raise ValueError(
                "the automatic lag count is undefined for this series (its leading "
                "autocovariances sum to zero to within rounding); pass n_lags explicitly"
            )


def _kpss_test(
    series: npt.NDArray[np.float64], n_lags: int | None, trend: str
) -> tuple[float, float, bool]:
    from statsmodels.tsa.stattools import kpss

    # The statistic and the automatic lag count are ratios from which the
    # units cancel; a power-of-two rescaling keeps every digit and brings the
    # sums of squares into range.
    scaled = series / _power_of_two_scale(float(np.max(np.abs(series))))
    _check_kpss_is_defined(scaled, n_lags, trend)
    with warnings.catch_warnings():
        # statsmodels warns when the statistic lies outside the table; the
        # caller gets that information as `p_value_at_bound`.
        warnings.simplefilter("ignore")
        statistic, p_value, _, critical_values = kpss(
            scaled,
            regression=trend,
            nlags="auto" if n_lags is None else n_lags,
            store=False,
            result_object=False,
        )
    if not (math.isfinite(statistic) and math.isfinite(p_value)):
        raise ValueError(
            "the KPSS statistic is undefined for this series (no variation left "
            "after removing the deterministic terms, or a zero long-run variance); "
            "pass n_lags explicitly"
        )
    p_value_at_bound = bool(
        statistic <= critical_values["10%"] or statistic >= critical_values["1%"]
    )
    return float(statistic), float(p_value), p_value_at_bound
