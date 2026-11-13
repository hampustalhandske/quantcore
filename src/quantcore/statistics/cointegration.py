"""Cointegration testing and mean-reversion statistics for pairs/spread trading.

Engle-Granger (1987) two-step:
    Step 1 — estimate cointegrating vector via OLS: y = alpha + beta*x + epsilon
    Step 2 — ADF test on residuals epsilon_hat, with no constant in the
             ADF regression (epsilon_hat is mean-zero by construction);
             null hypothesis: epsilon_hat has a unit root (no cointegration)

ADF test statistic (Dickey & Fuller 1979):
    Delta(y_t) = alpha + rho*y_{t-1} + sum_{j=1}^p gamma_j*Delta(y_{t-j}) + e_t
    t-stat for rho=0 is the ADF statistic. p-values come from MacKinnon's
    (1994) response-surface approximation, parameterized by N, the number of
    I(1) series in the regression the tested residuals came from: a direct
    unit-root test on one series (`adf_test`) uses N=1, the Engle-Granger
    residual-based test (`engle_granger_test`) uses N=2. Both tests are
    computed by `statsmodels` (`adfuller`, `coint`).

OU half-life (Avellaneda & Lee 2010):
    Delta(spread_t) = a * spread_{t-1} + noise  (fit via OLS)
    level-form AR(1) coefficient: phi = 1 + a
    half_life = -ln(2) / ln(phi)
    A slope that is not negative by more than its rounding error raises ValueError.

Z-score (rolling):
    z_t = (spread_t - rolling_mean_t) / rolling_std_t

References:
    Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error
    Correction: Representation, Estimation, and Testing." Econometrica.
    Avellaneda, M. and Lee, J.-H. (2010). "Statistical Arbitrage in the U.S.
    Equities Market." Quantitative Finance.
    Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
    Autoregressive Time Series with a Unit Root." JASA.
    MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution Functions for
    Unit-Root and Cointegration Tests." Journal of Business & Economic
    Statistics, 12(2), 167-176.
    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# ADF unit-root test
# ---------------------------------------------------------------------------

_ADF_REGRESSION_CASES = ("n", "c", "ct")
_ADF_AUTOLAGS = ("aic", "bic", "t-stat")


def _validate_adf_inputs(
    series: npt.NDArray[np.float64], max_lags: int, trend: str, autolag: str | None
) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    if trend not in _ADF_REGRESSION_CASES:
        raise ValueError(f"trend must be one of {_ADF_REGRESSION_CASES}")
    if autolag is not None and autolag not in _ADF_AUTOLAGS:
        raise ValueError('autolag must be "aic", "bic", "t-stat", or None')
    min_length = max_lags + 5
    if series.size < min_length:
        raise ValueError(
            f"series must have at least {min_length} observations for max_lags={max_lags}"
        )


def adf_test(
    series: npt.NDArray[np.float64],
    max_lags: int = 1,
    trend: str = "c",
    autolag: str | None = None,
) -> tuple[float, float]:
    """Augmented Dickey-Fuller test for a unit root.

    Args:
        series: The time series to test.
        max_lags: If `autolag` is None (default), the exact number of
            lagged differences included in the regression. If `autolag` is
            set, the maximum lag to search over (0, 1, ..., max_lags).
        trend: Deterministic terms in the regression: "n" (none), "c"
            (constant, the default — matches the original behavior), or
            "ct" (constant and linear trend).
        autolag: Lag-length selection criterion: `None` (default; use
            `max_lags` lags exactly, unchanged from the original behavior),
            `"aic"`, `"bic"` (minimize the corresponding information
            criterion over 0..max_lags, holding the regression sample fixed
            across candidates so the criteria are comparable), or
            `"t-stat"` (Hall's (1994) general-to-specific procedure: start
            at max_lags and drop the highest lag until its own t-statistic
            is significant at a 5%-sized test, `|t| >= 1.6448536...` =
            `scipy.stats.norm.ppf(0.95)`; matches
            `statsmodels.tsa.stattools.adfuller`'s `autolag="t-stat"`).

    Returns:
        Tuple (adf_statistic, p_value). p_value is computed via MacKinnon's
        (1994) response-surface approximation to the Dickey-Fuller
        distribution for the given `trend` case (N=1: `series` is tested
        directly, not a residual from an estimated cointegrating
        regression). It lies in [0, 1] and equals the p-value
        `statsmodels.tsa.stattools.adfuller` reports for the same
        statistic: many orders of magnitude below 1e-4 for a strongly
        stationary series, exactly 0.0 for a statistic below the range
        MacKinnon tabulated (-19.04, -18.83 and -16.18 for "n", "c" and
        "ct"), and exactly 1.0 for one above it (2.74 for "c", 0.70 for
        "ct"; the "n" surface has no upper limit). Guard against 0.0
        before taking its logarithm or dividing by it.

    References:
        Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the
        Estimators for Autoregressive Time Series with a Unit Root."
        *Journal of the American Statistical Association*, 74(366),
        427-431. MacKinnon, J.G. (1994). "Approximate Asymptotic
        Distribution Functions for Unit-Root and Cointegration Tests."
        *Journal of Business & Economic Statistics*, 12(2), 167-176.
        See docs/REFERENCES.md.
    """
    _validate_adf_inputs(series, max_lags, trend, autolag)
    return _adf_test(series, max_lags, trend, autolag)


def _adf_test(
    series: npt.NDArray[np.float64], max_lags: int, trend: str, autolag: str | None
) -> tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller

    statistic, p_value, *_ = adfuller(
        series,
        maxlag=max_lags,
        regression=trend,
        autolag=autolag,
        store=False,
        regresults=False,
        result_object=False,
    )
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        raise ValueError("the ADF statistic is undefined for this series (it may be constant)")
    return float(statistic), float(p_value)


# ---------------------------------------------------------------------------
# Engle-Granger two-step cointegration test
# ---------------------------------------------------------------------------


def _validate_engle_granger_inputs(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
    if y.size == 0 or x.size == 0:
        raise ValueError("y and x must be non-empty")
    if y.size != x.size:
        raise ValueError("y and x must have the same length")
    if y.size < 10:
        raise ValueError("y and x must have at least 10 observations")


def engle_granger_test(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Engle-Granger two-step cointegration test.

    Args:
        y: Dependent series.
        x: Candidate cointegrating series.

    Returns:
        Tuple (beta, adf_statistic, p_value): the OLS cointegrating
        coefficient (step 1 fits `y = alpha + beta*x + u` with an
        intercept) and the ADF statistic on the OLS residuals, with its
        p-value from MacKinnon's (1994) response surface for N=2 I(1)
        series (y and x) — the Engle-Granger distribution, not the plain
        (N=1) Dickey-Fuller distribution `adf_test` on a raw series uses.
        Residuals from an estimated cointegrating regression have already
        had one degree of unit-root freedom absorbed by the OLS fit, so
        their null distribution lies further left than a raw series' does;
        using the N=1 distribution here would overstate significance.

        The step-2 ADF regression uses one lagged difference and no
        constant (the residuals are mean-zero by construction of step 1),
        and the p-value is exactly 0.0 / 1.0 for a statistic beyond
        MacKinnon's tabulated range. Both conventions match
        `statsmodels.tsa.stattools.coint(y, x, trend="c", maxlag=1,
        autolag=None)`, which computes them.
    """
    _validate_engle_granger_inputs(y, x)
    return _engle_granger_test(y, x)


def _engle_granger_test(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    from statsmodels.tsa.stattools import coint

    design = np.column_stack([np.ones_like(x), x])
    beta = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
    # Step 2 is `coint`'s augmented Engle-Granger test: ADF on the residuals
    # of the constant-including regression, one lagged difference, no
    # deterministic terms, p-value from the N=2 surface.
    statistic, p_value, _ = coint(
        y,
        x,
        trend="c",
        method="aeg",
        maxlag=1,
        autolag=None,
        return_results=None,
    )
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        raise ValueError("the Engle-Granger statistic is undefined for these series")
    return beta, float(statistic), float(p_value)


# ---------------------------------------------------------------------------
# OU half-life
# ---------------------------------------------------------------------------


def _validate_ou_half_life_inputs(spread: npt.NDArray[np.float64]) -> None:
    if spread.size == 0:
        raise ValueError("spread must be non-empty")
    if spread.size < 5:
        raise ValueError("spread must have at least 5 observations")


def ou_half_life(spread: npt.NDArray[np.float64]) -> float:
    """Estimate the OU mean-reversion half-life of a spread series via AR(1) OLS.

    Args:
        spread: The spread (or residual) series.

    Returns:
        Half-life of mean reversion, in the same time units as `spread`.

    Raises:
        ValueError: If the fitted slope `a` does not show mean reversion: `a >= 0`, or
            `1 + a <= 0`, or the slope is not negative by more than its rounding error. The
            last case covers a series whose slope is exactly 0 in exact arithmetic, such as a
            straight line, where the computed slope is rounding noise of either sign. It also
            covers a series whose level does not vary before the last observation.
    """
    _validate_ou_half_life_inputs(spread)
    return _ou_half_life(spread)


def _slope_is_resolvably_negative(
    level: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> bool:
    """Whether the AR(1) slope of `delta` on `level` is negative beyond rounding error.

    The slope is computed in centred form, `sum(c * delta) / sum(c**2)` with `c = level -
    mean(level)`. Its worst-case floating-point error is at most
    `eps * (n * sum(|c * delta|) + 2 * max(|level|) * sum(|delta|)) / sum(c**2)`. The first
    term bounds the rounding of the dot product, the second the rounding of `c`. Returns False
    when `level` is constant, where the slope is undefined.
    """
    centred = level - level.mean()
    sum_sq = float(centred @ centred)
    if sum_sq == 0.0:
        return False
    slope = float(centred @ delta) / sum_sq
    abs_delta = np.abs(delta)
    rounding = float(
        np.finfo(np.float64).eps
        * (
            level.size * float(np.abs(centred) @ abs_delta)
            + 2.0 * float(np.max(np.abs(level))) * float(abs_delta.sum())
        )
        / sum_sq
    )
    return slope < -rounding


def _ou_half_life(spread: npt.NDArray[np.float64]) -> float:
    delta = spread[1:] - spread[:-1]
    level = spread[:-1]
    x_reg = np.column_stack([np.ones_like(level), level])
    beta, _, _, _ = np.linalg.lstsq(x_reg, delta, rcond=None)
    a = float(beta[1])
    level_coef = 1.0 + a
    if a >= 0.0 or level_coef <= 0.0 or not _slope_is_resolvably_negative(level, delta):
        raise ValueError("estimated series does not exhibit mean reversion (phi >= 0, diverging)")
    return float(-np.log(2.0) / np.log(level_coef))


# ---------------------------------------------------------------------------
# Rolling spread z-score
# ---------------------------------------------------------------------------


def _validate_spread_zscore_inputs(spread: npt.NDArray[np.float64], window: int) -> None:
    if spread.size == 0:
        raise ValueError("spread must be non-empty")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > spread.size:
        raise ValueError("window must not exceed the length of spread")


def spread_zscore(spread: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Rolling z-score of a spread series over a trailing window.

    Args:
        spread: The spread series.
        window: Lookback window length.

    Returns:
        Array of shape (T,); the first `window - 1` entries are NaN.
    """
    _validate_spread_zscore_inputs(spread, window)
    return _spread_zscore(spread, window)


def _spread_zscore(spread: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    n = spread.size
    z = np.full(n, np.nan, dtype=np.float64)
    for t in range(window - 1, n):
        segment = spread[t - window + 1 : t + 1]
        mean = segment.mean()
        std = segment.std(ddof=1)
        if std > 0.0:
            z[t] = (spread[t] - mean) / std
    return z
