"""Cointegration testing and mean-reversion statistics for pairs/spread trading.

Engle-Granger (1987) two-step:
    Step 1 — estimate cointegrating vector via OLS: y = beta*x + epsilon
    Step 2 — ADF test on residuals epsilon_hat, with no constant in the
             ADF regression (epsilon_hat is mean-zero by construction);
             null hypothesis: epsilon_hat has a unit root (no cointegration)

ADF test statistic (Dickey & Fuller 1979):
    Delta(y_t) = alpha + rho*y_{t-1} + sum_{j=1}^p gamma_j*Delta(y_{t-j}) + e_t
    t-stat for rho=0 is the ADF statistic ("constant, no trend" specification,
    case "c" in the Dickey-Fuller/MacKinnon literature). p-values come from
    MacKinnon's (1994) response-surface approximation for case "c",
    parameterized by N, the number of I(1) series in the regression the
    tested residuals came from — not a Student-t reference distribution, and
    not a single fixed distribution regardless of N. A direct unit-root test
    on one series (`adf_test`) uses N=1 (the ordinary Dickey-Fuller
    distribution); the Engle-Granger residual-based test (`engle_granger_test`)
    uses N=2, since its residuals come from a 2-variable cointegrating
    regression and their null distribution lies further left than N=1's. See
    `_mackinnon_pvalue` for the response-surface coefficients and the
    p-value of a statistic outside their tabulated range.

OU half-life (Avellaneda & Lee 2010):
    Delta(spread_t) = a * spread_{t-1} + noise  (fit via OLS)
    level-form AR(1) coefficient: phi = 1 + a
    half_life = -ln(2) / ln(phi)

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
from scipy.stats import norm

# ---------------------------------------------------------------------------
# ADF / Engle-Granger unit-root p-values (MacKinnon 1994 response surface)
# ---------------------------------------------------------------------------

# MacKinnon (1994): response-surface approximation to the
# asymptotic distribution of the (A)DF tau statistic, indexed by N = the
# number of I(1) variables in the regression the residuals came from (N=1
# for a plain unit-root test on a single series; N=2, 3, ... for the
# Engle-Granger residual-based test on a cointegrating regression with N
# I(1) series). This is *not* the ordinary Dickey-Fuller distribution
# (N=1 only) applied uniformly regardless of N: the Engle-Granger residual
# distribution shifts left as N grows, because the cointegrating
# regression itself removes some of the residual's unit root before the
# ADF regression ever sees it. Separate tables per deterministic case
# ("n" no constant/no trend, "c" constant only, "ct" constant and linear
# trend), since the tau statistic's null distribution depends on which
# deterministic terms are included in the regression.
#
# Index 0 below is N=1, index 1 is N=2, etc. (N=1..6, the range MacKinnon
# tabulated coefficients for).
#
# Every table equals its counterpart in `statsmodels.tsa.adfvalues`
# (`tests/oracle/statistics/test_cointegration_oracle.py` compares the two
# surfaces on a grid of statistics). The small-p coefficients and minimum
# statistics for "c" and "ct" also equal Table 4 of the article's
# working-paper version (Queen's Economics Department Working Paper 861) to
# the digits printed there.
_MACKINNON_TAU_MAX: dict[str, npt.NDArray[np.float64]] = {
    "n": np.array([np.inf, 1.51, 0.86, 0.88, 1.05, 1.24]),
    "c": np.array([2.74, 0.92, 0.55, 0.61, 0.79, 1.00]),
    "ct": np.array([0.70, 0.63, 0.71, 0.93, 1.19, 1.42]),
}
_MACKINNON_TAU_MIN: dict[str, npt.NDArray[np.float64]] = {
    "n": np.array([-19.04, -19.62, -21.21, -23.25, -21.63, -25.74]),
    "c": np.array([-18.83, -18.86, -23.48, -28.07, -25.96, -23.27]),
    "ct": np.array([-16.18, -21.15, -25.37, -26.63, -26.53, -26.18]),
}
_MACKINNON_TAU_STAR: dict[str, npt.NDArray[np.float64]] = {
    "n": np.array([-1.04, -1.53, -2.68, -3.09, -3.07, -3.77]),
    "c": np.array([-1.61, -2.62, -3.13, -3.47, -3.78, -3.93]),
    "ct": np.array([-2.89, -3.19, -3.50, -3.65, -3.80, -4.36]),
}
# Coefficients of the polynomial p(tau) fit by MacKinnon to the normal
# quantile of the empirical p-value, in ascending powers of tau: p_value =
# Phi(c0 + c1*tau + c2*tau^2 [+ c3*tau^3]). "Small" applies left of the
# N-specific star statistic above (deep in the rejection region, where a
# quadratic fits the simulated quantiles well); "large" applies to the right
# (a cubic is needed to track the distribution's right tail).
_MACKINNON_SMALL_P_COEF: dict[str, npt.NDArray[np.float64]] = {
    "n": np.array(
        [
            [0.6344, 1.2378, 0.032496],
            [1.9129, 1.3857, 0.035322],
            [2.7648, 1.4502, 0.034186],
            [3.4336, 1.4835, 0.031900],
            [4.0999, 1.5533, 0.035900],
            [4.5388, 1.5344, 0.029807],
        ]
    ),
    "c": np.array(
        [
            [2.1659, 1.4412, 0.038269],
            [2.9200, 1.5012, 0.039796],
            [3.4699, 1.4856, 0.031640],
            [3.9673, 1.4777, 0.026315],
            [4.5509, 1.5338, 0.029545],
            [5.1399, 1.6036, 0.034445],
        ]
    ),
    "ct": np.array(
        [
            [3.2512, 1.6047, 0.049588],
            [3.6646, 1.5419, 0.036448],
            [4.0983, 1.5173, 0.029898],
            [4.5844, 1.5338, 0.028796],
            [5.0722, 1.5634, 0.029472],
            [5.5300, 1.5914, 0.030392],
        ]
    ),
}
_MACKINNON_LARGE_P_COEF: dict[str, npt.NDArray[np.float64]] = {
    "n": np.array(
        [
            [0.4797, 0.93557, -0.06999, 0.033066],
            [1.5578, 0.85580, -0.20830, -0.033549],
            [2.2268, 0.68093, -0.32362, -0.054448],
            [2.7654, 0.64502, -0.30811, -0.044946],
            [3.2684, 0.68051, -0.26778, -0.034972],
            [3.7268, 0.71670, -0.23648, -0.028288],
        ]
    ),
    "c": np.array(
        [
            [1.7339, 0.93202, -0.12745, -0.010368],
            [2.1945, 0.64695, -0.29198, -0.042377],
            [2.5893, 0.45168, -0.36529, -0.050074],
            [3.0387, 0.45452, -0.33666, -0.041921],
            [3.5049, 0.52098, -0.29158, -0.033468],
            [3.9489, 0.58933, -0.25359, -0.027210],
        ]
    ),
    "ct": np.array(
        [
            [2.5261, 0.61654, -0.37956, -0.060285],
            [2.8500, 0.52720, -0.36622, -0.051695],
            [3.2210, 0.52550, -0.32685, -0.041501],
            [3.6520, 0.59758, -0.27483, -0.032081],
            [4.0712, 0.66428, -0.23464, -0.025460],
            [4.4735, 0.71757, -0.20681, -0.021196],
        ]
    ),
}
_ADF_REGRESSION_CASES = ("n", "c", "ct")


def _mackinnon_pvalue(t_stat: float, n_series: int, regression: str = "c") -> float:
    """MacKinnon (1994) response-surface p-value for the (A)DF tau statistic.

    Args:
        t_stat: The (A)DF tau statistic.
        n_series: Number of I(1) series in the regression the tested
            residuals came from (1 for a direct unit-root test; N for an
            N-variable Engle-Granger residual-based test).
        regression: Deterministic case the ADF regression included: "n"
            (no constant, no trend), "c" (constant only), or "ct" (constant
            and linear trend).

    Returns:
        Approximate p-value under the null of a unit root,
        `Phi(c0 + c1*tau + c2*tau^2 [+ c3*tau^3])`, in [0, 1]. The
        polynomials are monotone only between the tabulated minimum and
        maximum statistic, both of which belong to the range: a statistic
        below the minimum returns exactly 0.0 and one above the maximum
        exactly 1.0, where the approximation says no more than that the
        p-value is close to 0 or 1. Inside the range the value is returned
        as computed, however small (between 1e-22 and 1e-63 at the
        minimum, depending on the case and N). Convention and values are
        those of `statsmodels.tsa.adfvalues.mackinnonp`.
    """
    if regression not in _ADF_REGRESSION_CASES:
        raise ValueError(f"regression must be one of {_ADF_REGRESSION_CASES}")
    tau_max = _MACKINNON_TAU_MAX[regression]
    if not 1 <= n_series <= tau_max.size:
        raise ValueError(
            f"n_series must be between 1 and {tau_max.size} "
            f"(MacKinnon's tabulated range), got {n_series}"
        )
    idx = n_series - 1
    if t_stat > tau_max[idx]:
        return 1.0
    if t_stat < _MACKINNON_TAU_MIN[regression][idx]:
        return 0.0
    coefs = (
        _MACKINNON_SMALL_P_COEF[regression][idx]
        if t_stat <= _MACKINNON_TAU_STAR[regression][idx]
        else _MACKINNON_LARGE_P_COEF[regression][idx]
    )
    poly_value = float(np.polynomial.polynomial.polyval(t_stat, coefs))
    return float(norm.cdf(poly_value))


def _validate_adf_inputs(series: npt.NDArray[np.float64], max_lags: int, trend: str) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    if trend not in _ADF_REGRESSION_CASES:
        raise ValueError(f"trend must be one of {_ADF_REGRESSION_CASES}")
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
    _validate_adf_inputs(series, max_lags, trend)
    return _adf_test(series, max_lags, n_series=1, trend=trend, autolag=autolag)


def _adf_design_matrix(
    dy: npt.NDArray[np.float64],
    level: npt.NDArray[np.float64],
    start: int,
    end: int,
    n_lags: int,
    trend: str,
) -> tuple[npt.NDArray[np.float64], int]:
    """Build the ADF regression design matrix for `n_lags` lagged differences.

    Returns (x_reg, level_col): `level_col` is the column index of the
    level regressor `y_{t-1}` (whose coefficient's t-stat is the ADF
    statistic) -- it depends on how many deterministic columns precede it.
    """
    n_obs = end - start
    columns = []
    if trend in ("c", "ct"):
        columns.append(np.ones(n_obs))
    if trend == "ct":
        columns.append(np.arange(n_obs, dtype=np.float64))
    level_col = len(columns)
    columns.append(level[start:end])
    for lag in range(1, n_lags + 1):
        columns.append(dy[start - lag : end - lag])
    return np.column_stack(columns), level_col


def _adf_regression(
    dy: npt.NDArray[np.float64],
    level: npt.NDArray[np.float64],
    start: int,
    end: int,
    n_lags: int,
    trend: str,
) -> tuple[float, npt.NDArray[np.float64], float]:
    """Fit the ADF regression for a given lag count; return (t_stat, residuals, log_likelihood)."""
    x_reg, level_col = _adf_design_matrix(dy, level, start, end, n_lags, trend)
    y_reg = dy[start:end]
    n_obs = y_reg.size

    beta, _, _, _ = np.linalg.lstsq(x_reg, y_reg, rcond=None)
    residuals = y_reg - x_reg @ beta
    k = x_reg.shape[1]
    dof = n_obs - k
    ssr = float(np.sum(residuals**2))
    sigma2 = ssr / dof
    xtx_inv = np.linalg.inv(x_reg.T @ x_reg)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[level_col, level_col]))

    t_stat = float(beta[level_col] / se_rho)
    log_likelihood = -0.5 * n_obs * (np.log(2.0 * np.pi) + np.log(ssr / n_obs) + 1.0)
    return t_stat, residuals, log_likelihood


def _select_adf_lags(
    dy: npt.NDArray[np.float64],
    level: npt.NDArray[np.float64],
    start: int,
    end: int,
    max_lags: int,
    trend: str,
    autolag: str,
) -> int:
    """Select the lag count in 0..max_lags via `autolag`'s criterion.

    The regression sample (`start`/`end`, fixed at `max_lags`) is held
    identical across every candidate lag count, so the criteria being
    compared are all computed on the same data -- matching
    `statsmodels.tsa.stattools.adfuller`'s `autolag` behavior, where the
    sample is always trimmed to the largest candidate lag first.
    """
    if autolag in ("aic", "bic"):
        best_lag = 0
        best_score = np.inf
        for n_lags in range(max_lags + 1):
            x_reg, _ = _adf_design_matrix(dy, level, start, end, n_lags, trend)
            _, _, log_likelihood = _adf_regression(dy, level, start, end, n_lags, trend)
            k = x_reg.shape[1]
            n_obs = end - start
            score = (
                -2.0 * log_likelihood + 2.0 * k
                if autolag == "aic"
                else -2.0 * log_likelihood + k * np.log(n_obs)
            )
            if score < best_score:
                best_score = score
                best_lag = n_lags
        return best_lag

    if autolag == "t-stat":
        # scipy.stats.norm.ppf(0.95); avoids importing scipy.stats.norm's
        # ppf just for this one constant since `norm` here is already
        # imported for its cdf.
        critical_value = 1.6448536269514722
        for n_lags in range(max_lags, -1, -1):
            x_reg, _ = _adf_design_matrix(dy, level, start, end, n_lags, trend)
            if n_lags == 0:
                return 0
            y_reg = dy[start:end]
            beta, _, _, _ = np.linalg.lstsq(x_reg, y_reg, rcond=None)
            residuals = y_reg - x_reg @ beta
            n_obs = y_reg.size
            k = x_reg.shape[1]
            dof = n_obs - k
            sigma2 = float(np.sum(residuals**2) / dof)
            xtx_inv = np.linalg.inv(x_reg.T @ x_reg)
            se_last = float(np.sqrt(sigma2 * xtx_inv[-1, -1]))
            t_last = float(beta[-1] / se_last)
            if abs(t_last) >= critical_value:
                return n_lags
        return 0

    raise ValueError('autolag must be "aic", "bic", "t-stat", or None')


def _adf_test(
    series: npt.NDArray[np.float64],
    max_lags: int,
    n_series: int,
    trend: str = "c",
    autolag: str | None = None,
) -> tuple[float, float]:
    dy = np.diff(series)
    level = series[:-1]
    end = dy.size

    if autolag is None:
        n_lags = max_lags
    else:
        # Lag selection compares AIC/BIC/t-stat across candidates on a
        # sample trimmed to `max_lags` (fixed, so the criteria are computed
        # on identical data for every candidate) -- but the final reported
        # statistic is then re-fit on the larger sample actually available
        # once only `n_lags` observations are trimmed, matching
        # `statsmodels.tsa.stattools.adfuller`'s two-stage procedure.
        n_lags = _select_adf_lags(dy, level, max_lags, end, max_lags, trend, autolag)
    t_stat, _, _ = _adf_regression(dy, level, n_lags, end, n_lags, trend)
    p_value = _mackinnon_pvalue(t_stat, n_series, trend)
    return t_stat, p_value


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
        autolag=None)`, whose statistic and p-value this function
        reproduces to ~1e-10.
    """
    _validate_engle_granger_inputs(y, x)
    return _engle_granger_test(y, x)


def _engle_granger_test(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    x_reg = np.column_stack([np.ones_like(x), x])
    ols_beta, _, _, _ = np.linalg.lstsq(x_reg, y, rcond=None)
    beta = float(ols_beta[1])
    residuals = y - x_reg @ ols_beta
    # Step 2: ADF regression on the residuals with one lagged difference and
    # *no* deterministic terms -- the step-1 OLS fit already includes an
    # intercept, so its residuals are mean-zero by construction, and the
    # MacKinnon N=2 surface is tabulated for the statistic computed that
    # way (Engle & Granger 1987). The p-value's deterministic case is "c"
    # because that is what the *cointegrating* regression included, not
    # what the residual ADF regression included. Same conventions as
    # `statsmodels.tsa.stattools.coint`: `adfuller(resid, regression="n")`
    # then `mackinnonp(stat, regression="c", N=2)`.
    dy = np.diff(residuals)
    level = residuals[:-1]
    n_lags = 1
    adf_stat, _, _ = _adf_regression(dy, level, n_lags, dy.size, n_lags, trend="n")
    p_value = _mackinnon_pvalue(adf_stat, n_series=2, regression="c")
    return beta, adf_stat, p_value


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
    """
    _validate_ou_half_life_inputs(spread)
    return _ou_half_life(spread)


def _ou_half_life(spread: npt.NDArray[np.float64]) -> float:
    delta = spread[1:] - spread[:-1]
    level = spread[:-1]
    x_reg = np.column_stack([np.ones_like(level), level])
    beta, _, _, _ = np.linalg.lstsq(x_reg, delta, rcond=None)
    a = float(beta[1])
    level_coef = 1.0 + a
    if a >= 0.0 or level_coef <= 0.0:
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
