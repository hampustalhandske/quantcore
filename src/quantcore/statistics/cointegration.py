"""Cointegration testing and mean-reversion statistics for pairs/spread trading.

Engle-Granger (1987) two-step:
    Step 1 — estimate cointegrating vector via OLS: y = beta*x + epsilon
    Step 2 — ADF test on residuals epsilon_hat;
             null hypothesis: epsilon_hat has a unit root (no cointegration)

ADF test statistic (Dickey & Fuller 1979):
    Delta(y_t) = alpha + rho*y_{t-1} + sum_{j=1}^p gamma_j*Delta(y_{t-j}) + e_t
    t-stat for rho=0 is the ADF statistic. p-values here are approximated via
    a Student-t reference distribution on the regression's residual degrees
    of freedom (NOT the true Dickey-Fuller distribution, which has no closed
    form) — adequate for a directional accept/reject signal, not for
    publication-grade inference.

Johansen trace test (Johansen 1988): reduced-rank regression of Delta(Y_t) and
Y_{t-1} on lagged differences, then eigen-decomposition of
    S11^{-1/2} * S10 * S00^{-1} * S01 * S11^{-1/2}
where S00, S01, S10, S11 are the residual covariance matrices from those two
regressions. Trace statistic for rank r: -n * sum_{i=r+1}^{k} ln(1 - lambda_i).

OU half-life (Avellaneda & Lee 2010):
    Delta(spread_t) = a * spread_{t-1} + noise  (fit via OLS)
    level-form AR(1) coefficient: phi = 1 + a
    half_life = -ln(2) / ln(phi)

Z-score (rolling):
    z_t = (spread_t - rolling_mean_t) / rolling_std_t

References:
    Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error
    Correction: Representation, Estimation, and Testing." Econometrica.
    Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors."
    Journal of Economic Dynamics and Control.
    Avellaneda, M. and Lee, J.-H. (2010). "Statistical Arbitrage in the U.S.
    Equities Market." Quantitative Finance.
    Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
    Autoregressive Time Series with a Unit Root." JASA.
    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

# ---------------------------------------------------------------------------
# ADF unit-root test
# ---------------------------------------------------------------------------


def _validate_adf_inputs(series: npt.NDArray[np.float64], max_lags: int) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    min_length = max_lags + 5
    if series.size < min_length:
        raise ValueError(
            f"series must have at least {min_length} observations for max_lags={max_lags}"
        )


def adf_test(series: npt.NDArray[np.float64], max_lags: int = 1) -> tuple[float, float]:
    """Augmented Dickey-Fuller test for a unit root.

    Args:
        series: The time series to test.
        max_lags: Number of lagged differences included in the regression.

    Returns:
        Tuple (adf_statistic, p_value). p_value is a Student-t approximation
        of the true (tabulated) Dickey-Fuller distribution.
    """
    _validate_adf_inputs(series, max_lags)
    return _adf_test(series, max_lags)


def _adf_test(series: npt.NDArray[np.float64], max_lags: int) -> tuple[float, float]:
    dy = np.diff(series)
    level = series[:-1]
    start = max_lags
    end = dy.size
    n_obs = end - start

    y_reg = dy[start:end]
    columns = [np.ones(n_obs), level[start:end]]
    for lag in range(1, max_lags + 1):
        columns.append(dy[start - lag : end - lag])
    x_reg = np.column_stack(columns)

    beta, _, _, _ = np.linalg.lstsq(x_reg, y_reg, rcond=None)
    residuals = y_reg - x_reg @ beta
    k = x_reg.shape[1]
    dof = n_obs - k
    sigma2 = float(np.sum(residuals**2) / dof)
    xtx_inv = np.linalg.inv(x_reg.T @ x_reg)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    t_stat = float(beta[1] / se_rho)
    p_value = float(stats.t.cdf(t_stat, dof))
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
        coefficient and the ADF test result on the OLS residuals.
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
    adf_stat, p_value = _adf_test(residuals, max_lags=1)
    return beta, adf_stat, p_value


# ---------------------------------------------------------------------------
# Johansen trace test
# ---------------------------------------------------------------------------

# Standard asymptotic 95% critical values for the Johansen trace test, case
# "unrestricted constant / no trend" (Johansen 1988; commonly reproduced as
# in Osterwald-Lenum 1992 / MacKinnon, Haug & Michelis 1999), indexed by
# k - r (the number of common trends under the null).
_JOHANSEN_TRACE_CRIT_95: dict[int, float] = {
    1: 12.3212,
    2: 24.2761,
    3: 40.1749,
    4: 60.0627,
    5: 84.4500,
    6: 114.9020,
    7: 149.7180,
    8: 185.9000,
    9: 227.7300,
    10: 271.7900,
}


def _validate_johansen_inputs(data: npt.NDArray[np.float64], n_lags: int) -> None:
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must be 2D with at least 2 columns")
    if n_lags < 1:
        raise ValueError("n_lags must be a positive integer")
    if data.shape[1] > max(_JOHANSEN_TRACE_CRIT_95):
        raise ValueError("johansen_trace_test only supports up to 10 series")
    min_rows = n_lags + data.shape[1] + 5
    if data.shape[0] < min_rows:
        raise ValueError(
            f"data must have at least {min_rows} rows for n_lags={n_lags} and "
            f"{data.shape[1]} columns"
        )


def johansen_trace_test(
    data: npt.NDArray[np.float64], n_lags: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Johansen trace test for cointegration rank via reduced-rank regression.

    Args:
        data: Shape (T, k) — k time series.
        n_lags: Number of lagged differences in the VECM short-run dynamics.

    Returns:
        Tuple (trace_statistics, critical_values_95pct, eigenvectors):
        trace_statistics has shape (k,) — trace_statistics[r] tests
        H0: rank <= r; critical_values_95pct has shape (k,); eigenvectors has
        shape (k, k), columns are the (unnormalized) cointegrating vectors.
    """
    _validate_johansen_inputs(data, n_lags)
    return _johansen_trace_test(data, n_lags)


def _johansen_trace_test(
    data: npt.NDArray[np.float64], n_lags: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    _, k = data.shape
    d_y = np.diff(data, axis=0)  # d_y[i] = Y_{i+1} - Y_i

    dep = d_y[n_lags:]  # Delta Y_t, t = n_lags+1 .. t_total-1
    level = data[n_lags:-1]  # Y_{t-1}, same t range
    n_obs = dep.shape[0]

    z = np.column_stack(
        [np.ones(n_obs)] + [d_y[n_lags - lag : d_y.shape[0] - lag] for lag in range(1, n_lags + 1)]
    )

    def _residuals(target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        coefs, _, _, _ = np.linalg.lstsq(z, target, rcond=None)
        result: npt.NDArray[np.float64] = target - z @ coefs
        return result

    r0 = _residuals(dep)
    r1 = _residuals(level)

    s00 = r0.T @ r0 / n_obs
    s01 = r0.T @ r1 / n_obs
    s10 = r1.T @ r0 / n_obs
    s11 = r1.T @ r1 / n_obs

    s11_eigvals, s11_eigvecs = np.linalg.eigh(s11)
    s11_eigvals = np.clip(s11_eigvals, 1e-12, None)
    s11_inv_sqrt = s11_eigvecs @ np.diag(s11_eigvals**-0.5) @ s11_eigvecs.T
    s00_inv = np.linalg.inv(s00)

    m = s11_inv_sqrt @ s10 @ s00_inv @ s01 @ s11_inv_sqrt
    m = (m + m.T) / 2.0  # symmetrize away floating-point asymmetry
    eigvals, eigvecs_sym = np.linalg.eigh(m)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, 1.0 - 1e-12)
    eigvecs_sym = eigvecs_sym[:, order]
    eigenvectors = s11_inv_sqrt @ eigvecs_sym

    trace_stats = np.array(
        [-n_obs * np.sum(np.log(1.0 - eigvals[r:])) for r in range(k)], dtype=np.float64
    )
    crit_values = np.array([_JOHANSEN_TRACE_CRIT_95[k - r] for r in range(k)], dtype=np.float64)
    return trace_stats, crit_values, eigenvectors


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
