"""Ljung-Box autocorrelation testing and ARIMA(p, d, q) fitting/forecasting.

ARIMA(p, d, q) (Box, Jenkins & Reinsel 2015):
    phi(B) * Delta^d(y_t) = theta(B) * e_t
    where phi(B) = 1 - phi_1*B - ... - phi_p*B^p  (AR polynomial)
          theta(B) = 1 + theta_1*B + ... + theta_q*B^q (MA polynomial)
          Delta = (1 - B) (difference operator)
    Estimated via conditional sum-of-squares (CSS): residuals are computed
    recursively from the AR/MA polynomials with pre-sample residuals (and,
    for the CSS objective, pre-sample y-lags) treated as 0, and the
    parameters minimize sum_t e_t^2.

Ljung-Box Q-statistic (Ljung & Box 1978):
    Q(m) = T*(T+2) * sum_{k=1}^m rho_k^2 / (T-k)
    ~ chi^2(m) under H0: no autocorrelation up to lag m

References:
    Box, G.E.P., Jenkins, G.M., and Reinsel, G.C. (2015). *Time Series
    Analysis: Forecasting and Control* (5th ed.). Wiley.

    Ljung, G.M. and Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time
    Series Models." *Biometrika*, 65(2), 297-303.

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import chi2


def _validate_ljung_box_inputs(residuals: npt.NDArray[np.float64], n_lags: int) -> None:
    if residuals.size == 0:
        raise ValueError("residuals must be non-empty")
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    if n_lags >= residuals.size:
        raise ValueError("n_lags must be smaller than the number of residuals")


def ljung_box_test(
    residuals: npt.NDArray[np.float64],
    n_lags: int,
) -> tuple[float, float]:
    """Ljung-Box Q-test for autocorrelation in a residual series.

    Args:
        residuals: Residual series to test.
        n_lags: Number of lags to include in the test statistic.

    Returns:
        Tuple (q_statistic, p_value), with p_value from chi2.sf(Q, df=n_lags).
    """
    _validate_ljung_box_inputs(residuals, n_lags)
    return _ljung_box_test(residuals, n_lags)


def _ljung_box_test(residuals: npt.NDArray[np.float64], n_lags: int) -> tuple[float, float]:
    n = residuals.size
    centered = residuals - residuals.mean()
    denom = float(np.sum(centered**2))

    autocorr = np.empty(n_lags, dtype=np.float64)
    for k in range(1, n_lags + 1):
        autocorr[k - 1] = float(np.sum(centered[k:] * centered[:-k])) / denom

    lags = np.arange(1, n_lags + 1, dtype=np.float64)
    q_statistic = float(n * (n + 2) * np.sum(autocorr**2 / (n - lags)))
    p_value = float(chi2.sf(q_statistic, df=n_lags))
    return q_statistic, p_value


def _validate_arima_fit_inputs(series: npt.NDArray[np.float64], p: int, d: int, q: int) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if p < 0:
        raise ValueError("p must be non-negative")
    if q < 0:
        raise ValueError("q must be non-negative")
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1, or 2")
    if p == 0 and q == 0:
        raise ValueError("p and q cannot both be 0")
    if series.size <= d + max(p, q):
        raise ValueError("series is too short for the requested (p, d, q) order")


def arima_fit(
    series: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Estimate ARIMA(p, d, q) parameters via conditional sum-of-squares (CSS).

    Args:
        series: Raw (undifferenced) time series.
        p: AR order.
        d: Integration order (0, 1, or 2).
        q: MA order.

    Returns:
        Tuple (ar_coefficients, ma_coefficients, sigma2_hat).
    """
    _validate_arima_fit_inputs(series, p, d, q)
    return _arima_fit(series, p, d, q)


def _difference(series: npt.NDArray[np.float64], d: int) -> npt.NDArray[np.float64]:
    diffed = series
    for _ in range(d):
        diffed = np.diff(diffed)
    return diffed


def _css_residuals(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p: int,
    q: int,
) -> npt.NDArray[np.float64]:
    phi = params[:p]
    ma = params[p : p + q]
    n = y.size
    start = max(p, q)
    residuals = np.zeros(n, dtype=np.float64)

    for t in range(start, n):
        ar_term = float(np.dot(phi, y[t - p : t][::-1])) if p > 0 else 0.0
        ma_term = float(np.dot(ma, residuals[t - q : t][::-1])) if q > 0 else 0.0
        residuals[t] = y[t] - ar_term - ma_term

    return residuals[start:]


def _css_objective(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p: int,
    q: int,
) -> float:
    residuals = _css_residuals(params, y, p, q)
    return float(np.sum(residuals**2))


def _arima_fit(
    series: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    y = _difference(series, d)
    x0 = np.zeros(p + q, dtype=np.float64)

    result = minimize(
        _css_objective,
        x0,
        args=(y, p, q),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000, "maxfev": 5000},
    )

    params = result.x
    ar_coefs = params[:p]
    ma_coefs = params[p : p + q]
    n_used = y.size - max(p, q)
    sigma2_hat = float(result.fun) / n_used

    return ar_coefs, ma_coefs, sigma2_hat


def _validate_arima_forecast_inputs(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1, or 2")
    if h <= 0:
        raise ValueError("h must be a positive integer")
    if series.size <= d + ar_coefs.size:
        raise ValueError("series is too short to seed the forecast for the given order")


def arima_forecast(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Produce h-step-ahead ARIMA point forecasts on the original scale.

    Future innovations are assumed to be zero (standard conditional-mean
    forecast), so the MA terms do not contribute beyond the fitted sample.

    Args:
        series: Original (undifferenced) series the model was fit on.
        ar_coefs: Fitted AR coefficients, shape (p,).
        ma_coefs: Fitted MA coefficients, shape (q,) (unused beyond validation
            since future/pre-forecast innovations are treated as zero).
        d: Integration order used when fitting.
        h: Forecast horizon.

    Returns:
        Array of shape (h,) with point forecasts on the original scale.
    """
    _validate_arima_forecast_inputs(series, ar_coefs, d, h)
    return _arima_forecast(series, ar_coefs, ma_coefs, d, h)


def _arima_forecast(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    del ma_coefs  # future/unavailable innovations are treated as zero throughout

    diffs = [series]
    for _ in range(d):
        diffs.append(np.diff(diffs[-1]))

    p = ar_coefs.size
    history = list(diffs[-1])
    for _ in range(h):
        if p > 0:
            lags = np.array(history[-p:][::-1], dtype=np.float64)
            next_value = float(np.dot(ar_coefs, lags))
        else:
            next_value = 0.0
        history.append(next_value)

    forecast = np.array(history[-h:], dtype=np.float64)

    for k in range(d - 1, -1, -1):
        seed = diffs[k][-1]
        forecast = seed + np.cumsum(forecast)

    return forecast
