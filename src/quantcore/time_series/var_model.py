"""Vector Autoregression: estimation, forecasting, impulse response, Granger causality.

VAR(p) model (Sims 1980, Lutkepohl 2005):
    y_t = c + A_1*y_{t-1} + ... + A_p*y_{t-p} + u_t,  u_t ~ N(0, Sigma_u)
    Estimated by OLS equation-by-equation.

Orthogonalized impulse response function (Cholesky):
    Sigma_u = P * P^T  (Cholesky)
    IRF_0 = P,  IRF_h = sum_{j=1}^h A_j * IRF_{h-j}

Granger causality (Granger 1969):
    F = [(RSS_R - RSS_U) / p] / [RSS_U / (T - k_U)]
    H0: x does not Granger-cause y (coefficients on lags of x in the y
    equation are all zero).

References:
    Sims, C.A. (1980). "Macroeconomics and Reality." VAR(p) model.
    Lutkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. OLS estimation, impulse response functions.
    Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric
    Models and Cross-Spectral Methods." Granger causality F-test.
    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import f as f_dist


def _build_lagged_design(data: npt.NDArray[np.float64], n_lags: int) -> npt.NDArray[np.float64]:
    n_obs, k = data.shape
    n_eff = n_obs - n_lags
    x = np.empty((n_eff, k * n_lags + 1), dtype=np.float64)
    for lag in range(1, n_lags + 1):
        x[:, (lag - 1) * k : lag * k] = data[n_lags - lag : n_obs - lag]
    x[:, -1] = 1.0
    return x


def _validate_var_fit_inputs(data: npt.NDArray[np.float64], n_lags: int) -> None:
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must be a non-empty 2-D array of shape (T, k)")
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    n_obs, k = data.shape
    n_eff = n_obs - n_lags
    if n_eff <= k * n_lags + 1:
        raise ValueError("not enough observations for the requested n_lags")


def var_fit(
    data: npt.NDArray[np.float64],
    n_lags: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate a VAR(n_lags) model via OLS, equation by equation.

    Args:
        data: Observations, shape (T, k).
        n_lags: Number of lags p.

    Returns:
        Tuple (coef_matrix, sigma_u): coef_matrix has shape
        (k, k*n_lags + 1) with the intercept in the last column;
        sigma_u is the (k, k) residual covariance matrix, normalized by
        `T_eff - (k*n_lags + 1)` (Lutkepohl 2005's unbiased estimator: the
        number of effective observations minus the number of parameters
        estimated per equation), not `T_eff - 1` -- a plain per-column
        `ddof=1` sample covariance of the residuals would silently use the
        wrong degrees of freedom, since the residuals came from a
        `k*n_lags + 1`-parameter regression per equation, not a 1-parameter
        (mean-only) one.

    References:
        Lutkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. (Sigma_u_hat, Section 3.2.2.) See
        docs/REFERENCES.md.
    """
    _validate_var_fit_inputs(data, n_lags)
    return _var_fit(data, n_lags)


def _var_fit(
    data: npt.NDArray[np.float64],
    n_lags: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    k = data.shape[1]
    x = _build_lagged_design(data, n_lags)
    y = data[n_lags:]
    coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    coef_matrix = coefs.T
    residuals = y - x @ coefs
    n_eff, n_params = x.shape
    dof = n_eff - n_params
    sigma_u = (residuals.T @ residuals) / dof
    if k == 1:
        sigma_u = sigma_u.reshape(1, 1)
    return coef_matrix, sigma_u


def select_var_lag_order(
    system: npt.NDArray[np.float64],
    max_lags: int,
    criterion: str = "aic",
) -> int:
    """Grid-search VAR lag order and return the one minimizing an information criterion.

    Args:
        system: Observations, shape (T, k).
        max_lags: Largest lag order to try (inclusive), searched from 1.
        criterion: "aic" or "bic".

    Returns:
        The n_lags minimizing the chosen criterion among candidates that
        `var_fit` accepts. Candidates rejected by `var_fit` (not enough
        observations for the requested lag order) are skipped.

    Raises:
        ValueError: If `criterion` is not "aic" or "bic", or if no candidate
            lag order is feasible for the given system length.

    Note:
        Every candidate lag order is fit on the *same* trailing window of
        `n_obs - max_lags` observations (dropping the first `max_lags -
        n_lags` rows for a given `n_lags`), not on `n_obs - n_lags`
        observations each. Information criteria are only comparable across
        models fit on the same effective sample size; holding it fixed at
        the smallest one usable by the largest candidate is the standard
        procedure (Lutkepohl 2005, pp. 146-150), matching
        `statsmodels.tsa.api.VAR.select_order`. The criteria themselves use
        the maximum-likelihood (bias-uncorrected, 1/n_eff) residual
        covariance, not `var_fit`'s unbiased estimator -- also per
        Lutkepohl's definition and `statsmodels`' `sigma_u_mle`.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError('criterion must be "aic" or "bic"')

    n_obs, k = system.shape
    best_n_lags: int | None = None
    best_score = np.inf

    for n_lags in range(1, max_lags + 1):
        try:
            coef_matrix, _ = var_fit(system[max_lags - n_lags :], n_lags)
        except ValueError:
            continue

        n_eff = n_obs - max_lags
        x = _build_lagged_design(system[max_lags - n_lags :], n_lags)
        residuals = system[max_lags:] - x @ coef_matrix.T
        sigma_u_mle = (residuals.T @ residuals) / n_eff
        if k == 1:
            sigma_u_mle = sigma_u_mle.reshape(1, 1)
        _, log_det = np.linalg.slogdet(sigma_u_mle)

        free_params = n_lags * k**2 + k
        score = (
            log_det + (2.0 / n_eff) * free_params
            if criterion == "aic"
            else log_det + (np.log(n_eff) / n_eff) * free_params
        )

        if score < best_score:
            best_score = score
            best_n_lags = n_lags

    if best_n_lags is None:
        raise ValueError("no feasible n_lags found for the given system and bounds")

    return best_n_lags


def _validate_var_forecast_inputs(
    data: npt.NDArray[np.float64],
    coef_matrix: npt.NDArray[np.float64],
    n_lags: int,
    h: int,
) -> None:
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must be a non-empty 2-D array of shape (T, k)")
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    if h <= 0:
        raise ValueError("h must be a positive integer")
    k = data.shape[1]
    if coef_matrix.shape != (k, k * n_lags + 1):
        raise ValueError(
            f"coef_matrix must have shape ({k}, {k * n_lags + 1}) for k={k}, n_lags={n_lags}"
        )
    if data.shape[0] < n_lags:
        raise ValueError("data must contain at least n_lags observations")


def var_forecast(
    data: npt.NDArray[np.float64],
    coef_matrix: npt.NDArray[np.float64],
    n_lags: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Produce h-step-ahead point forecasts from a fitted VAR(n_lags) model.

    Args:
        data: History, shape (T, k).
        coef_matrix: Fitted VAR coefficients, shape (k, k*n_lags + 1).
        n_lags: Number of lags p used to fit coef_matrix.
        h: Forecast horizon.

    Returns:
        Point forecasts, shape (h, k).
    """
    _validate_var_forecast_inputs(data, coef_matrix, n_lags, h)
    return _var_forecast(data, coef_matrix, n_lags, h)


def _var_forecast(
    data: npt.NDArray[np.float64],
    coef_matrix: npt.NDArray[np.float64],
    n_lags: int,
    h: int,
) -> npt.NDArray[np.float64]:
    k = data.shape[1]
    intercept = coef_matrix[:, -1]
    history = list(data[-n_lags:])
    forecasts = np.empty((h, k), dtype=np.float64)
    for step in range(h):
        y_next = intercept.copy()
        for lag in range(1, n_lags + 1):
            a_lag = coef_matrix[:, (lag - 1) * k : lag * k]
            y_next = y_next + a_lag @ history[-lag]
        forecasts[step] = y_next
        history.append(y_next)
    return forecasts


def _validate_impulse_response_inputs(
    coef_matrix: npt.NDArray[np.float64],
    sigma_u: npt.NDArray[np.float64],
    n_lags: int,
    n_periods: int,
) -> None:
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    if n_periods <= 0:
        raise ValueError("n_periods must be a positive integer")
    if sigma_u.ndim != 2 or sigma_u.shape[0] != sigma_u.shape[1]:
        raise ValueError("sigma_u must be a square 2-D array")
    k = sigma_u.shape[0]
    if coef_matrix.shape != (k, k * n_lags + 1):
        raise ValueError(
            f"coef_matrix must have shape ({k}, {k * n_lags + 1}) matching sigma_u's dimension"
        )


def impulse_response(
    coef_matrix: npt.NDArray[np.float64],
    sigma_u: npt.NDArray[np.float64],
    n_lags: int,
    n_periods: int,
) -> npt.NDArray[np.float64]:
    """Compute the Cholesky-orthogonalized impulse response function.

    Args:
        coef_matrix: Fitted VAR coefficients, shape (k, k*n_lags + 1).
        sigma_u: Residual covariance, shape (k, k).
        n_lags: Number of lags p.
        n_periods: Number of IRF horizons to compute.

    Returns:
        Array of shape (n_periods, k, k); irf[h, i, j] is the response of
        variable i to a shock in variable j at horizon h.
    """
    _validate_impulse_response_inputs(coef_matrix, sigma_u, n_lags, n_periods)
    return _impulse_response(coef_matrix, sigma_u, n_lags, n_periods)


def _impulse_response(
    coef_matrix: npt.NDArray[np.float64],
    sigma_u: npt.NDArray[np.float64],
    n_lags: int,
    n_periods: int,
) -> npt.NDArray[np.float64]:
    k = sigma_u.shape[0]
    p_chol = np.linalg.cholesky(sigma_u)
    irf = np.zeros((n_periods, k, k), dtype=np.float64)
    irf[0] = p_chol
    a_lags = [coef_matrix[:, (lag - 1) * k : lag * k] for lag in range(1, n_lags + 1)]
    for h in range(1, n_periods):
        acc = np.zeros((k, k), dtype=np.float64)
        for lag in range(1, min(h, n_lags) + 1):
            acc = acc + a_lags[lag - 1] @ irf[h - lag]
        irf[h] = acc
    return irf


def _validate_granger_causality_inputs(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    n_lags: int,
) -> None:
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    n_eff = y.shape[0] - n_lags
    n_regressors_unrestricted = 2 * n_lags + 1
    if n_eff <= n_regressors_unrestricted:
        raise ValueError("not enough observations for the requested n_lags")


def granger_causality_test(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    n_lags: int,
) -> tuple[float, float]:
    """Test H0: x does not Granger-cause y, via a restricted-vs-unrestricted F-test.

    Args:
        y: Dependent series, shape (T,).
        x: Potentially causal series, shape (T,).
        n_lags: Number of lags p.

    Returns:
        Tuple (f_statistic, p_value).
    """
    _validate_granger_causality_inputs(y, x, n_lags)
    return _granger_causality_test(y, x, n_lags)


def _granger_causality_test(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    n_lags: int,
) -> tuple[float, float]:
    n_obs = y.shape[0]
    n_eff = n_obs - n_lags
    target = y[n_lags:]

    x_restricted = np.empty((n_eff, n_lags + 1), dtype=np.float64)
    for lag in range(1, n_lags + 1):
        x_restricted[:, lag - 1] = y[n_lags - lag : n_obs - lag]
    x_restricted[:, -1] = 1.0

    x_unrestricted = np.empty((n_eff, 2 * n_lags + 1), dtype=np.float64)
    x_unrestricted[:, :n_lags] = x_restricted[:, :n_lags]
    for lag in range(1, n_lags + 1):
        x_unrestricted[:, n_lags + lag - 1] = x[n_lags - lag : n_obs - lag]
    x_unrestricted[:, -1] = 1.0

    beta_r, _, _, _ = np.linalg.lstsq(x_restricted, target, rcond=None)
    rss_r = float(np.sum((target - x_restricted @ beta_r) ** 2))

    beta_u, _, _, _ = np.linalg.lstsq(x_unrestricted, target, rcond=None)
    rss_u = float(np.sum((target - x_unrestricted @ beta_u) ** 2))

    k_u = x_unrestricted.shape[1]
    dfd = n_eff - k_u
    dfn = n_lags
    f_stat = ((rss_r - rss_u) / dfn) / (rss_u / dfd)
    f_stat = max(f_stat, 0.0)
    p_value = float(f_dist.sf(f_stat, dfn, dfd))
    return float(f_stat), p_value
