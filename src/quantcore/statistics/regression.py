"""Cross-sectional and time-series regression tools for factor models.

OLS (Gauss-Markov):
    beta = (x^T x)^{-1} x^T y
    R^2  = 1 - SS_res/SS_tot

Newey-West HAC covariance (Newey & West 1987):
    V_NW = (x^T x)^{-1} * S * (x^T x)^{-1}
    S = n * [Gamma_0 + sum_{h=1}^L w_h*(Gamma_h + Gamma_h^T)]
    w_h = 1 - h/(L+1)  (Bartlett kernel),  Gamma_h = (1/n)*sum_t x_t*x_t^T*e_t*e_{t-h}

Fama-MacBeth (1973):
    Step 1 — time-series: estimate beta_i = loadings for each asset
    Step 2 — cross-section: each period t: regress r_{i,t} on beta_i to get lambda_t
    Mean risk premia: lambda_bar = (1/T)*sum_t lambda_t
    t-stat: lambda_bar / (std(lambda_t) / sqrt(T))

References:
    Fama, E.F. and French, K.R. (1993, 2015); Fama, E.F. and MacBeth, J.D.
    (1973); Newey, W.K. and West, K.D. (1987). See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_ols_inputs(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
    if y.size == 0 or x.size == 0:
        raise ValueError("y and x must be non-empty")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations")
    if x.shape[0] < x.shape[1]:
        raise ValueError(
            "underdetermined system: number of observations must be >= number of regressors"
        )


def ols(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Ordinary least squares regression.

    Args:
        y: Dependent variable, shape (T,).
        x: Regressors, shape (T, k) (include an intercept column if desired).

    Returns:
        Tuple (coefficients, residuals, r_squared).
    """
    _validate_ols_inputs(y, x)
    return _ols(y, x)


def _ols(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return coefficients, residuals, r_squared


def _validate_newey_west_inputs(
    x: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    n_lags: int,
) -> None:
    if x.size == 0 or residuals.size == 0:
        raise ValueError("x and residuals must be non-empty")
    if x.shape[0] != residuals.shape[0]:
        raise ValueError("x and residuals must have the same number of observations")
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")


def newey_west_cov(
    x: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    n_lags: int,
) -> npt.NDArray[np.float64]:
    """Newey-West heteroskedasticity- and autocorrelation-consistent covariance of beta_hat.

    Args:
        x: Regressors used to estimate beta_hat, shape (T, k).
        residuals: OLS residuals, shape (T,).
        n_lags: Number of Bartlett-kernel lags (L).

    Returns:
        HAC covariance matrix of beta_hat, shape (k, k).
    """
    _validate_newey_west_inputs(x, residuals, n_lags)
    return _newey_west_cov(x, residuals, n_lags)


def _newey_west_cov(
    x: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    n_lags: int,
) -> npt.NDArray[np.float64]:
    n_obs = x.shape[0]
    xe = x * residuals[:, None]  # x_t * e_t, shape (T, k)

    gamma_0 = (xe.T @ xe) / n_obs
    weighted_sum = gamma_0.copy()
    for h in range(1, n_lags + 1):
        gamma_h = (xe[h:].T @ xe[:-h]) / n_obs
        weight = 1.0 - h / (n_lags + 1)
        weighted_sum += weight * (gamma_h + gamma_h.T)
    weighted_sum *= n_obs

    xtx_inv = np.linalg.inv(x.T @ x)
    result: npt.NDArray[np.float64] = xtx_inv @ weighted_sum @ xtx_inv
    return result


def _validate_factor_loadings_inputs(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
) -> None:
    if returns.size == 0 or factors.size == 0:
        raise ValueError("returns and factors must be non-empty")
    if returns.shape[0] != factors.shape[0]:
        raise ValueError("returns and factors must have the same number of observations")


def factor_loadings(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    n_lags_nw: int = 4,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """OLS time-series regression of asset returns on factor returns.

    Args:
        returns: Asset excess returns, shape (T,).
        factors: Factor excess returns, shape (T, f).
        n_lags_nw: Number of Newey-West lags for HAC standard errors.

    Returns:
        Tuple (loadings, t_stats): intercept (alpha) + f factor betas, and
        their Newey-West t-statistics, each shape (f+1,).
    """
    _validate_factor_loadings_inputs(returns, factors)
    return _factor_loadings(returns, factors, n_lags_nw)


def _factor_loadings(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    n_lags_nw: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_obs = returns.shape[0]
    design = np.column_stack([np.ones(n_obs), factors])
    loadings, residuals, _ = _ols(returns, design)
    cov = _newey_west_cov(design, residuals, n_lags_nw)
    standard_errors = np.sqrt(np.diag(cov))
    t_stats = loadings / standard_errors
    return loadings, t_stats


def _validate_fama_macbeth_inputs(
    returns: npt.NDArray[np.float64],
    betas: npt.NDArray[np.float64],
) -> None:
    if returns.size == 0 or betas.size == 0:
        raise ValueError("returns and betas must be non-empty")
    if returns.shape[1] != betas.shape[0]:
        raise ValueError("returns' asset count must match betas' asset count")


def fama_macbeth_regression(
    returns: npt.NDArray[np.float64],
    betas: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fama-MacBeth two-pass cross-sectional regression.

    Args:
        returns: Asset returns, shape (T, N) — T periods, N assets.
        betas: Precomputed factor loadings, shape (N, f).

    Returns:
        Tuple (mean_risk_premia, t_stats): time-averaged cross-sectional
        slope for each factor, and its Fama-MacBeth t-statistic, each
        shape (f,).
    """
    _validate_fama_macbeth_inputs(returns, betas)
    return _fama_macbeth_regression(returns, betas)


def _fama_macbeth_regression(
    returns: npt.NDArray[np.float64],
    betas: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_periods = returns.shape[0]
    n_factors = betas.shape[1]
    lambdas = np.empty((n_periods, n_factors), dtype=np.float64)
    for t in range(n_periods):
        lambda_t, _, _, _ = np.linalg.lstsq(betas, returns[t, :], rcond=None)
        lambdas[t, :] = lambda_t

    mean_risk_premia = lambdas.mean(axis=0)
    std_risk_premia = lambdas.std(axis=0, ddof=1)
    t_stats = mean_risk_premia / (std_risk_premia / np.sqrt(n_periods))
    return mean_risk_premia, t_stats
