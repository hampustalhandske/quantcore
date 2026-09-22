"""Cross-sectional and time-series regression tools for factor models.

OLS (Gauss-Markov):
    beta = (x^T x)^{-1} x^T y
    R^2  = 1 - SS_res/SS_tot, SS_tot centered (sum((y - mean(y))^2)) if `x`
    has an intercept column (any column constant across all rows),
    uncentered (sum(y^2)) otherwise — matching the standard convention
    (e.g. statsmodels) that R^2 without an intercept measures fit against
    the origin, not against the mean. NaN, not 1.0, when SS_tot is exactly
    0 (y constant in the centered case, or all-zero in the uncentered
    case): R^2 is undefined there, not a perfect fit by definition.

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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


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
        Tuple (coefficients, residuals, r_squared). `r_squared` uses the
        centered convention (against `y`'s mean) if `x` has an intercept
        column (any column constant across all rows), the uncentered
        convention (against the origin) otherwise — see
        `_has_intercept_column`. `r_squared` is `NaN`, not `1.0`, when its
        denominator is exactly 0: a constant `y` (centered case) or an
        all-zero `y` (uncentered case) has an undefined R², not a perfect
        fit by definition. This is a documented value, not an error —
        callers that can't accept `NaN` must check for it explicitly.
    """
    _validate_ols_inputs(y, x)
    return _ols(y, x)


def _has_intercept_column(x: npt.NDArray[np.float64]) -> bool:
    """Whether any column of `x` is constant across all rows.

    Same simple check statsmodels applies before falling back to its rank
    test for an implicit (non-column-of-ones) constant; quantcore does not
    replicate that fallback, so a constant that's only implicit in a
    combination of columns (e.g. a full set of dummy variables) is not
    detected here and `ols` will use the uncentered R^2 convention for it.
    """
    if x.shape[1] == 0:
        return False
    return bool(np.any(np.ptp(x, axis=0) == 0.0))


def _ols(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    if _has_intercept_column(x):
        ss_tot = float(np.sum((y - y.mean()) ** 2))
    else:
        ss_tot = float(np.sum(y**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
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
    small_sample_correction: bool = False,
) -> npt.NDArray[np.float64]:
    """Newey-West heteroskedasticity- and autocorrelation-consistent covariance of beta_hat.

    Args:
        x: Regressors used to estimate beta_hat, shape (T, k).
        residuals: OLS residuals, shape (T,).
        n_lags: Number of Bartlett-kernel lags (L).
        small_sample_correction: If True, scale the covariance matrix by
            `T / (T - k)` (T = number of observations, k = number of
            regressors) — a simple degrees-of-freedom correction for small
            samples, matching `statsmodels`' `cov_type="HAC"` with
            `use_correction=True`. Default False (no correction) matches
            the original, only behavior.

    Returns:
        HAC covariance matrix of beta_hat, shape (k, k).
    """
    _validate_newey_west_inputs(x, residuals, n_lags)
    return _newey_west_cov(x, residuals, n_lags, small_sample_correction)


def _newey_west_cov(
    x: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    n_lags: int,
    small_sample_correction: bool = False,
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
    if small_sample_correction:
        k = x.shape[1]
        result = result * (n_obs / (n_obs - k))
    return result


def newey_west_optimal_lags(residuals: npt.NDArray[np.float64]) -> int:
    """Newey & West's (1994) closed-form rule-of-thumb bandwidth for HAC covariance.

    L = floor(4 * (n/100)^(2/9)), a function of the sample size n alone —
    `residuals`' *values* are not used, only its length. This is the
    fixed-form rule from Newey & West (1994), not the paper's separate
    fully data-driven procedure (an AR(1)-approximation to the residual
    autocorrelation, in the style of Andrews 1991), which would use the
    residual values themselves and is not implemented here. If a
    data-adaptive bandwidth is what you need, compute one independently and
    pass it directly to `newey_west_cov`'s `n_lags` — this function only
    ever returns the sample-size-based rule of thumb.

    Args:
        residuals: Residual series the HAC covariance will be estimated
            from; only `residuals.size` is used.

    Returns:
        Suggested number of Bartlett-kernel lags L (non-negative integer).

    References:
        Newey, W.K. and West, K.D. (1994). "Automatic Lag Selection in
        Covariance Matrix Estimation." *Review of Economic Studies*, 61(4),
        631-653. See docs/REFERENCES.md.
    """
    if residuals.size == 0:
        raise ValueError("residuals must be non-empty")
    n = residuals.size
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(lags, 0)


def _validate_factor_loadings_inputs(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
) -> None:
    if returns.size == 0 or factors.size == 0:
        raise ValueError("returns and factors must be non-empty")
    if returns.shape[0] != factors.shape[0]:
        raise ValueError("returns and factors must have the same number of observations")


@dataclass(frozen=True)
class FactorLoadingsResult:
    """OLS time-series factor regression result, with Newey-West HAC inference.

    Attributes:
        loadings: Intercept (alpha) + f factor betas, shape (f+1,).
        standard_errors: Newey-West HAC standard errors of `loadings`,
            shape (f+1,).
        t_stats: `loadings / standard_errors`, shape (f+1,).
        p_values: Two-sided p-values from the standard normal distribution
            (the standard convention for asymptotic HAC-based inference —
            the Newey-West covariance itself is only asymptotically valid,
            so a finite-sample Student-t reference adds a precision the
            estimator doesn't have; matches `statsmodels`' HAC default),
            shape (f+1,).
        r_squared: Centered R^2 of the regression (the design always
            includes an intercept column, so this is always the centered
            convention — see `ols`).
        residuals: OLS residuals, shape (T,).
        n_lags_nw: Newey-West lag count actually used.
    """

    loadings: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    t_stats: npt.NDArray[np.float64]
    p_values: npt.NDArray[np.float64]
    r_squared: float
    residuals: npt.NDArray[np.float64]
    n_lags_nw: int


def factor_loadings(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    n_lags_nw: int | None = None,
) -> FactorLoadingsResult:
    """OLS time-series regression of asset returns on factor returns.

    Args:
        returns: Asset excess returns, shape (T,).
        factors: Factor excess returns, shape (T, f).
        n_lags_nw: Number of Newey-West lags for HAC standard errors. `None`
            (default) selects `newey_west_optimal_lags(residuals)`
            automatically, rather than a fixed lag count regardless of
            sample size.

    Returns:
        `FactorLoadingsResult`.
    """
    _validate_factor_loadings_inputs(returns, factors)
    return _factor_loadings(returns, factors, n_lags_nw)


def _factor_loadings(
    returns: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    n_lags_nw: int | None,
) -> FactorLoadingsResult:
    n_obs = returns.shape[0]
    design = np.column_stack([np.ones(n_obs), factors])
    loadings, residuals, r_squared = _ols(returns, design)
    resolved_n_lags_nw = newey_west_optimal_lags(residuals) if n_lags_nw is None else n_lags_nw
    cov = _newey_west_cov(design, residuals, resolved_n_lags_nw)
    standard_errors = np.sqrt(np.diag(cov))
    t_stats = loadings / standard_errors
    p_values = 2.0 * norm.sf(np.abs(t_stats))
    return FactorLoadingsResult(
        loadings=loadings,
        standard_errors=standard_errors,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=r_squared,
        residuals=residuals,
        n_lags_nw=resolved_n_lags_nw,
    )


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
    n_lags_nw: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fama-MacBeth two-pass cross-sectional regression.

    Standard errors of the mean risk premia are plain time-series standard
    errors (`std(lambda_t, ddof=1) / sqrt(T)`) by default — the original,
    Fama & MacBeth (1973) convention, which assumes the period-by-period
    risk premia lambda_t are serially uncorrelated. If they aren't (a
    common concern this convention doesn't detect), the resulting t-stats
    overstate significance; `n_lags_nw` switches to Newey-West HAC standard
    errors of the mean instead, which don't assume that.

    Args:
        returns: Asset returns, shape (T, N) — T periods, N assets.
        betas: Precomputed factor loadings, shape (N, f).
        n_lags_nw: If None (default), use plain time-series standard
            errors of the mean risk premia (the original behavior). If an
            int, use Newey-West HAC standard errors of the mean with this
            many Bartlett-kernel lags instead — robust to serial
            correlation in lambda_t, which the plain time-series SE
            ignores. Shanken's (1992) errors-in-variables correction for
            using estimated (not true) betas is a separate, larger
            correction this function does not implement either way.

    Returns:
        Tuple (mean_risk_premia, t_stats): time-averaged cross-sectional
        slope for each factor, and its t-statistic (see `n_lags_nw`), each
        shape (f,).

    References:
        Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and
        Equilibrium: Empirical Tests." *Journal of Political Economy*,
        81(3), 607-636. Newey, W.K. and West, K.D. (1987). "A Simple,
        Positive Semi-Definite, Heteroskedasticity and Autocorrelation
        Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.
        (`n_lags_nw`.) See docs/REFERENCES.md.
    """
    _validate_fama_macbeth_inputs(returns, betas)
    if n_lags_nw is not None and n_lags_nw < 0:
        raise ValueError("n_lags_nw must be non-negative")
    return _fama_macbeth_regression(returns, betas, n_lags_nw)


def _fama_macbeth_regression(
    returns: npt.NDArray[np.float64],
    betas: npt.NDArray[np.float64],
    n_lags_nw: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_periods = returns.shape[0]
    n_factors = betas.shape[1]
    lambdas = np.empty((n_periods, n_factors), dtype=np.float64)
    for t in range(n_periods):
        lambda_t, _, _, _ = np.linalg.lstsq(betas, returns[t, :], rcond=None)
        lambdas[t, :] = lambda_t

    mean_risk_premia = lambdas.mean(axis=0)
    if n_lags_nw is None:
        std_risk_premia = lambdas.std(axis=0, ddof=1)
        standard_errors = std_risk_premia / np.sqrt(n_periods)
    else:
        # Newey-West HAC variance of the sample mean of each lambda_t
        # column: the HAC covariance of a "regression on a constant"
        # (design = ones(T, 1)), whose residuals are just lambda_t
        # demeaned -- `_newey_west_cov` already implements exactly that
        # covariance-of-a-coefficient machinery.
        design = np.ones((n_periods, 1))
        standard_errors = np.empty(n_factors, dtype=np.float64)
        for j in range(n_factors):
            residuals = lambdas[:, j] - mean_risk_premia[j]
            nw_var = _newey_west_cov(design, residuals, n_lags_nw)[0, 0]
            standard_errors[j] = float(np.sqrt(nw_var))

    t_stats = mean_risk_premia / standard_errors
    return mean_risk_premia, t_stats
