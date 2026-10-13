"""Time-series targets: factor regressions, ARIMA, VAR, Granger causality, spread mean reversion.

Factor loadings are OLS with Newey-West HAC inference (normal p-values); Fama-MacBeth uses plain
time-series standard errors; ARIMA is a zero-mean ARMA on the differenced series (no constant),
fit by exact Gaussian maximum likelihood, with the MA sign convention
`theta(B) = 1 + theta_1*B + ...` that statsmodels shares. quantcore's `var_fit` puts the VAR
intercept in the last column (`_var_layout_from_statsmodels` converts statsmodels' layout once).

`arima_fit` is an iterative estimator: its optimizer and statsmodels' stop at slightly different
points near the same maximum, so its parameter vector is reported but not compared. quantcore's
solution and statsmodels' own are both scored by the same statsmodels exact Gaussian state-space
log-likelihood, and that attained log-likelihood is the quantity that decides conformance.
`arima_forecast` is compared with the forecast of statsmodels' state-space ARIMA model filtered
(never fit) at the same coefficients, the exact conditional mean on both sides. `arima_residuals`
returns conditional (sum-of-squares) residuals, which statsmodels does not compute; its oracle is
`scipy.signal.lfilter` run as the same inverse ARMA filter from the same initial conditions.
`select_arima_order` and `select_var_lag_order` return one of possibly several
criterion-optimal orders (near-ties are common), so their agreement with the oracle is reported,
not asserted (`ReportOnlyScalarComparator`).

Each target's `citation` names its source in short form: Fama & French (1993, 2015) and Newey &
West (1987) for factor loadings; Fama & MacBeth (1973); Box, Jenkins & Reinsel (2015), Gardner,
Harvey & Phillips (1980) and Harvey (1989) for ARIMA;
Lutkepohl (2005) and Granger (1969) for VAR and Granger causality; Chan (2013) for the OU
half-life. `docs/REFERENCES.md` gives the full entry for each of these except Chan (2013), which
the `ou_half_life` citation spells out itself.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.signal import lfilter, lfiltic
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.var_model import forecast as sm_var_forecast

import quantcore.statistics as qc_statistics
import quantcore.time_series as qc_time_series
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    DecisiveFloatArrayComparator,
    ReportOnlyScalarComparator,
    ScalarComparator,
    TupleComparator,
)

__all__ = ["ArimaFitResult"]


# factor_loadings: registered as six targets (loadings, standard errors, t-stats, p-values,
# R-squared, residuals), so each `Comparator` is homogeneous.


def _sm_factor_loadings_fit(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> sm.regression.linear_model.RegressionResultsWrapper:
    design = sm.add_constant(factors, has_constant="add")
    return sm.OLS(returns, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": n_lags_nw, "use_correction": False}
    )


def _factor_loadings_loadings_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_statistics.factor_loadings(returns, factors, n_lags_nw).loadings, dtype=np.float64
    )


def _factor_loadings_loadings_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(_sm_factor_loadings_fit(returns, factors, n_lags_nw).params, dtype=np.float64)


_FACTOR_LOADINGS_LOADINGS = ConformanceTarget(
    name="factor_loadings_loadings",
    tier=1,
    oracle="statsmodels (OLS with an intercept column)",
    citation="Fama, E.F. and French, K.R. (1993, 2015). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_factor_loadings_loadings_quantcore,
    reference=_factor_loadings_loadings_reference,
)


def _factor_loadings_standard_errors_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_statistics.factor_loadings(returns, factors, n_lags_nw).standard_errors,
        dtype=np.float64,
    )


def _factor_loadings_standard_errors_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(_sm_factor_loadings_fit(returns, factors, n_lags_nw).bse, dtype=np.float64)


_FACTOR_LOADINGS_STANDARD_ERRORS = ConformanceTarget(
    name="factor_loadings_standard_errors",
    tier=1,
    oracle="statsmodels (OLS, cov_type='HAC', use_correction=False)",
    citation="Newey, W.K. and West, K.D. (1987). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_factor_loadings_standard_errors_quantcore,
    reference=_factor_loadings_standard_errors_reference,
)


def _factor_loadings_t_stats_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_statistics.factor_loadings(returns, factors, n_lags_nw).t_stats, dtype=np.float64
    )


def _factor_loadings_t_stats_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    fit = _sm_factor_loadings_fit(returns, factors, n_lags_nw)
    return np.asarray(fit.tvalues, dtype=np.float64)


_FACTOR_LOADINGS_T_STATS = ConformanceTarget(
    name="factor_loadings_t_stats",
    tier=1,
    oracle="statsmodels (OLS, cov_type='HAC', use_correction=False)",
    citation="Newey, W.K. and West, K.D. (1987). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_factor_loadings_t_stats_quantcore,
    reference=_factor_loadings_t_stats_reference,
)


def _factor_loadings_p_values_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_statistics.factor_loadings(returns, factors, n_lags_nw).p_values, dtype=np.float64
    )


def _factor_loadings_p_values_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    fit = _sm_factor_loadings_fit(returns, factors, n_lags_nw)
    return np.asarray(fit.pvalues, dtype=np.float64)


_FACTOR_LOADINGS_P_VALUES = ConformanceTarget(
    name="factor_loadings_p_values",
    tier=1,
    oracle="statsmodels (OLS, cov_type='HAC', use_correction=False; two-sided normal p-values)",
    citation="Newey, W.K. and West, K.D. (1987). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_factor_loadings_p_values_quantcore,
    reference=_factor_loadings_p_values_reference,
)


def _factor_loadings_r_squared_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> float:
    return float(qc_statistics.factor_loadings(returns, factors, n_lags_nw).r_squared)


def _factor_loadings_r_squared_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> float:
    return float(_sm_factor_loadings_fit(returns, factors, n_lags_nw).rsquared)


_FACTOR_LOADINGS_R_SQUARED = ConformanceTarget(
    name="factor_loadings_r_squared",
    tier=1,
    oracle="statsmodels (OLS with an intercept column, centred R-squared)",
    citation="Gauss-Markov theorem; standard OLS coefficient of determination.",
    compare=ScalarComparator(),
    quantcore=_factor_loadings_r_squared_quantcore,
    reference=_factor_loadings_r_squared_reference,
)


def _factor_loadings_residuals_quantcore(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_statistics.factor_loadings(returns, factors, n_lags_nw).residuals, dtype=np.float64
    )


def _factor_loadings_residuals_reference(
    returns: npt.NDArray[np.float64], factors: npt.NDArray[np.float64], n_lags_nw: int
) -> npt.NDArray[np.float64]:
    return np.asarray(_sm_factor_loadings_fit(returns, factors, n_lags_nw).resid, dtype=np.float64)


_FACTOR_LOADINGS_RESIDUALS = ConformanceTarget(
    name="factor_loadings_residuals",
    tier=1,
    oracle="statsmodels (OLS with an intercept column)",
    citation="Gauss-Markov theorem; standard OLS residuals.",
    compare=ArrayComparator(),
    quantcore=_factor_loadings_residuals_quantcore,
    reference=_factor_loadings_residuals_reference,
)


# --- fama_macbeth_regression ------------------------------------------------------------------


def _fama_macbeth_mean_risk_premia_quantcore(
    returns: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    mean_risk_premia, _t_stats = qc_statistics.fama_macbeth_regression(returns, betas)
    return np.asarray(mean_risk_premia, dtype=np.float64)


def _cross_sectional_lambdas(
    returns: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Per-period cross-sectional risk premia `lambda_t`, one OLS-on-`betas` fit per period."""
    n_periods = returns.shape[0]
    n_factors = betas.shape[1]
    lambdas = np.empty((n_periods, n_factors), dtype=np.float64)
    for t in range(n_periods):
        lambda_t, _residuals, _rank, _singular = np.linalg.lstsq(betas, returns[t, :], rcond=None)
        lambdas[t, :] = lambda_t
    return lambdas


def _fama_macbeth_mean_risk_premia_reference(
    returns: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Mean of the per-period cross-sectional OLS slopes (`linearmodels` unadjusted)."""
    return np.asarray(np.mean(_cross_sectional_lambdas(returns, betas), axis=0), dtype=np.float64)


_FAMA_MACBETH_MEAN_RISK_PREMIA = ConformanceTarget(
    name="fama_macbeth_mean_risk_premia",
    tier=2,
    oracle="linearmodels.FamaMacBeth(cov_type='unadjusted') (cross-checked, see test module)",
    citation="Fama, E.F. and MacBeth, J.D. (1973). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_fama_macbeth_mean_risk_premia_quantcore,
    reference=_fama_macbeth_mean_risk_premia_reference,
)


def _fama_macbeth_standard_errors_quantcore(
    returns: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    mean_risk_premia, t_stats = qc_statistics.fama_macbeth_regression(returns, betas)
    return np.asarray(mean_risk_premia / t_stats, dtype=np.float64)


def _fama_macbeth_standard_errors_reference(
    returns: npt.NDArray[np.float64], betas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Plain time-series standard error of the mean risk premia: `std(lambda_t, ddof=1)/sqrt(T)`.

    The original Fama & MacBeth (1973) convention `quantcore`'s default (`n_lags_nw=None`)
    implements, verified against `linearmodels.FamaMacBeth(cov_type='unadjusted')` in the
    conformance test itself.
    """
    lambdas = _cross_sectional_lambdas(returns, betas)
    n_periods = lambdas.shape[0]
    standard_errors: npt.NDArray[np.float64] = lambdas.std(axis=0, ddof=1) / np.sqrt(n_periods)
    return standard_errors


_FAMA_MACBETH_STANDARD_ERRORS = ConformanceTarget(
    name="fama_macbeth_standard_errors",
    tier=2,
    oracle="linearmodels.FamaMacBeth(cov_type='unadjusted') (cross-checked, see test module)",
    citation="Fama, E.F. and MacBeth, J.D. (1973). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_fama_macbeth_standard_errors_quantcore,
    reference=_fama_macbeth_standard_errors_reference,
)


# --- ARIMA -----------------------------------------------------------------------------------


def _difference(series: npt.NDArray[np.float64], d: int) -> npt.NDArray[np.float64]:
    diffed = series
    for _ in range(d):
        diffed = np.diff(diffed)
    return diffed


def _arima_loglike_model(series: npt.NDArray[np.float64], p: int, d: int, q: int) -> ARIMA:
    return ARIMA(series, order=(p, d, q), trend="n", concentrate_scale=False)


ArimaFitResult = tuple[float, npt.NDArray[np.float64]]
"""`(log_likelihood, params)` where `params = concatenate([ar_coefs, ma_coefs, [sigma2_hat]])`
and `log_likelihood` is `statsmodels.tsa.arima.model.ARIMA(...).loglike` at those parameters; see
the module docstring on why the log-likelihood, not the parameters, is compared.
"""


def _arima_fit_quantcore(series: npt.NDArray[np.float64], p: int, d: int, q: int) -> ArimaFitResult:
    ar_coefs, ma_coefs, sigma2_hat = qc_time_series.arima_fit(series, p, d, q)
    params = np.concatenate([ar_coefs, ma_coefs, [sigma2_hat]])
    log_likelihood = float(_arima_loglike_model(series, p, d, q).loglike(params))
    return log_likelihood, params.astype(np.float64)


def _arima_fit_reference(series: npt.NDArray[np.float64], p: int, d: int, q: int) -> ArimaFitResult:
    model = _arima_loglike_model(series, p, d, q)
    result = model.fit()
    params = np.asarray(result.params, dtype=np.float64)
    return float(result.llf), params


_ARIMA_FIT = ConformanceTarget(
    name="arima_fit",
    tier=2,
    oracle="statsmodels (ARIMA, exact Gaussian state-space likelihood, both solutions evaluated "
    "under the SAME model's loglike -- see module docstring)",
    citation="Gardner, Harvey & Phillips (1980); Box, Jenkins & Reinsel (2015). See module "
    "docstring.",
    compare=DecisiveFloatArrayComparator(),
    quantcore=_arima_fit_quantcore,
    reference=_arima_fit_reference,
)


def _arima_residuals_quantcore(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_time_series.arima_residuals(series, ar_coefs, ma_coefs, d), dtype=np.float64
    )


def _arima_residuals_reference(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
) -> npt.NDArray[np.float64]:
    """Conditional residuals by `scipy.signal.lfilter`, an independent implementation.

    The conditional-sum-of-squares residuals (Box, Jenkins & Reinsel 2015) are the output of the
    inverse ARMA filter `phi(B) / theta(B)` applied to the differenced series `w`: numerator
    `(1, -phi_1, ..., -phi_p)`, denominator `(1, theta_1, ..., theta_q)`. quantcore's convention
    starts the recursion at index `max(p, q)`, with every earlier residual taken as 0 and the
    earlier values of `w` used as they are; `lfiltic` builds the filter state for exactly those
    initial conditions. `lfilter` evaluates the recursion in transposed direct form II, a
    different arrangement of the arithmetic than a loop over the difference equation.
    """
    w = _difference(series, d)
    p, q = ar_coefs.shape[0], ma_coefs.shape[0]
    start = max(p, q)
    numerator = np.concatenate([[1.0], -ar_coefs])
    denominator = np.concatenate([[1.0], ma_coefs])
    initial_state = lfiltic(numerator, denominator, y=np.zeros(q), x=w[start - p : start][::-1])
    residuals, _ = lfilter(numerator, denominator, w[start:], zi=initial_state)
    return np.asarray(residuals, dtype=np.float64)


_ARIMA_RESIDUALS = ConformanceTarget(
    name="arima_residuals",
    tier=2,
    oracle="scipy.signal.lfilter (the inverse ARMA filter phi(B)/theta(B) in transposed direct "
    "form II, initial conditions from lfiltic; statsmodels computes no conditional residuals "
    "-- see `_arima_residuals_reference`)",
    citation="Box, Jenkins & Reinsel (2015), conditional sum-of-squares residual recursion. See "
    "module docstring.",
    compare=ArrayComparator(),
    quantcore=_arima_residuals_quantcore,
    reference=_arima_residuals_reference,
)


def _arima_forecast_quantcore(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_time_series.arima_forecast(series, ar_coefs, ma_coefs, d, h), dtype=np.float64
    )


def _arima_forecast_reference(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Exact conditional-mean forecast from statsmodels' state-space ARIMA at fixed coefficients.

    `SARIMAX`, the model `statsmodels.tsa.arima.model.ARIMA` wraps, is filtered at the given
    coefficients (never fit) and its `forecast` is the Gaussian conditional mean of the next `h`
    values given the whole series. Two of its options make that exact instead of approximate:
    `use_exact_diffuse=True` gives the integrated states (`d > 0`) an exactly diffuse prior
    instead of a variance of 1e6, and `tolerance=0.0` keeps the filter updating its gain at every
    step instead of freezing it at an early steady-state estimate, which leaves an error of 1e-10
    to 1e-9 in the forecast. The forecast does not depend on the innovation variance, set to 1.
    """
    p, q = ar_coefs.shape[0], ma_coefs.shape[0]
    model = SARIMAX(series, order=(p, d, q), trend="n", use_exact_diffuse=True, tolerance=0.0)
    result = model.filter(np.concatenate([ar_coefs, ma_coefs, [1.0]]))
    return np.asarray(result.forecast(h), dtype=np.float64)


_ARIMA_FORECAST = ConformanceTarget(
    name="arima_forecast",
    tier=2,
    oracle="statsmodels (SARIMAX filtered at the given coefficients, exact diffuse initial "
    "state, steady-state shortcut off, then forecast -- see `_arima_forecast_reference`)",
    citation="Box, Jenkins & Reinsel (2015), minimum mean square error forecast; Harvey (1989), "
    "prediction from the Kalman filter state. See module docstring.",
    compare=ArrayComparator(),
    quantcore=_arima_forecast_quantcore,
    reference=_arima_forecast_reference,
)


def _select_arima_order_quantcore(
    series: npt.NDArray[np.float64], max_p: int, max_d: int, max_q: int, criterion: str
) -> float:
    """quantcore's selected `(p, d, q)`, encoded as the single number `p*100 + d*10 + q`."""
    p, d, q = qc_time_series.select_arima_order(series, max_p, max_d, max_q, criterion)
    return float(p * 100 + d * 10 + q)


def _select_arima_order_reference(
    series: npt.NDArray[np.float64], max_p: int, max_d: int, max_q: int, criterion: str
) -> float:
    """Grid-search over `statsmodels.tsa.arima.model.ARIMA.fit`'s own reported `aic`/`bic`.

    The same information-criterion definition quantcore's docstring cites (k free parameters
    including the intercept-free mean, `-2*loglik + 2k` / `+ k*ln(n)`), evaluated on every
    feasible `(p, d, q)` up to the given bounds.
    """
    best_order: tuple[int, int, int] | None = None
    best_score = np.inf
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                if series.shape[0] <= d + max(p, q) + 1:
                    continue
                try:
                    fit = ARIMA(series, order=(p, d, q), trend="n").fit()
                except Exception:
                    # a numerically infeasible order; any of them just excludes this candidate.
                    continue
                score = fit.aic if criterion == "aic" else fit.bic
                if score < best_score:
                    best_score = score
                    best_order = (p, d, q)
    if best_order is None:
        raise ValueError("no feasible (p, d, q) order found for the given series and bounds")
    p, d, q = best_order
    return float(p * 100 + d * 10 + q)


_SELECT_ARIMA_ORDER = ConformanceTarget(
    name="select_arima_order",
    tier=2,
    oracle="statsmodels (ARIMA.fit, aic/bic; agreement rate reported, not asserted -- see "
    "module docstring)",
    citation="Box, Jenkins & Reinsel (2015). See module docstring.",
    compare=ReportOnlyScalarComparator(),
    quantcore=_select_arima_order_quantcore,
    reference=_select_arima_order_reference,
)


# --- VAR -------------------------------------------------------------------------------------


def _var_layout_from_statsmodels(
    params: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reorder statsmodels' VAR coefficient layout into quantcore's intercept-last layout.

    Statsmodels' `(1+k*n_lags, k)` `params` (intercept in the first row, each lag block with rows
    = source variable, columns = target equation) becomes quantcore's `(k, k*n_lags+1)` layout,
    intercept in the last column. A lossless transpose/stack.
    """
    return np.column_stack([params[1:, :].T, params[0, :]])


def _var_fit_coef_matrix_quantcore(
    data: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    coef_matrix, _sigma_u = qc_time_series.var_fit(data, n_lags)
    return np.asarray(coef_matrix, dtype=np.float64)


def _var_fit_coef_matrix_reference(
    data: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    result = VAR(data).fit(n_lags)
    return _var_layout_from_statsmodels(np.asarray(result.params, dtype=np.float64))


_VAR_FIT_COEF_MATRIX = ConformanceTarget(
    name="var_fit_coef_matrix",
    tier=2,
    oracle="statsmodels.tsa.api.VAR (params reordered to quantcore's intercept-last layout, "
    "finding H -- see module docstring)",
    citation="Lutkepohl, H. (2005). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_var_fit_coef_matrix_quantcore,
    reference=_var_fit_coef_matrix_reference,
)


def _var_fit_sigma_u_quantcore(
    data: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    _coef_matrix, sigma_u = qc_time_series.var_fit(data, n_lags)
    return np.asarray(sigma_u, dtype=np.float64)


def _var_fit_sigma_u_reference(
    data: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    return np.asarray(VAR(data).fit(n_lags).sigma_u, dtype=np.float64)


_VAR_FIT_SIGMA_U = ConformanceTarget(
    name="var_fit_sigma_u",
    tier=2,
    oracle="statsmodels.tsa.api.VAR (sigma_u, the unbiased residual covariance estimator)",
    citation="Lutkepohl, H. (2005), Section 3.2.2. See module docstring.",
    compare=ArrayComparator(),
    quantcore=_var_fit_sigma_u_quantcore,
    reference=_var_fit_sigma_u_reference,
)


def _var_forecast_quantcore(
    data: npt.NDArray[np.float64], coef_matrix: npt.NDArray[np.float64], n_lags: int, h: int
) -> npt.NDArray[np.float64]:
    return np.asarray(qc_time_series.var_forecast(data, coef_matrix, n_lags, h), dtype=np.float64)


def _var_forecast_reference(
    data: npt.NDArray[np.float64], coef_matrix: npt.NDArray[np.float64], n_lags: int, h: int
) -> npt.NDArray[np.float64]:
    k = data.shape[1]
    coefs = np.stack([coef_matrix[:, (lag - 1) * k : lag * k] for lag in range(1, n_lags + 1)])
    trend = coef_matrix[:, -1]
    return np.asarray(sm_var_forecast(data[-n_lags:], coefs, trend, h), dtype=np.float64)


_VAR_FORECAST = ConformanceTarget(
    name="var_forecast",
    tier=2,
    oracle="statsmodels.tsa.vector_ar.var_model.forecast (same coefficients, no re-fit)",
    citation="Lutkepohl, H. (2005), p. 37. See module docstring.",
    compare=ArrayComparator(),
    quantcore=_var_forecast_quantcore,
    reference=_var_forecast_reference,
)


def _impulse_response_quantcore(
    coef_matrix: npt.NDArray[np.float64],
    sigma_u: npt.NDArray[np.float64],
    n_lags: int,
    n_periods: int,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_time_series.impulse_response(coef_matrix, sigma_u, n_lags, n_periods), dtype=np.float64
    )


def _impulse_response_reference(
    coef_matrix: npt.NDArray[np.float64],
    sigma_u: npt.NDArray[np.float64],
    n_lags: int,
    n_periods: int,
) -> npt.NDArray[np.float64]:
    k = sigma_u.shape[0]
    coefs = np.stack([coef_matrix[:, (lag - 1) * k : lag * k] for lag in range(1, n_lags + 1)])
    trend = coef_matrix[:, -1]
    process = VARProcess(coefs, trend, sigma_u)
    return np.asarray(process.orth_ma_rep(n_periods - 1), dtype=np.float64)


_IMPULSE_RESPONSE = ConformanceTarget(
    name="impulse_response",
    tier=2,
    oracle="statsmodels.tsa.vector_ar.var_model.VARProcess.orth_ma_rep (Cholesky-orthogonalized)",
    citation="Lutkepohl, H. (2005). See module docstring.",
    compare=ArrayComparator(),
    quantcore=_impulse_response_quantcore,
    reference=_impulse_response_reference,
)


def _select_var_lag_order_quantcore(
    system: npt.NDArray[np.float64], max_lags: int, criterion: str
) -> float:
    return float(qc_time_series.select_var_lag_order(system, max_lags, criterion))


def _select_var_lag_order_reference(
    system: npt.NDArray[np.float64], max_lags: int, criterion: str
) -> float:
    result = VAR(system).select_order(maxlags=max_lags)
    return float(result.aic if criterion == "aic" else result.bic)


_SELECT_VAR_LAG_ORDER = ConformanceTarget(
    name="select_var_lag_order",
    tier=2,
    oracle="statsmodels.tsa.api.VAR.select_order (agreement rate reported, not asserted -- see "
    "module docstring)",
    citation="Lutkepohl, H. (2005), pp. 146-150. See module docstring.",
    compare=ReportOnlyScalarComparator(),
    quantcore=_select_var_lag_order_quantcore,
    reference=_select_var_lag_order_reference,
)


# --- Granger causality -------------------------------------------------------------------------


def _granger_causality_test_quantcore(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64], n_lags: int
) -> tuple[float, float]:
    f_statistic, p_value = qc_time_series.granger_causality_test(y, x, n_lags)
    return float(f_statistic), float(p_value)


def _granger_causality_test_reference(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64], n_lags: int
) -> tuple[float, float]:
    data = np.column_stack([y, x])
    result = grangercausalitytests(data, maxlag=[n_lags])
    f_statistic, p_value, _df_denom, _df_num = result[n_lags][0]["ssr_ftest"]
    return float(f_statistic), float(p_value)


_GRANGER_CAUSALITY_TEST: ConformanceTarget[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], int], tuple[float, float]
] = ConformanceTarget(
    name="granger_causality_test",
    tier=2,
    oracle="statsmodels.tsa.stattools.grangercausalitytests (ssr_ftest)",
    citation="Granger, C.W.J. (1969). See module docstring.",
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_granger_causality_test_quantcore,
    reference=_granger_causality_test_reference,
)


# --- OU half-life and spread z-score ------------------------------------------------------------


def _ou_half_life_quantcore(spread: npt.NDArray[np.float64]) -> float:
    return float(qc_statistics.ou_half_life(spread))


def _ou_half_life_reference(spread: npt.NDArray[np.float64]) -> float:
    """`statsmodels.OLS`'s slope on the lagged series, then `-ln(2)/ln(1+a)`.

    An independent library's own OLS fit, not a hand-rolled `numpy.linalg.lstsq` re-derivation of
    quantcore's own documented AR(1) formula (Uhlenbeck & Ornstein 1930; Chan 2013, ch. 2).
    """
    delta = spread[1:] - spread[:-1]
    level = spread[:-1]
    design = sm.add_constant(level)
    fit = sm.OLS(delta, design).fit()
    a = float(fit.params[1])
    phi = 1.0 + a
    if a >= 0.0 or phi <= 0.0:
        raise ValueError("estimated series does not exhibit mean reversion (phi >= 0, diverging)")
    return float(-np.log(2.0) / np.log(phi))


_OU_HALF_LIFE = ConformanceTarget(
    name="ou_half_life",
    tier=2,
    oracle="statsmodels.OLS slope on the lagged series",
    citation="Chan, E.P. (2013), Algorithmic Trading, ch. 2. See module docstring.",
    compare=ScalarComparator(),
    quantcore=_ou_half_life_quantcore,
    reference=_ou_half_life_reference,
)


def _spread_zscore_quantcore(
    spread: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """quantcore's rolling z-score without its unfilled first `window - 1` entries."""
    full = qc_statistics.spread_zscore(spread, window)
    result: npt.NDArray[np.float64] = full[window - 1 :]
    return result


def _spread_zscore_reference(
    spread: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """`pandas.Series.rolling(window)`'s own mean and `ddof=1` std: an independent rolling z-score.

    Not a hand-rolled loop re-deriving quantcore's own documented formula. `.std()`'s `ddof=1`
    is pandas' rolling-window default and is passed explicitly for clarity; a zero-variance
    window (constant segment) gives `NaN`, matching quantcore's own documented behaviour there.
    """
    series = pd.Series(spread)
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.apply(lambda segment: segment.std(ddof=1), raw=True)
    z = (series - mean) / std
    result: npt.NDArray[np.float64] = z.to_numpy()[window - 1 :]
    return result


_SPREAD_ZSCORE = ConformanceTarget(
    name="spread_zscore",
    tier=2,
    oracle="pandas.Series.rolling(window).apply(...) with ddof=1",
    citation="Standard rolling z-score; ddof=1 sample standard deviation.",
    compare=ArrayComparator(),
    quantcore=_spread_zscore_quantcore,
    reference=_spread_zscore_reference,
)
