"""Tier-1 statistics targets: OLS, the Newey-West lag rule, Ljung-Box, ADF and three diagnostics.

OLS is two targets, `ols_coefficients` and `ols_r_squared` (centred R-squared with a constant
column, else uncentred). The Newey-West lag count is the rule of thumb
`floor(4 * (n/100)^(2/9))`, not the data-driven procedure. Ljung-Box uses `model_df=0`; the
ADF test is constant-only, fixed-lag, with MacKinnon (1994) p-values.

The diagnostics are `jarque_bera` against `scipy.stats.jarque_bera`, `arch_lm` against
`statsmodels.stats.diagnostic.het_arch`, and `kpss` against `statsmodels.tsa.stattools.kpss`.
ARCH-LM and KPSS take the lag count as an argument, `None` meaning each side's own default
(`min(10, n // 5)` and the automatic bandwidth); KPSS also takes the null hypothesis, `"c"`
(level) or `"ct"` (trend). KPSS's third quantity is the flag saying the p-value sits at an end
of the published table, recorded as `1.0` or `0.0` so that every component is a float.
"""

import warnings

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from scipy.stats import jarque_bera as _scipy_jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as _sm_kpss

import quantcore.statistics as qc_statistics
import quantcore.time_series as qc_time_series
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    ScalarComparator,
    TupleComparator,
)

# OLS: two targets (coefficients and R-squared), as their comparators differ.


def _ols_coefficients_quantcore(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    coefficients, _residuals, _r_squared = qc_statistics.ols(y, x)
    result: npt.NDArray[np.float64] = coefficients
    return result


def _ols_coefficients_reference(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    fit = sm.OLS(y, x).fit()
    return np.asarray(fit.params, dtype=np.float64)


_OLS_COEFFICIENTS = ConformanceTarget(
    name="ols_coefficients",
    tier=1,
    oracle="statsmodels",
    citation="Gauss-Markov theorem; standard OLS estimator.",
    compare=ArrayComparator(),
    quantcore=_ols_coefficients_quantcore,
    reference=_ols_coefficients_reference,
)


def _ols_r_squared_quantcore(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    _coefficients, _residuals, r_squared = qc_statistics.ols(y, x)
    return float(r_squared)


def _ols_r_squared_reference(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    return float(sm.OLS(y, x).fit().rsquared)


_OLS_R_SQUARED = ConformanceTarget(
    name="ols_r_squared",
    tier=1,
    oracle="statsmodels",
    citation="Gauss-Markov theorem; standard OLS estimator's coefficient of determination.",
    compare=ScalarComparator(),
    quantcore=_ols_r_squared_quantcore,
    reference=_ols_r_squared_reference,
)


# --- Newey-West optimal lag rule of thumb --------------------------------------------------------


def _newey_west_optimal_lags_quantcore(n: int) -> float:
    return float(qc_statistics.newey_west_optimal_lags(np.zeros(n, dtype=np.float64)))


def _newey_west_optimal_lags_reference(n: int) -> float:
    return float(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


_NEWEY_WEST_OPTIMAL_LAGS = ConformanceTarget(
    name="newey_west_optimal_lags",
    tier=1,
    oracle="hand-computed (the rule of thumb itself)",
    citation=(
        "Newey, W.K. and West, K.D. (1994), 'Automatic Lag Selection in Covariance Matrix "
        "Estimation.' Review of Economic Studies, 61(4), 631-653."
    ),
    compare=ScalarComparator(),
    quantcore=_newey_west_optimal_lags_quantcore,
    reference=_newey_west_optimal_lags_reference,
)


# --- Ljung-Box ------------------------------------------------------------------------------------


def _ljung_box_test_quantcore(
    residuals: npt.NDArray[np.float64], n_lags: int
) -> tuple[float, float]:
    statistic, p_value = qc_time_series.ljung_box_test(residuals, n_lags, model_df=0)
    return float(statistic), float(p_value)


def _ljung_box_test_reference(
    residuals: npt.NDArray[np.float64], n_lags: int
) -> tuple[float, float]:
    table = acorr_ljungbox(residuals, lags=[n_lags], return_df=True)
    return float(table["lb_stat"].iloc[-1]), float(table["lb_pvalue"].iloc[-1])


_LJUNG_BOX_TEST = ConformanceTarget(
    name="ljung_box_test",
    tier=1,
    oracle="statsmodels",
    citation=(
        "Ljung, G.M. and Box, G.E.P. (1978), 'On a Measure of Lack of Fit in Time Series "
        "Models.' Biometrika, 65(2), 297-303."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_ljung_box_test_quantcore,
    reference=_ljung_box_test_reference,
)


# --- Augmented Dickey-Fuller ---------------------------------------------------------------


def _adf_test_quantcore(series: npt.NDArray[np.float64], max_lags: int) -> tuple[float, float]:
    statistic, p_value = qc_statistics.adf_test(series, max_lags, trend="c", autolag=None)
    return float(statistic), float(p_value)


def _adf_test_reference(series: npt.NDArray[np.float64], max_lags: int) -> tuple[float, float]:
    # `autolag=None` makes adfuller return the 5-tuple (statistic, p_value, used_lag, n_obs,
    # critical_values).
    result = adfuller(series, maxlag=max_lags, regression="c", autolag=None, result_object=False)
    statistic, p_value = result[0], result[1]
    return float(statistic), float(p_value)


_ADF_TEST = ConformanceTarget(
    name="adf_test",
    tier=1,
    oracle="statsmodels",
    citation=(
        "Dickey, D.A. and Fuller, W.A. (1979), 'Distribution of the Estimators for "
        "Autoregressive Time Series with a Unit Root.' JASA, 74(366a), 427-431; MacKinnon, J.G. "
        "(1994), 'Approximate Asymptotic Distribution Functions for Unit-Root and Cointegration "
        "Tests.' Journal of Business & Economic Statistics, 12(2), 167-176."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_adf_test_quantcore,
    reference=_adf_test_reference,
)


# --- Jarque-Bera ----------------------------------------------------------------------------------


def _jarque_bera_quantcore(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    statistic, p_value = qc_statistics.jarque_bera_test(series)
    return float(statistic), float(p_value)


def _jarque_bera_reference(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    result = _scipy_jarque_bera(series)
    return float(result.statistic), float(result.pvalue)


_JARQUE_BERA = ConformanceTarget(
    name="jarque_bera",
    tier=1,
    oracle="scipy.stats.jarque_bera",
    citation=(
        "Jarque, C.M. and Bera, A.K. (1980), 'Efficient Tests for Normality, Homoscedasticity "
        "and Serial Independence of Regression Residuals.' Economics Letters, 6(3), 255-259; "
        "Jarque, C.M. and Bera, A.K. (1987), 'A Test for Normality of Observations and "
        "Regression Residuals.' International Statistical Review, 55(2), 163-172."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_jarque_bera_quantcore,
    reference=_jarque_bera_reference,
)


# --- ARCH-LM --------------------------------------------------------------------------------------


def _arch_lm_quantcore(
    residuals: npt.NDArray[np.float64], n_lags: int | None
) -> tuple[float, float]:
    statistic, p_value = qc_statistics.arch_lm_test(residuals, n_lags)
    return float(statistic), float(p_value)


def _arch_lm_reference(
    residuals: npt.NDArray[np.float64], n_lags: int | None
) -> tuple[float, float]:
    lm_stat, lm_pvalue, _f_stat, _f_pvalue = het_arch(residuals, nlags=n_lags, result_object=False)
    return float(lm_stat), float(lm_pvalue)


_ARCH_LM = ConformanceTarget(
    name="arch_lm",
    tier=1,
    oracle="statsmodels.stats.diagnostic.het_arch",
    citation=(
        "Engle, R.F. (1982), 'Autoregressive Conditional Heteroscedasticity with Estimates of "
        "the Variance of United Kingdom Inflation.' Econometrica, 50(4), 987-1007."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_arch_lm_quantcore,
    reference=_arch_lm_reference,
)


# --- KPSS -----------------------------------------------------------------------------------------


def _kpss_quantcore(
    series: npt.NDArray[np.float64], n_lags: int | None, trend: str
) -> tuple[float, float, float]:
    statistic, p_value, at_bound = qc_statistics.kpss_test(series, n_lags, trend)
    return float(statistic), float(p_value), 1.0 if at_bound else 0.0


def _kpss_reference(
    series: npt.NDArray[np.float64], n_lags: int | None, trend: str
) -> tuple[float, float, float]:
    # statsmodels reports a p-value outside its table as the nearest tabulated one and says so
    # only through an `InterpolationWarning`; the warning is the oracle's at-bound flag.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, p_value, _lags, _crit = _sm_kpss(
            series,
            regression=trend,
            nlags="auto" if n_lags is None else n_lags,
            result_object=False,
        )
    at_bound = any(issubclass(w.category, InterpolationWarning) for w in caught)
    return float(statistic), float(p_value), 1.0 if at_bound else 0.0


_KPSS = ConformanceTarget(
    name="kpss",
    tier=1,
    oracle="statsmodels.tsa.stattools.kpss",
    citation=(
        "Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., and Shin, Y. (1992), 'Testing the Null "
        "Hypothesis of Stationarity against the Alternative of a Unit Root.' Journal of "
        "Econometrics, 54(1-3), 159-178; Hobijn, B., Franses, P.H., and Ooms, M. (2004), "
        "'Generalizations of the KPSS-test for Stationarity.' Statistica Neerlandica, 58(4), "
        "483-502 (automatic bandwidth)."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator(), ScalarComparator()),
    quantcore=_kpss_quantcore,
    reference=_kpss_reference,
)
