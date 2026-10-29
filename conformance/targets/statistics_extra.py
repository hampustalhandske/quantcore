"""Tier-1 statistics targets: OLS, the Newey-West lag rule, Jarque-Bera and the HAC t-test.

OLS is two targets, `ols_coefficients` and `ols_r_squared` (centred R-squared with a constant
column, else uncentred). The Newey-West lag count is the rule of thumb
`floor(4 * (n/100)^(2/9))`, not the data-driven procedure. `jarque_bera` is checked
against `scipy.stats.jarque_bera`. `hac_t_test` is checked against a `statsmodels` constant-only
OLS with `cov_type="HAC"`, Bartlett kernel, `use_correction=True` and `use_t=True`.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from scipy.stats import jarque_bera as _scipy_jarque_bera
from scipy.stats import t as _student_t

import quantcore.statistics as qc_statistics
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


# --- HAC t-test -----------------------------------------------------------------------------------


def _hac_t_test_quantcore(
    x: npt.NDArray[np.float64],
    *,
    lags: int,
    alternative: Literal["two-sided", "greater", "less"],
    null: float,
) -> tuple[float, float, float]:
    result = qc_statistics.hac_t_test(x, lags=lags, alternative=alternative, null=null)
    return result.t, result.p_value, result.se


def _hac_t_test_reference(
    x: npt.NDArray[np.float64],
    *,
    lags: int,
    alternative: Literal["two-sided", "greater", "less"],
    null: float,
) -> tuple[float, float, float]:
    fit = sm.OLS(x, np.ones(x.size)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags, "use_correction": True, "kernel": "bartlett"},
        use_t=True,
    )
    test = fit.t_test((np.ones((1, 1)), np.array([null])), use_t=True)
    t_stat = float(np.squeeze(test.tvalue))
    if alternative == "two-sided":
        p_value = float(np.squeeze(test.pvalue))
    else:
        sign = 1.0 if alternative == "greater" else -1.0
        p_value = float(_student_t.sf(sign * t_stat, fit.df_resid))
    return t_stat, p_value, float(fit.bse[0])


_HAC_T_TEST = ConformanceTarget(
    name="hac_t_test",
    tier=1,
    oracle="statsmodels OLS cov_type='HAC'",
    citation=(
        "Newey, W.K. and West, K.D. (1987), 'A Simple, Positive Semi-Definite, Heteroskedasticity "
        "and Autocorrelation Consistent Covariance Matrix.' Econometrica, 55(3), 703-708."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator(), ScalarComparator()),
    quantcore=_hac_t_test_quantcore,
    reference=_hac_t_test_reference,
)


# --- ARCH-LM --------------------------------------------------------------------------------------


# --- KPSS -----------------------------------------------------------------------------------------
