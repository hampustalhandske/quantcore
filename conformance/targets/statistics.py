"""Tier-1 statistics targets: `newey_west_cov` and `engle_granger_test`, against statsmodels.

Newey-West HAC covariance uses the Bartlett kernel `w_h = 1 - |h|/(L+1)` and no small-sample
correction. Engle-Granger regresses `y` on `x` with a constant, then runs an ADF test without a
constant on the residuals (1 lagged difference), with the Engle-Granger/MacKinnon `N = 2`
distribution (the plain Dickey-Fuller one is anti-conservative here).
"""

import numpy as np
import numpy.typing as npt
from statsmodels.stats.sandwich_covariance import S_hac_simple
from statsmodels.tsa.stattools import coint

import quantcore.statistics as qc_statistics
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    ScalarComparator,
    TupleComparator,
)


def _newey_west_cov_quantcore(
    x: npt.NDArray[np.float64], residuals: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = qc_statistics.newey_west_cov(x, residuals, n_lags)
    return result


def _newey_west_cov_reference(
    x: npt.NDArray[np.float64], residuals: npt.NDArray[np.float64], n_lags: int
) -> npt.NDArray[np.float64]:
    xe = x * residuals[:, None]
    meat = S_hac_simple(xe, nlags=n_lags)
    bread = np.linalg.inv(x.T @ x)
    result: npt.NDArray[np.float64] = bread @ meat @ bread
    return result


_NEWEY_WEST_COV = ConformanceTarget(
    name="newey_west_cov",
    tier=1,
    oracle="statsmodels",
    citation=(
        "Newey, W.K. and West, K.D. (1987), 'A Simple, Positive Semi-Definite, "
        "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.' "
        "Econometrica, 55(3), 703-708."
    ),
    compare=ArrayComparator(),
    quantcore=_newey_west_cov_quantcore,
    reference=_newey_west_cov_reference,
)


def _engle_granger_test_quantcore(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float]:
    _beta, statistic, p_value = qc_statistics.engle_granger_test(y, x)
    return statistic, p_value


def _engle_granger_test_reference(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[float, float]:
    statistic, p_value, _critical_values = coint(y, x, trend="c", maxlag=1, autolag=None)
    return float(statistic), float(p_value)


_ENGLE_GRANGER_TEST: ConformanceTarget[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], tuple[float, float]
] = ConformanceTarget(
    name="engle_granger_test",
    tier=1,
    oracle="statsmodels",
    citation=(
        "Engle, R.F. and Granger, C.W.J. (1987), 'Co-Integration and Error Correction: "
        "Representation, Estimation, and Testing.' Econometrica, 55(2), 251-276; "
        "MacKinnon, J.G. (1994), 'Approximate Asymptotic Distribution Functions for Unit-Root "
        "and Cointegration Tests.' Journal of Business & Economic Statistics, 12(2), 167-176."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_engle_granger_test_quantcore,
    reference=_engle_granger_test_reference,
)
