"""Tier-1 statistics target: `newey_west_cov`, against statsmodels.

Newey-West HAC covariance uses the Bartlett kernel `w_h = 1 - |h|/(L+1)` and no small-sample
correction.
"""

import numpy as np
import numpy.typing as npt
from statsmodels.stats.sandwich_covariance import S_hac_simple

import quantcore.statistics as qc_statistics
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
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
