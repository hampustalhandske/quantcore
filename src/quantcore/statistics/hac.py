"""HAC t-test of a sample mean against a null value."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

from quantcore.statistics.regression import newey_west_cov, newey_west_optimal_lags

_MIN_OBS = 10


@dataclass(frozen=True)
class HacTResult:
    """Result of `hac_t_test`: t-statistic, p-value, lags used, n, mean and HAC standard error."""

    t: float
    p_value: float
    lags: int
    n: int
    mean: float
    se: float


def hac_t_test(
    x: npt.NDArray[np.float64],
    *,
    lags: int | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    null: float = 0.0,
) -> HacTResult:
    """Test `mean(x) == null` with a Newey-West (Bartlett kernel) standard error of the mean.

    The standard error is the HAC covariance of a constant-only regression, scaled by n/(n-1);
    the p-value uses Student-t with n-1 degrees of freedom. This equals statsmodels' OLS fit with
    `cov_type="HAC"`, `use_correction=True`, `use_t=True`; at `lags=0` it is `ttest_1samp`.

    Args:
        x: Observations, 1-D, finite, at least 10.
        lags: Bartlett lags, 0 <= lags < n. None uses `newey_west_optimal_lags`.
        alternative: "two-sided" (mean != null), "greater" (mean > null) or "less" (mean < null).
        null: Finite value the mean is tested against.

    Returns:
        `HacTResult` with the t-statistic of `mean - null`, its p-value, the lags used, n, the
        sample mean and its HAC standard error.

    Raises:
        ValueError: Invalid `x`, `lags`, `alternative` or `null`, or a non-positive HAC variance.
    """
    if x.ndim != 1 or x.size < _MIN_OBS or not np.isfinite(x).all():
        raise ValueError(f"x must be a finite 1-D array of at least {_MIN_OBS} observations")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}"
        )
    if not np.isfinite(null):
        raise ValueError("null must be finite")
    n = x.size
    n_lags = newey_west_optimal_lags(x) if lags is None else lags
    if not 0 <= n_lags < n:
        raise ValueError(f"lags must satisfy 0 <= lags < n={n}, got {n_lags}")
    mean = float(x.mean())
    cov = newey_west_cov(np.ones((n, 1)), x - mean, n_lags, small_sample_correction=True)
    variance = float(cov[0, 0])
    if not (np.isfinite(variance) and variance > 0.0):
        raise ValueError("Newey-West variance is not positive")
    se = variance**0.5
    t_stat = (mean - null) / se
    tail = {"two-sided": -abs(t_stat), "greater": -t_stat, "less": t_stat}[alternative]
    p_value = float(student_t.cdf(tail, n - 1)) * (2.0 if alternative == "two-sided" else 1.0)
    return HacTResult(t_stat, p_value, n_lags, n, mean, se)
