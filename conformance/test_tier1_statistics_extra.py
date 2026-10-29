"""Tier-1 differential tests for the statistics functions a downstream pipeline calls.

`ols` and `newey_west_optimal_lags` against `statsmodels`, `jarque_bera_test`
against `scipy`. The Ljung-Box, ARCH-LM and KPSS tests are computed by `statsmodels` itself, so
their checks are the convention tests in `tests/unit/statistics` and `tests/unit/time_series`.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from conformance.compare import registry
from conformance.generators import (
    REGRESSION_FAMILIES,
    RETURN_FAMILIES,
    TaggedRegression,
    TaggedReturns,
    length_bucket_for,
    regression_problem,
    returns_series,
)
from conformance.harness import check

_OLS_COEFFICIENTS = registry()["ols_coefficients"]
_OLS_R_SQUARED = registry()["ols_r_squared"]
_NEWEY_WEST_OPTIMAL_LAGS = registry()["newey_west_optimal_lags"]
_JARQUE_BERA = registry()["jarque_bera"]

OLS_MIN_LEN = 3
OLS_FAMILIES: tuple[str, ...] = REGRESSION_FAMILIES
OLS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

NEWEY_WEST_OPTIMAL_LAGS_MIN_LEN = 1
NEWEY_WEST_OPTIMAL_LAGS_FAMILIES: tuple[str, ...] = ("sample_size",)
NEWEY_WEST_OPTIMAL_LAGS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

JARQUE_BERA_MIN_LEN = 2
JARQUE_BERA_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
JARQUE_BERA_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(problem=regression_problem(min_len=OLS_MIN_LEN))
def test_ols_matches_statsmodels(problem: TaggedRegression) -> None:
    """Quantcore's `coefficients` and `r_squared` against `statsmodels.api.OLS`."""
    check(
        _OLS_COEFFICIENTS,
        problem.y,
        problem.x,
        family=problem.family,
        length_bucket=length_bucket_for(problem.x.shape[0]),
        seed=problem.seed,
    )
    check(
        _OLS_R_SQUARED,
        problem.y,
        problem.x,
        family=problem.family,
        length_bucket=length_bucket_for(problem.x.shape[0]),
        seed=problem.seed,
    )


@given(n=st.integers(min_value=1, max_value=5000))
def test_newey_west_optimal_lags_matches_hand_formula(n: int) -> None:
    """Quantcore's `newey_west_optimal_lags` against the rule-of-thumb formula itself."""
    check(
        _NEWEY_WEST_OPTIMAL_LAGS,
        n,
        family="sample_size",
        length_bucket=length_bucket_for(n),
        seed=None,
    )


@given(tagged=returns_series(min_len=JARQUE_BERA_MIN_LEN))
def test_jarque_bera_matches_scipy(tagged: TaggedReturns) -> None:
    """Quantcore's `jarque_bera_test` against `scipy.stats.jarque_bera`."""
    # A constant series has no skewness or kurtosis: quantcore raises and scipy returns NaN.
    assume(np.ptp(tagged.values) > 0.0)
    check(
        _JARQUE_BERA,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("statistic", "p_value"),
    )
