"""Tier-1 differential tests for the statistics functions a downstream pipeline calls.

`ols`, `newey_west_optimal_lags` and `hac_t_test` against `statsmodels`, `jarque_bera_test`
against `scipy`, `cscv_pbo` against a hand CSCV with `scipy.stats.rankdata` mid-ranks and
`effective_n_eigen` against the trace identity. The Ljung-Box, ARCH-LM and KPSS tests are
computed by `statsmodels` itself, so their checks are the convention tests in
`tests/unit/statistics` and `tests/unit/time_series`.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

import conformance.targets.statistics_extra  # noqa: F401
from conformance.compare import registry
from conformance.generators import (
    COVARIANCE_FAMILIES,
    REGRESSION_FAMILIES,
    RETURN_FAMILIES,
    TaggedPanel,
    TaggedRegression,
    TaggedReturns,
    length_bucket_for,
    regression_problem,
    returns_panel,
    returns_series,
)
from conformance.harness import check

_OLS_COEFFICIENTS = registry()["ols_coefficients"]
_OLS_R_SQUARED = registry()["ols_r_squared"]
_NEWEY_WEST_OPTIMAL_LAGS = registry()["newey_west_optimal_lags"]
_JARQUE_BERA = registry()["jarque_bera"]
_HAC_T_TEST = registry()["hac_t_test"]
_CSCV_PBO = registry()["cscv_pbo"]
_EFFECTIVE_N_EIGEN = registry()["effective_n_eigen"]

OLS_MIN_LEN = 3
OLS_FAMILIES: tuple[str, ...] = REGRESSION_FAMILIES
OLS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

NEWEY_WEST_OPTIMAL_LAGS_MIN_LEN = 1
NEWEY_WEST_OPTIMAL_LAGS_FAMILIES: tuple[str, ...] = ("sample_size",)
NEWEY_WEST_OPTIMAL_LAGS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

JARQUE_BERA_MIN_LEN = 2
JARQUE_BERA_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
JARQUE_BERA_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

HAC_T_TEST_MIN_LEN = 10
HAC_T_TEST_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
HAC_T_TEST_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

CSCV_PBO_MIN_LEN = 20
CSCV_PBO_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
CSCV_PBO_BUCKETS: tuple[str, ...] = ("medium", "long")

EFFECTIVE_N_EIGEN_MIN_LEN = 20
EFFECTIVE_N_EIGEN_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
EFFECTIVE_N_EIGEN_BUCKETS: tuple[str, ...] = ("medium", "long")


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


@given(
    tagged=returns_series(min_len=HAC_T_TEST_MIN_LEN),
    lags=st.integers(min_value=0, max_value=30),
    alternative=st.sampled_from(["two-sided", "greater", "less"]),
    null_scale=st.floats(min_value=-2.0, max_value=2.0),
)
def test_hac_t_test_matches_statsmodels(
    tagged: TaggedReturns, lags: int, alternative: str, null_scale: float
) -> None:
    """Quantcore's `hac_t_test` against statsmodels OLS with `cov_type="HAC"` (Bartlett)."""
    # A constant series has zero HAC variance; quantcore raises and statsmodels returns NaN.
    assume(np.ptp(tagged.values) > 0.0)
    assume(lags < tagged.n)
    check(
        _HAC_T_TEST,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("t", "p_value", "se"),
        lags=lags,
        alternative=alternative,
        null=null_scale * float(tagged.values.std()),
    )


@given(
    tagged=returns_panel(min_len=CSCV_PBO_MIN_LEN, max_len=1500, max_k=10),
    n_splits=st.sampled_from([2, 4, 6, 8, 10]),
)
def test_cscv_pbo_matches_hand_cscv(tagged: TaggedPanel, n_splits: int) -> None:
    """Quantcore's `cscv_pbo` against a hand CSCV with `scipy.stats.rankdata` mid-ranks."""
    check(
        _CSCV_PBO,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        n_splits=n_splits,
    )


@given(tagged=returns_panel(min_len=EFFECTIVE_N_EIGEN_MIN_LEN, max_k=30))
def test_effective_n_eigen_matches_trace_identity(tagged: TaggedPanel) -> None:
    """Quantcore's `effective_n_eigen` against `trace(C)^2 / ||C||_F^2`."""
    check(
        _EFFECTIVE_N_EIGEN,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )
