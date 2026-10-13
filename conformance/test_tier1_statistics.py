"""Tier-1 differential tests: `newey_west_cov`, `engle_granger_test` against `statsmodels`.

The `newey_west_cov` comparison is anchored twice: once directly (quantcore vs. the
bread-meat-bread reference construction on arbitrary `(x, residuals)`), and once by fitting an
actual `statsmodels` OLS with `cov_type="HAC"` and requiring `cov_params()` to equal both
quantcore and the reference on that fit's own residuals, so the reference construction is not
just self-consistent but anchored to the documented statsmodels estimator.
"""

import numpy as np
import statsmodels.api as sm
from hypothesis import given
from hypothesis import strategies as st

import conformance.targets.statistics  # noqa: F401
from conformance.compare import registry
from conformance.generators import (
    COINTEGRATION_FAMILIES,
    REGRESSION_FAMILIES,
    TaggedPair,
    TaggedRegression,
    cointegration_pair,
    length_bucket_for,
    regression_problem,
)
from conformance.harness import check

_NEWEY_WEST_COV = registry()["newey_west_cov"]
_ENGLE_GRANGER_TEST = registry()["engle_granger_test"]

# Declared coverage: the smallest input length each test's generator draws, and the families and
# length buckets it is expected to record at least one case in.
NEWEY_WEST_COV_MIN_LEN = 3
"""Smallest `x` row count the `newey_west_cov` tests draw: `regression_problem` always draws a
2-column `x` (intercept + one regressor), and a meaningful estimate needs strictly more
observations than regressors (`n_obs > n_regressors`) -- at `n_obs == n_regressors` the OLS fit is
exact, every residual is identically zero, and the "covariance" is a degenerate all-zero matrix
(quantcore does not raise there, it returns that degenerate matrix)."""
NEWEY_WEST_COV_FAMILIES: tuple[str, ...] = REGRESSION_FAMILIES
NEWEY_WEST_COV_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

ENGLE_GRANGER_TEST_MIN_LEN = 10
"""Smallest `y`/`x` length `engle_granger_test` accepts: quantcore's own
`_validate_engle_granger_inputs` rejects fewer than 10 observations."""
ENGLE_GRANGER_TEST_FAMILIES: tuple[str, ...] = COINTEGRATION_FAMILIES
ENGLE_GRANGER_TEST_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    problem=regression_problem(min_len=NEWEY_WEST_COV_MIN_LEN),
    n_lags=st.integers(min_value=0, max_value=10),
)
def test_newey_west_cov_matches_statsmodels(problem: TaggedRegression, n_lags: int) -> None:
    """Quantcore's `newey_west_cov` against a bread-meat-bread reference built from statsmodels."""
    beta, _residuals, _rank, _singular = np.linalg.lstsq(problem.x, problem.y, rcond=None)
    residuals = problem.y - problem.x @ beta
    check(
        _NEWEY_WEST_COV,
        problem.x,
        residuals,
        n_lags,
        family=problem.family,
        length_bucket=length_bucket_for(problem.x.shape[0]),
        seed=problem.seed,
    )


@given(
    problem=regression_problem(min_len=NEWEY_WEST_COV_MIN_LEN),
    n_lags=st.integers(min_value=0, max_value=10),
)
def test_newey_west_cov_anchored_to_statsmodels_ols_hac(
    problem: TaggedRegression, n_lags: int
) -> None:
    """Quantcore's and the oracle's `newey_west_cov` reproduce `statsmodels` OLS `cov_type="HAC"`.

    This anchors the oracle: it proves the reference construction used in
    `test_newey_west_cov_matches_statsmodels` is not merely self-consistent but equal to the
    estimator `statsmodels` itself documents and ships.
    """
    fit = sm.OLS(problem.y, problem.x).fit(
        cov_type="HAC", cov_kwds={"maxlags": n_lags, "use_correction": False}
    )
    expected = fit.cov_params()

    for provider_name in ("quantcore", "reference"):
        provider = getattr(_NEWEY_WEST_COV, provider_name)
        if provider is None:
            continue
        actual = provider(problem.x, fit.resid, n_lags)
        assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10), (
            f"{provider_name} newey_west_cov does not match statsmodels OLS cov_type='HAC': "
            f"actual={actual!r} expected={expected!r}"
        )


@given(pair=cointegration_pair(min_len=ENGLE_GRANGER_TEST_MIN_LEN))
def test_engle_granger_test_matches_statsmodels(pair: TaggedPair) -> None:
    """Quantcore's (statistic, p-value) against `statsmodels.tsa.stattools.coint`."""
    check(
        _ENGLE_GRANGER_TEST,
        pair.y,
        pair.x,
        family=pair.family,
        length_bucket=length_bucket_for(pair.y.shape[0]),
        seed=pair.seed,
        quantities=("statistic", "p_value"),
    )
