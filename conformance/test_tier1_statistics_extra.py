"""Tier-1 differential tests for the statistics functions a downstream pipeline calls.

`ols`, `newey_west_optimal_lags`, `ljung_box_test`, `adf_test`, `arch_lm_test` and `kpss_test`
against `statsmodels`, `jarque_bera_test` against `scipy`.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from conformance.compare import registry
from conformance.generators import (
    COINTEGRATION_FAMILIES,
    REGRESSION_FAMILIES,
    RETURN_FAMILIES,
    TaggedPair,
    TaggedRegression,
    TaggedReturns,
    cointegration_pair,
    length_bucket_for,
    regression_problem,
    returns_series,
)
from conformance.harness import check
from conformance.targets.statistics_extra import _adf_test_quantcore, _adf_test_reference

_OLS_COEFFICIENTS = registry()["ols_coefficients"]
_OLS_R_SQUARED = registry()["ols_r_squared"]
_NEWEY_WEST_OPTIMAL_LAGS = registry()["newey_west_optimal_lags"]
_LJUNG_BOX_TEST = registry()["ljung_box_test"]
_ADF_TEST = registry()["adf_test"]
_JARQUE_BERA = registry()["jarque_bera"]
_ARCH_LM = registry()["arch_lm"]
_KPSS = registry()["kpss"]

OLS_MIN_LEN = 3
OLS_FAMILIES: tuple[str, ...] = REGRESSION_FAMILIES
OLS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

NEWEY_WEST_OPTIMAL_LAGS_MIN_LEN = 1
NEWEY_WEST_OPTIMAL_LAGS_FAMILIES: tuple[str, ...] = ("sample_size",)
NEWEY_WEST_OPTIMAL_LAGS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

LJUNG_BOX_TEST_MIN_LEN = 3
LJUNG_BOX_TEST_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
LJUNG_BOX_TEST_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

ADF_TEST_MIN_LEN = 10
ADF_TEST_FAMILIES: tuple[str, ...] = ("stationary", "unit_root")
ADF_TEST_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

JARQUE_BERA_MIN_LEN = 2
JARQUE_BERA_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
JARQUE_BERA_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

ARCH_LM_MIN_LEN = 5
"""Shortest series the default lag count `min(10, n // 5)` gives at least one lag for."""
ARCH_LM_FAMILIES: tuple[str, ...] = (
    "normal",
    "student_t3",
    "skewed",
    "ar1",
    "garch11",
    "all_positive",
    "all_negative",
    "huge_scale",
)
"""Homoskedastic (`normal`, `student_t3`, `skewed`, `ar1` and the one-signed pair) and
conditionally heteroskedastic (`garch11`) series, plus `huge_scale` for squares of order 1e6.

Two return families are left out because the comparison says nothing about quantcore on them.
On `tiny_scale` the oracle is wrong: squares of order 1e-16 fall below the cutoff of the
pseudo-inverse statsmodels fits the auxiliary regression with, the lag columns are discarded and
`het_arch` returns a statistic of zero for any series, heteroskedastic or not. On `near_constant`
the squares vary by 5e-6 of their level, so R-squared is a ratio of two quantities known to
about `eps * n * 2e5` and two correct implementations differ by 1e-7 in the statistic.
`arch_lm_test` scales its input before squaring it; `tests/unit/statistics/test_diagnostics.py`
checks that its result does not depend on the scale of the residuals.
"""
ARCH_LM_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

KPSS_MIN_LEN = 3
KPSS_FAMILIES: tuple[str, ...] = RETURN_FAMILIES + COINTEGRATION_FAMILIES
"""Return series (stationary, the input a returns pipeline passes) and the first leg of the
cointegration pairs: a random walk (`independent_rw`), a random walk plus noise (`cointegrated`)
and an AR(1) with coefficient 0.995 (`near_unit_root`)."""
KPSS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


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


@given(
    tagged=returns_series(min_len=LJUNG_BOX_TEST_MIN_LEN),
    n_lags=st.integers(min_value=1, max_value=9),
)
def test_ljung_box_test_matches_statsmodels(tagged: TaggedReturns, n_lags: int) -> None:
    """Quantcore's `ljung_box_test` against `statsmodels.stats.diagnostic.acorr_ljungbox`."""
    assume(n_lags < tagged.n)
    check(
        _LJUNG_BOX_TEST,
        tagged.values,
        n_lags,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("statistic", "p_value"),
    )


@st.composite
def _adf_series(draw: st.DrawFn) -> tuple[np.ndarray, str, int, int]:
    family = draw(st.sampled_from(["stationary", "unit_root"]))
    n = draw(st.integers(min_value=ADF_TEST_MIN_LEN, max_value=1000))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, n)
    if family == "stationary":
        phi = 0.5
        series = np.empty(n, dtype=np.float64)
        series[0] = innovations[0]
        for t in range(1, n):
            series[t] = phi * series[t - 1] + innovations[t]
    else:
        series = np.cumsum(innovations)
    return series, family, seed, n


@given(data=_adf_series(), max_lags=st.integers(min_value=0, max_value=3))
def test_adf_test_matches_statsmodels(
    data: tuple[np.ndarray, str, int, int], max_lags: int
) -> None:
    """Quantcore's `(statistic, p_value)` against `statsmodels.tsa.stattools.adfuller`.

    The two quantities are recorded separately and judged at the function's one tolerance (see
    `conformance/tolerances.py`): the p-value is the same closed-form function of the statistic
    on both sides.
    """
    series, family, seed, n = data
    assume(n >= max_lags + 5)
    check(
        _ADF_TEST,
        series,
        max_lags,
        family=family,
        length_bucket=length_bucket_for(n),
        seed=seed,
        quantities=("statistic", "p_value"),
    )


def test_adf_test_decision_agreement_report() -> None:
    """Reject/accept agreement between quantcore and statsmodels at 1%/5%/10%.

    Over a fixed, seeded set of 300 stationary and unit-root series, the two must reach the same
    decision at every level: their p-values are the same function of statistics that agree to
    1e-11. The agreement rates are also printed, so they are visible in `pytest -s` and CI logs.
    """
    rng = np.random.default_rng(0)
    levels = (0.01, 0.05, 0.10)
    agreements = dict.fromkeys(levels, 0)
    n_series = 300
    for i in range(n_series):
        n = int(rng.integers(30, 300))
        family = "stationary" if i % 2 == 0 else "unit_root"
        innovations = rng.normal(0.0, 1.0, n)
        if family == "stationary":
            phi = 0.5
            series = np.empty(n, dtype=np.float64)
            series[0] = innovations[0]
            for t in range(1, n):
                series[t] = phi * series[t - 1] + innovations[t]
        else:
            series = np.cumsum(innovations)
        _stat_qc, p_qc = _adf_test_quantcore(series, 1)
        _stat_sm, p_sm = _adf_test_reference(series, 1)
        for level in levels:
            if (p_qc < level) == (p_sm < level):
                agreements[level] += 1

    rates = {level: count / n_series for level, count in agreements.items()}
    print(f"adf_test decision agreement over {n_series} generated series: {rates}")
    assert all(rate == 1.0 for rate in rates.values())


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
    tagged=returns_series(min_len=ARCH_LM_MIN_LEN, families=ARCH_LM_FAMILIES),
    n_lags=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
)
def test_arch_lm_matches_statsmodels(tagged: TaggedReturns, n_lags: int | None) -> None:
    """Quantcore's `arch_lm_test` against `statsmodels.stats.diagnostic.het_arch`.

    `n_lags=None` leaves the lag count to each side's default, `min(10, n // 5)`.
    """
    resolved_lags = min(10, tagged.n // 5) if n_lags is None else n_lags
    # quantcore requires more regression rows than coefficients, and squared values that vary
    # over the regression sample; statsmodels returns a perfect-fit statistic or -inf instead.
    assume(tagged.n >= 2 * resolved_lags + 2)
    assume(np.ptp(tagged.values[resolved_lags:] ** 2) > 0.0)
    check(
        _ARCH_LM,
        tagged.values,
        n_lags,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("statistic", "p_value"),
    )


@given(tagged=returns_series(min_len=KPSS_MIN_LEN))
def test_kpss_matches_statsmodels_on_returns(tagged: TaggedReturns) -> None:
    """Quantcore's `kpss_test` against `statsmodels.tsa.stattools.kpss` on return series.

    Level-stationarity null with the automatic lag count: the call a returns pipeline makes.
    """
    # A constant series leaves nothing to test: quantcore raises and statsmodels fails or
    # returns a statistic built from rounding residue.
    assume(np.ptp(tagged.values) > 0.0)
    check(
        _KPSS,
        tagged.values,
        None,
        "c",
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("statistic", "p_value", "p_value_at_bound"),
    )


@given(
    pair=cointegration_pair(min_len=KPSS_MIN_LEN + 1),
    trend=st.sampled_from(("c", "ct")),
    n_lags=st.one_of(st.none(), st.integers(min_value=0, max_value=12)),
)
def test_kpss_matches_statsmodels_on_unit_root_series(
    pair: TaggedPair, trend: str, n_lags: int | None
) -> None:
    """Quantcore's `kpss_test` against `statsmodels.tsa.stattools.kpss` on (near) unit-root series.

    Uses the first leg of each cointegration pair, under both null hypotheses and with the
    automatic lag count (`n_lags=None`) or a fixed one. The trend null needs four observations,
    one more than the level null.
    """
    series = pair.x
    assume(n_lags is None or n_lags < series.shape[0])
    check(
        _KPSS,
        series,
        n_lags,
        trend,
        family=pair.family,
        length_bucket=length_bucket_for(series.shape[0]),
        seed=pair.seed,
        quantities=("statistic", "p_value", "p_value_at_bound"),
    )
