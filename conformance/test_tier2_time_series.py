"""Differential tests for `conformance.targets.time_series` against its oracles.

Compares against `statsmodels`, `linearmodels` and, for `arima_residuals`, `scipy.signal`.

`factor_loadings` is Tier 1 (the cross-sectional risk model calls it); everything else in this
module (Fama-MacBeth, ARIMA, VAR, Granger causality, OU half-life, spread z-score) is Tier 2.

`arima_fit` follows `conformance.targets.risk_models`'s GARCH-family pattern: two attained
log-likelihoods recorded per case (`log_likelihood`, the decision quantity; `params`, reported,
never compared, since this is an iterative estimator without a unique optimum).
"""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from linearmodels import FamaMacBeth

import conformance.targets.time_series  # noqa: F401
import quantcore.statistics as qc_statistics
from conformance.compare import registry
from conformance.generators import (
    ARMA_FAMILIES,
    VAR_FAMILIES,
    TaggedArma,
    TaggedRegression,
    TaggedVarSystem,
    _length_strategy,
    arma_series,
    length_bucket_for,
    regression_problem,
    var_system,
)
from conformance.harness import check
from quantcore.time_series import var_fit as qc_var_fit

_FACTOR_LOADINGS_LOADINGS = registry()["factor_loadings_loadings"]
_FACTOR_LOADINGS_STANDARD_ERRORS = registry()["factor_loadings_standard_errors"]
_FACTOR_LOADINGS_T_STATS = registry()["factor_loadings_t_stats"]
_FACTOR_LOADINGS_P_VALUES = registry()["factor_loadings_p_values"]
_FACTOR_LOADINGS_R_SQUARED = registry()["factor_loadings_r_squared"]
_FACTOR_LOADINGS_RESIDUALS = registry()["factor_loadings_residuals"]
_FAMA_MACBETH_MEAN_RISK_PREMIA = registry()["fama_macbeth_mean_risk_premia"]
_FAMA_MACBETH_STANDARD_ERRORS = registry()["fama_macbeth_standard_errors"]
_ARIMA_FIT = registry()["arima_fit"]
_ARIMA_RESIDUALS = registry()["arima_residuals"]
_ARIMA_FORECAST = registry()["arima_forecast"]
_SELECT_ARIMA_ORDER = registry()["select_arima_order"]
_VAR_FIT_COEF_MATRIX = registry()["var_fit_coef_matrix"]
_VAR_FIT_SIGMA_U = registry()["var_fit_sigma_u"]
_VAR_FORECAST = registry()["var_forecast"]
_IMPULSE_RESPONSE = registry()["impulse_response"]
_SELECT_VAR_LAG_ORDER = registry()["select_var_lag_order"]
_GRANGER_CAUSALITY_TEST = registry()["granger_causality_test"]
_OU_HALF_LIFE = registry()["ou_half_life"]
_SPREAD_ZSCORE = registry()["spread_zscore"]

# Declared coverage: the smallest input length each test's generator draws, and the families and
# length buckets it is expected to record at least one case in.

# regression_problem() always draws a 2-column x (intercept + one regressor), i.e. one factor;
# factor_loadings needs at least 3 observations (as many as loadings to estimate: intercept + 1
# factor + >=1 residual df).
FACTOR_LOADINGS_MIN_LEN = 3
FACTOR_LOADINGS_FAMILIES: tuple[str, ...] = ("homoskedastic", "heteroskedastic", "autocorrelated")
FACTOR_LOADINGS_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


def _one_factor_case(problem: TaggedRegression) -> tuple[np.ndarray, np.ndarray]:
    """`regression_problem`'s intercept+regressor `x` split into `(returns, factors)`."""
    return problem.y, problem.x[:, 1:2]


@given(problem=regression_problem(min_len=FACTOR_LOADINGS_MIN_LEN), n_lags=st.integers(0, 10))
def test_factor_loadings_matches_statsmodels_hac_ols(
    problem: TaggedRegression, n_lags: int
) -> None:
    returns, factors = _one_factor_case(problem)
    bucket = length_bucket_for(returns.shape[0])
    kwargs = {
        "family": problem.family,
        "length_bucket": bucket,
        "seed": problem.seed,
    }
    check(_FACTOR_LOADINGS_LOADINGS, returns, factors, n_lags, **kwargs)
    check(_FACTOR_LOADINGS_STANDARD_ERRORS, returns, factors, n_lags, **kwargs)
    check(_FACTOR_LOADINGS_T_STATS, returns, factors, n_lags, **kwargs)
    check(_FACTOR_LOADINGS_P_VALUES, returns, factors, n_lags, **kwargs)
    check(_FACTOR_LOADINGS_R_SQUARED, returns, factors, n_lags, **kwargs)
    check(_FACTOR_LOADINGS_RESIDUALS, returns, factors, n_lags, **kwargs)


@given(problem=regression_problem(min_len=30, max_len=500))
def test_factor_loadings_auto_lag_selection_matches_rule_of_thumb(
    problem: TaggedRegression,
) -> None:
    """`n_lags_nw=None` resolves to `floor(4*(n/100)**(2/9))`, brief item 6/12's own rule.

    Calls quantcore's `newey_west_optimal_lags` directly (not through the backend dispatch, which
    would require every registered Tier-1 function -- including ones outside this family -- to
    already have a resolution-map entry) since this is a pure statement about the rule's formula,
    not a provider comparison `harness.check()` needs to record.
    """
    returns, _factors = _one_factor_case(problem)
    resolved = qc_statistics.newey_west_optimal_lags(returns)
    expected = int(np.floor(4.0 * (returns.shape[0] / 100.0) ** (2.0 / 9.0)))
    assert resolved == expected


# --- Fama-MacBeth -------------------------------------------------------------------------------

FAMA_MACBETH_MIN_LEN = 5
FAMA_MACBETH_FAMILIES: tuple[str, ...] = ("normal",)
FAMA_MACBETH_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@st.composite
def _fama_macbeth_case(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int, str]:
    n_periods = draw(_length_strategy(FAMA_MACBETH_MIN_LEN, 1000))
    n_assets = draw(st.integers(min_value=5, max_value=15))
    n_factors = draw(st.integers(min_value=1, max_value=3))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    betas = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    lambdas = rng.normal(0.0, 0.01, size=(n_periods, n_factors))
    returns = (betas @ lambdas.T).T + rng.normal(0.0, 0.001, size=(n_periods, n_assets))
    return returns.astype(np.float64), betas.astype(np.float64), n_periods, "normal"


@given(case=_fama_macbeth_case())
def test_fama_macbeth_matches_linearmodels(case: tuple[np.ndarray, np.ndarray, int, str]) -> None:
    """Also proves the hand `reference` equals `linearmodels.FamaMacBeth`.

    `check()` compares quantcore against this module's hand `reference` (the plain time-series
    convention); this test separately proves that hand reference equals `linearmodels.
    FamaMacBeth(cov_type='unadjusted')`, closing the same loop `test_newey_west_cov_anchored_to_
    statsmodels_ols_hac` closes for `newey_west_cov`.
    """
    returns, betas, n_periods, family = case
    kwargs = {"family": family, "length_bucket": length_bucket_for(n_periods), "seed": None}
    check(_FAMA_MACBETH_MEAN_RISK_PREMIA, returns, betas, **kwargs)
    check(_FAMA_MACBETH_STANDARD_ERRORS, returns, betas, **kwargs)

    n_assets, n_factors = betas.shape
    entities = [f"a{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([entities, range(n_periods)], names=["entity", "time"])
    exog_rows = [betas[i].tolist() for i in range(n_assets) for _t in range(n_periods)]
    dep_rows = [returns[t, i] for i in range(n_assets) for t in range(n_periods)]
    exog = pd.DataFrame(exog_rows, index=idx, columns=[f"f{j}" for j in range(n_factors)])
    dep = pd.Series(dep_rows, index=idx, name="y")
    lm_result = FamaMacBeth(dep, exog).fit(cov_type="unadjusted")

    # Direct provider calls (not the public wrapper, which dispatches through the backend
    # resolution map -- irrelevant to what this assertion checks: that the hand `reference`
    # construction `check()` compared quantcore against above equals linearmodels' own estimator).
    assert _FAMA_MACBETH_MEAN_RISK_PREMIA.reference is not None
    mean_risk_premia = _FAMA_MACBETH_MEAN_RISK_PREMIA.reference(returns, betas)
    assert np.allclose(mean_risk_premia, lm_result.params.to_numpy(), rtol=1e-6, atol=1e-8)
    assert _FAMA_MACBETH_STANDARD_ERRORS.reference is not None
    standard_errors = _FAMA_MACBETH_STANDARD_ERRORS.reference(returns, betas)
    assert np.allclose(standard_errors, lm_result.std_errors.to_numpy(), rtol=1e-6, atol=1e-8)


# --- ARIMA ----------------------------------------------------------------------------------

_ARIMA_ORDER_FOR_FAMILY: dict[str, tuple[int, int, int]] = {
    "white_noise": (1, 0, 0),
    "ar1": (1, 0, 0),
    "ar2": (2, 0, 0),
    "ma1": (0, 0, 1),
    "ma2": (0, 0, 2),
    "arma11": (1, 0, 1),
    "near_unit_root": (1, 0, 0),
}

ARIMA_FIT_MIN_LEN = 100
ARIMA_FIT_FAMILIES: tuple[str, ...] = ARMA_FAMILIES
ARIMA_FIT_BUCKETS: tuple[str, ...] = ("medium", "long")
_ARIMA_FIT_QUANTITIES = ("log_likelihood", "params")


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(tagged=arma_series(min_len=ARIMA_FIT_MIN_LEN, max_len=800))
def test_arima_fit_matches_statsmodels_loglike(tagged: TaggedArma) -> None:
    p, d, q = _ARIMA_ORDER_FOR_FAMILY[tagged.family]
    check(
        _ARIMA_FIT,
        tagged.values,
        p,
        d,
        q,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=_ARIMA_FIT_QUANTITIES,
    )


ARIMA_RESIDUALS_MIN_LEN = 20
ARIMA_RESIDUALS_FAMILIES: tuple[str, ...] = ARMA_FAMILIES
ARIMA_RESIDUALS_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(tagged=arma_series(min_len=ARIMA_RESIDUALS_MIN_LEN, max_len=1000))
def test_arima_residuals_matches_scipy_lfilter(tagged: TaggedArma) -> None:
    p, d, q = _ARIMA_ORDER_FOR_FAMILY[tagged.family]
    ar_coefs = tagged.true_ar if tagged.true_ar.shape[0] == p else np.zeros(p)
    ma_coefs = tagged.true_ma if tagged.true_ma.shape[0] == q else np.zeros(q)
    check(
        _ARIMA_RESIDUALS,
        tagged.values,
        ar_coefs,
        ma_coefs,
        d,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


ARIMA_FORECAST_MIN_LEN = 20
ARIMA_FORECAST_FAMILIES: tuple[str, ...] = ARMA_FAMILIES
ARIMA_FORECAST_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(tagged=arma_series(min_len=ARIMA_FORECAST_MIN_LEN, max_len=1000), h=st.integers(1, 10))
def test_arima_forecast_matches_statsmodels_conditional_mean(tagged: TaggedArma, h: int) -> None:
    p, d, q = _ARIMA_ORDER_FOR_FAMILY[tagged.family]
    ar_coefs = tagged.true_ar if tagged.true_ar.shape[0] == p else np.zeros(p)
    ma_coefs = tagged.true_ma if tagged.true_ma.shape[0] == q else np.zeros(q)
    check(
        _ARIMA_FORECAST,
        tagged.values,
        ar_coefs,
        ma_coefs,
        d,
        h,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


SELECT_ARIMA_ORDER_MIN_LEN = 60
SELECT_ARIMA_ORDER_FAMILIES: tuple[str, ...] = ("ar1", "ar2", "ma1", "arma11")
SELECT_ARIMA_ORDER_BUCKETS: tuple[str, ...] = ("medium",)


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(
    tagged=arma_series(
        min_len=SELECT_ARIMA_ORDER_MIN_LEN,
        max_len=200,
        families=("ar1", "ar2", "ma1", "arma11"),
    )
)
def test_select_arima_order_agreement_rate(tagged: TaggedArma) -> None:
    """Agreement rate is reported, not asserted.

    Order selection is inherently ambiguous between correlated models, so agreement rate is
    reported for visibility, not asserted as a pass/fail gate.
    """
    check(
        _SELECT_ARIMA_ORDER,
        tagged.values,
        3,
        0,
        3,
        "aic",
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


# --- VAR -------------------------------------------------------------------------------------

VAR_FIT_MIN_LEN = 30
VAR_FIT_FAMILIES: tuple[str, ...] = VAR_FAMILIES
VAR_FIT_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(tagged=var_system(min_len=VAR_FIT_MIN_LEN, max_len=1000), n_lags=st.integers(1, 3))
def test_var_fit_matches_statsmodels(tagged: TaggedVarSystem, n_lags: int) -> None:
    n_obs = tagged.data.shape[0]
    n_eff = n_obs - n_lags
    assume(n_eff > tagged.k * n_lags + 1)
    kwargs = {
        "family": tagged.family,
        "length_bucket": length_bucket_for(tagged.n),
        "seed": tagged.seed,
    }
    check(_VAR_FIT_COEF_MATRIX, tagged.data, n_lags, **kwargs)
    check(_VAR_FIT_SIGMA_U, tagged.data, n_lags, **kwargs)


VAR_FORECAST_MIN_LEN = 30
VAR_FORECAST_FAMILIES: tuple[str, ...] = VAR_FAMILIES
VAR_FORECAST_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(
    tagged=var_system(min_len=VAR_FORECAST_MIN_LEN, max_len=1000),
    n_lags=st.integers(1, 3),
    h=st.integers(1, 8),
)
def test_var_forecast_matches_statsmodels(tagged: TaggedVarSystem, n_lags: int, h: int) -> None:
    n_obs = tagged.data.shape[0]
    n_eff = n_obs - n_lags
    assume(n_eff > tagged.k * n_lags + 1)
    coef_matrix, _sigma_u = qc_var_fit(tagged.data, n_lags)
    check(
        _VAR_FORECAST,
        tagged.data,
        coef_matrix,
        n_lags,
        h,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


IMPULSE_RESPONSE_MIN_LEN = 30
IMPULSE_RESPONSE_FAMILIES: tuple[str, ...] = VAR_FAMILIES
IMPULSE_RESPONSE_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(
    tagged=var_system(min_len=IMPULSE_RESPONSE_MIN_LEN, max_len=1000),
    n_lags=st.integers(1, 3),
    n_periods=st.integers(2, 10),
)
def test_impulse_response_matches_statsmodels(
    tagged: TaggedVarSystem, n_lags: int, n_periods: int
) -> None:
    n_obs = tagged.data.shape[0]
    n_eff = n_obs - n_lags
    assume(n_eff > tagged.k * n_lags + 1)
    coef_matrix, sigma_u = qc_var_fit(tagged.data, n_lags)
    check(
        _IMPULSE_RESPONSE,
        coef_matrix,
        sigma_u,
        n_lags,
        n_periods,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


SELECT_VAR_LAG_ORDER_MIN_LEN = 40
SELECT_VAR_LAG_ORDER_FAMILIES: tuple[str, ...] = VAR_FAMILIES
SELECT_VAR_LAG_ORDER_BUCKETS: tuple[str, ...] = ("medium",)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(tagged=var_system(min_len=SELECT_VAR_LAG_ORDER_MIN_LEN, max_len=300))
def test_select_var_lag_order_agreement_rate(tagged: TaggedVarSystem) -> None:
    """Agreement rate reported, not asserted -- see `test_select_arima_order_agreement_rate`."""
    check(
        _SELECT_VAR_LAG_ORDER,
        tagged.data,
        5,
        "aic",
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


# --- Granger causality ----------------------------------------------------------------------

GRANGER_CAUSALITY_TEST_MIN_LEN = 20
GRANGER_CAUSALITY_TEST_FAMILIES: tuple[str, ...] = VAR_FAMILIES
GRANGER_CAUSALITY_TEST_BUCKETS: tuple[str, ...] = ("medium", "long")


@given(
    tagged=var_system(
        min_len=GRANGER_CAUSALITY_TEST_MIN_LEN,
        max_len=1000,
        families=("var2_stationary", "var3_stationary", "var2_near_unit_root"),
    ),
    n_lags=st.integers(1, 4),
)
def test_granger_causality_test_matches_statsmodels(tagged: TaggedVarSystem, n_lags: int) -> None:
    n_eff = tagged.n - n_lags
    assume(n_eff > 2 * n_lags + 1)
    y, x = tagged.data[:, 0], tagged.data[:, 1]
    check(
        _GRANGER_CAUSALITY_TEST,
        y,
        x,
        n_lags,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=("statistic", "p_value"),
    )


# --- OU half-life and spread z-score -------------------------------------------------------

OU_HALF_LIFE_MIN_LEN = 5
OU_HALF_LIFE_FAMILIES: tuple[str, ...] = ("ar1", "near_unit_root")
OU_HALF_LIFE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    tagged=arma_series(
        min_len=OU_HALF_LIFE_MIN_LEN, max_len=2000, families=("ar1", "near_unit_root")
    )
)
def test_ou_half_life_matches_hand_formula(tagged: TaggedArma) -> None:
    # A short, noisy draw's OLS slope can come out non-mean-reverting (a >= 0) by chance even for
    # a true mean-reverting process; both providers document that as an input contract violation
    # (a ValueError, not a numerical disagreement), so such a draw is skipped here rather than
    # recorded as a "reference raised" build failure.
    spread = tagged.values
    delta = spread[1:] - spread[:-1]
    level = spread[:-1]
    design = np.column_stack([np.ones_like(level), level])
    beta, _residuals, _rank, _singular = np.linalg.lstsq(design, delta, rcond=None)
    assume(beta[1] < 0.0 and 1.0 + beta[1] > 0.0)
    check(
        _OU_HALF_LIFE,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


SPREAD_ZSCORE_MIN_LEN = 2
SPREAD_ZSCORE_FAMILIES: tuple[str, ...] = ("ar1", "near_unit_root", "white_noise")
SPREAD_ZSCORE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    tagged=arma_series(
        min_len=SPREAD_ZSCORE_MIN_LEN,
        max_len=2000,
        families=("ar1", "near_unit_root", "white_noise"),
    ),
    window_fraction=st.floats(min_value=0.1, max_value=0.9),
)
def test_spread_zscore_matches_hand_formula(tagged: TaggedArma, window_fraction: float) -> None:
    window = max(2, min(tagged.n, int(tagged.n * window_fraction)))
    check(
        _SPREAD_ZSCORE,
        tagged.values,
        window,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )
