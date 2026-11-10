"""Every public function that returns a single number returns a finite one or raises.

The rule (`docs/UNDEFINED_RESULTS.md`): a public function annotated to return `float` or `int`
never returns NaN or infinity and never lets a `ZeroDivisionError` escape. When its number is
undefined or infinite it raises `quantcore.UndefinedResultError`, a `ValueError`.

The single-number functions are discovered from the `__all__` of every subpackage. Each is either
covered here, with a call that feeds it a 1-D returns series, or excluded with the reason it
cannot take one. A new public single-number function fails `test_every_single_number_function_is
_covered_or_excluded` until it is added to one of the two tables. Every covered function is run
on a battery of degenerate series: constants (including 0.002, whose computed standard deviation
is rounding residue rather than 0), all-positive, all-negative, all-zero, a single non-zero value
among zeros, two observations, a series whose compounded wealth overflows float64, and one so
small that the fourth power of its standard deviation underflows to 0.

The functions that return several numbers (tuples, arrays, result dataclasses) are run on a
constant-input battery: each input that the function reads as a series is held at 0.0, 0.01 and
3.7 in turn. Each case either raises `UndefinedResultError`, returns NaN exactly in its documented
members and finite numbers everywhere else, or is listed as defined, with the reason, and returns
finite numbers only. A new public multi-value function fails `test_every_multi_value_function_is_
in_the_constant_battery_or_excluded` until it is added to the battery or excluded with a reason.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import pkgutil
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

import quantcore
from quantcore import UndefinedResultError
from quantcore import filtering as qf
from quantcore import performance as qp
from quantcore import portfolio as qpf
from quantcore import regime as qg
from quantcore import risk as qr
from quantcore import statistics as qs
from quantcore import time_series as qt

FloatArray = npt.NDArray[np.float64]
Call = Callable[[FloatArray], float]

_N = 12


def _panel(x: FloatArray) -> FloatArray:
    """Two trials: the series and a copy shifted by one period."""
    return np.column_stack([x, np.roll(x, 1)])


# Qualified name -> (call on a 1-D returns series, fewest observations the function accepts).
# Each call looks the function up on its package at call time, so a monkeypatched package
# attribute is what runs.
COVERED: dict[str, tuple[Call, int]] = {
    "performance.sharpe_ratio": (lambda x: qp.sharpe_ratio(x), 1),
    "performance.sortino_ratio": (lambda x: qp.sortino_ratio(x), 1),
    "performance.calmar_ratio": (lambda x: qp.calmar_ratio(x), 1),
    "performance.information_ratio": (
        lambda x: qp.information_ratio(x, np.linspace(-0.001, 0.001, x.size)),
        1,
    ),
    "performance.hit_rate": (lambda x: qp.hit_rate(x, zero_policy="exclude"), 1),
    "performance.profit_factor": (lambda x: qp.profit_factor(x), 1),
    "performance.maximum_drawdown": (lambda x: qp.maximum_drawdown(x), 1),
    "performance.ulcer_index": (lambda x: qp.ulcer_index(x), 2),
    "performance.ulcer_performance_index": (lambda x: qp.ulcer_performance_index(x), 2),
    "performance.serenity_index": (lambda x: qp.serenity_index(x), 2),
    "performance.gain_to_pain_ratio": (lambda x: qp.gain_to_pain_ratio(x), 1),
    "performance.common_sense_ratio": (lambda x: qp.common_sense_ratio(x), 1),
    "performance.cpc_index": (lambda x: qp.cpc_index(x), 1),
    "performance.outlier_win_ratio": (lambda x: qp.outlier_win_ratio(x), 1),
    "performance.outlier_loss_ratio": (lambda x: qp.outlier_loss_ratio(x), 1),
    "performance.risk_of_ruin": (lambda x: qp.risk_of_ruin(x), 1),
    "performance.probabilistic_sharpe_ratio": (lambda x: qp.probabilistic_sharpe_ratio(x), 10),
    "performance.minimum_track_record_length": (
        lambda x: qp.minimum_track_record_length(x, 0.0, 0.05),
        10,
    ),
    "risk.value_at_risk": (lambda x: qr.value_at_risk(x, 0.95, method="cornish_fisher"), 2),
    "risk.conditional_value_at_risk": (
        lambda x: qr.conditional_value_at_risk(x, 0.95, method="cornish_fisher"),
        2,
    ),
    "risk.variance_backcast": (lambda x: qr.variance_backcast(x), 1),
    "statistics.effective_n_eigen": (lambda x: qs.effective_n_eigen(_panel(x)), 2),
    "statistics.newey_west_optimal_lags": (lambda x: qs.newey_west_optimal_lags(x), 1),
    "statistics.optimal_block_length": (lambda x: qs.optimal_block_length(x), 8),
    "statistics.ou_half_life": (lambda x: qs.ou_half_life(x), 5),
}

# The other methods of the two risk measures, run on the same battery.
EXTRA_CALLS: dict[str, tuple[Call, int]] = {
    "risk.value_at_risk[gaussian]": (lambda x: qr.value_at_risk(x, 0.95), 2),
    "risk.value_at_risk[historical]": (
        lambda x: qr.value_at_risk(x, 0.95, method="historical"),
        2,
    ),
    "risk.conditional_value_at_risk[gaussian]": (
        lambda x: qr.conditional_value_at_risk(x, 0.95),
        2,
    ),
    "risk.conditional_value_at_risk[historical]": (
        lambda x: qr.conditional_value_at_risk(x, 0.95, method="historical"),
        2,
    ),
    "performance.calmar_ratio[arithmetic]": (
        lambda x: qp.calmar_ratio(x, return_method="arithmetic"),
        1,
    ),
    "performance.hit_rate[loss]": (lambda x: qp.hit_rate(x), 1),
}

# Qualified name -> why it takes no returns series. Each is tested on its own degenerate inputs
# in its module's tests.
EXCLUDED: dict[str, str] = {
    "execution.margin_financing_cost": "scalar notional, rate and day count",
    "execution.short_borrow_cost": "scalar notional, rate and day count",
    "execution.square_root_market_impact": "scalar trade size, volume and volatility",
    "performance.expected_max_sharpe": "scalar number of trials and Sharpe variance",
    "pricing.black_scholes_call": "scalar option inputs",
    "pricing.black_scholes_put": "scalar option inputs",
    "pricing.bs_delta": "scalar option inputs",
    "pricing.bs_gamma": "scalar option inputs",
    "pricing.bs_theta": "scalar option inputs",
    "pricing.bs_vega": "scalar option inputs",
    "time_series.select_var_lag_order": (
        "a T x k system and a lag bound; its constant-column case is in the constant battery"
    ),
}

BATTERY: dict[str, FloatArray] = {
    "constant_0.002": np.full(_N, 0.002),
    "constant_negative": np.full(_N, -0.01),
    "all_zero": np.zeros(_N),
    "all_positive": np.linspace(0.001, 0.02, _N),
    "all_negative": -np.linspace(0.001, 0.02, _N),
    "single_positive_among_zeros": np.r_[np.zeros(_N - 1), 0.01],
    "single_negative_among_zeros": np.r_[np.zeros(_N - 1), -0.01],
    "two_observations": np.array([0.01, -0.02]),
    "two_equal_observations": np.array([0.01, 0.01]),
    "wealth_overflows": np.r_[np.full(6, 1e150), -0.5, 0.25, np.full(4, 1e150)],
    "moments_underflow": np.r_[1e-110, -2e-110, np.zeros(_N - 4), 3e-110, -1e-110],
    "mixed": np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.005, -0.004, 0.0, 0.01, -0.03, 0.02]),
}


def _single_number_functions() -> set[str]:
    """Qualified names of every public function annotated to return `float` or `int`."""
    found: set[str] = set()
    for info in pkgutil.iter_modules(quantcore.__path__):
        if not info.ispkg:
            continue
        package = importlib.import_module(f"quantcore.{info.name}")
        for name in getattr(package, "__all__", ()):
            obj = getattr(package, name)
            if not inspect.isfunction(obj):
                continue
            returns = inspect.signature(obj).return_annotation
            if returns in ("float", "int", float, int):
                found.add(f"{info.name}.{name}")
    return found


def problem(call: Call, x: FloatArray) -> str | None:
    """None if `call(x)` returns a finite number or raises UndefinedResultError; else why not."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            value = call(x)
    except UndefinedResultError:
        return None
    except Exception as exc:
        return f"raised {type(exc).__name__}: {exc}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"returned a {type(value).__name__}, not a float or int"
    if not math.isfinite(value):
        return f"returned {value!r}"
    return None


def test_every_single_number_function_is_covered_or_excluded() -> None:
    assert not set(COVERED) & set(EXCLUDED)
    assert set(COVERED) | set(EXCLUDED) == _single_number_functions()


def test_discovery_sees_functions_returning_int() -> None:
    assert "statistics.newey_west_optimal_lags" in _single_number_functions()


_CASES = [
    pytest.param(call, x, id=f"{name}-{label}")
    for name, (call, min_obs) in {**COVERED, **EXTRA_CALLS}.items()
    for label, x in BATTERY.items()
    if x.size >= min_obs
]


@pytest.mark.parametrize(("call", "x"), _CASES)
def test_result_is_finite_or_undefined(call: Call, x: FloatArray) -> None:
    assert problem(call, x) is None


def test_every_covered_function_runs_on_at_least_eight_battery_inputs() -> None:
    for name, (_call, min_obs) in COVERED.items():
        assert sum(x.size >= min_obs for x in BATTERY.values()) >= 8, name


def test_checker_reports_an_infinite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    call, _ = COVERED["performance.sharpe_ratio"]
    x = BATTERY["mixed"]
    assert problem(call, x) is None
    monkeypatch.setattr(qp, "sharpe_ratio", lambda *args, **kwargs: float("inf"))
    assert problem(call, x) == "returned inf"


def test_checker_reports_nan_zero_division_and_a_plain_value_error() -> None:
    x = BATTERY["mixed"]

    def zero_division(_x: FloatArray) -> float:
        return 1.0 / 0.0

    def plain_value_error(_x: FloatArray) -> float:
        raise ValueError("bad input")

    assert problem(lambda _x: float("nan"), x) == "returned nan"
    assert problem(zero_division, x) == "raised ZeroDivisionError: float division by zero"
    assert problem(plain_value_error, x) == "raised ValueError: bad input"


# --- Constant-input battery over the multi-value functions -------------------------------------

_T = 60
_CONSTANTS = (0.0, 0.01, 3.7)
_WALK = np.cumsum(np.random.default_rng(1).normal(size=_T))
_NOISE = np.random.default_rng(2).normal(scale=0.01, size=_T)
_FACTORS = np.random.default_rng(3).normal(scale=0.01, size=(_T, 2))
_DESIGN = np.column_stack([np.ones(_T), np.linspace(-1.0, 1.0, _T)])
_BETAS = np.random.default_rng(4).normal(size=(10, 2))
_TRANSITION = np.array([[0.9, 0.1], [0.2, 0.8]])
_HMM_MEANS = np.array([0.0, 0.01])
_HMM_VARIANCES = np.array([1e-4, 4e-4])
_HMM_START = np.array([0.5, 0.5])
_VAR_COEFS = np.array([[0.5, 0.1, 0.0], [0.0, 0.3, 0.0]])

Outcome = Literal["raises", "nan", "defined"]


@dataclasses.dataclass(frozen=True)
class ConstantCase:
    """How one multi-value function must behave when its series input is a constant `v`.

    `call(v)` runs the function on the constant input. `outcome` is "raises" (it must raise
    `UndefinedResultError`), "nan" (the members named in `nan_members` must be all NaN and every
    other numeric member finite) or "defined" (every numeric member finite; `reason` says why
    the result is defined). `check`, if given, asserts more about a returned result. `values`
    are the constants the case runs on.
    """

    call: Callable[[float], object]
    outcome: Outcome
    nan_members: tuple[str, ...] = ()
    reason: str = ""
    check: Callable[[object, float], None] | None = None
    values: tuple[float, ...] = _CONSTANTS


def _hmm_fit_check(result: object, v: float) -> None:
    _, means, variances, _ = result  # type: ignore[misc]
    np.testing.assert_allclose(means, np.full(2, v), rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(variances, np.full(2, 1e-8))


def _bootstrap_check(result: object, v: float) -> None:
    interval = result
    assert interval.ci_low == interval.ci_high == interval.estimate  # type: ignore[attr-defined]
    assert interval.estimate == pytest.approx(v, abs=1e-12)  # type: ignore[attr-defined]


def _with_column(v: float, column: int) -> FloatArray:
    system = np.column_stack([_WALK, _NOISE])
    system[:, column] = v
    return system


_FL_NAN = ("r_squared", "t_stats", "p_values")
_BOOT_REASON = (
    "with an explicit block_length every resample of a constant is that constant, so the "
    "interval is (c, c)"
)
_GIVEN_PARAMS = "the recursion runs from given parameters and a positive start"

# Name (qualified, an optional "[variant]" suffix) -> the case.
CONSTANT_CASES: dict[str, ConstantCase] = {
    "statistics.adf_test": ConstantCase(lambda v: qs.adf_test(np.full(_T, v)), "raises"),
    "statistics.arch_lm_test": ConstantCase(lambda v: qs.arch_lm_test(np.full(_T, v)), "raises"),
    "statistics.cscv_pbo": ConstantCase(
        lambda v: qs.cscv_pbo(np.full((64, 4), v), n_splits=4), "raises"
    ),
    "statistics.engle_granger_test[y]": ConstantCase(
        lambda v: qs.engle_granger_test(np.full(_T, v), _WALK), "raises"
    ),
    "statistics.engle_granger_test[x]": ConstantCase(
        lambda v: qs.engle_granger_test(_WALK, np.full(_T, v)), "raises"
    ),
    "statistics.factor_loadings[returns]": ConstantCase(
        lambda v: qs.factor_loadings(np.full(_T, v), _FACTORS), "nan", nan_members=_FL_NAN
    ),
    "statistics.factor_loadings[factor column]": ConstantCase(
        lambda v: qs.factor_loadings(_NOISE, np.column_stack([_FACTORS[:, 0], np.full(_T, v)])),
        "raises",
    ),
    "statistics.fama_macbeth_regression[returns]": ConstantCase(
        lambda v: qs.fama_macbeth_regression(np.full((_T, 10), v), _BETAS), "nan", ("1",)
    ),
    "statistics.fama_macbeth_regression[returns, newey-west]": ConstantCase(
        lambda v: qs.fama_macbeth_regression(np.full((_T, 10), v), _BETAS, n_lags_nw=2),
        "nan",
        ("1",),
    ),
    "statistics.fama_macbeth_regression[zero beta column]": ConstantCase(
        lambda v: qs.fama_macbeth_regression(
            np.full((_T, 10), v) + _NOISE[:, None], np.column_stack([_BETAS[:, 0], np.zeros(10)])
        ),
        "raises",
    ),
    "statistics.hac_t_test": ConstantCase(lambda v: qs.hac_t_test(np.full(_T, v)), "raises"),
    "statistics.jarque_bera_test": ConstantCase(
        lambda v: qs.jarque_bera_test(np.full(_T, v)), "raises"
    ),
    "statistics.kpss_test": ConstantCase(lambda v: qs.kpss_test(np.full(_T, v)), "raises"),
    "statistics.newey_west_cov": ConstantCase(
        lambda v: qs.newey_west_cov(_DESIGN, np.full(_T, v), 2),
        "defined",
        reason="the residuals are given; their covariance with the regressors is a sum of "
        "finite products",
    ),
    "statistics.ols": ConstantCase(lambda v: qs.ols(np.full(_T, v), _DESIGN), "nan", ("2",)),
    "statistics.spa_test": ConstantCase(
        lambda v: qs.spa_test(
            np.full(_T, v),
            np.full((_T, 2), v),
            rng=np.random.default_rng(0),
            block_size=4,
            reps=20,
            studentize=True,
        ),
        "raises",
    ),
    "statistics.spa_test[constant offset]": ConstantCase(
        lambda v: qs.spa_test(
            np.full(_T, v),
            np.column_stack([_WALK, np.full(_T, v + 0.5)]),
            rng=np.random.default_rng(0),
            block_size=4,
            reps=20,
            studentize=True,
        ),
        "raises",
    ),
    "statistics.spread_zscore": ConstantCase(
        lambda v: qs.spread_zscore(np.full(_T, v), 5), "nan", ("value",)
    ),
    "statistics.stationary_bootstrap_ci[explicit block length]": ConstantCase(
        lambda v: qs.stationary_bootstrap_ci(
            np.full(_T, v), np.mean, rng=np.random.default_rng(0), n_boot=50, block_length=3.0
        ),
        "defined",
        reason=_BOOT_REASON,
        check=_bootstrap_check,
    ),
    "statistics.stationary_bootstrap_ci[bca]": ConstantCase(
        lambda v: qs.stationary_bootstrap_ci(
            np.full(_T, v),
            np.mean,
            rng=np.random.default_rng(0),
            n_boot=50,
            block_length=3.0,
            method="bca",
        ),
        "defined",
        reason=_BOOT_REASON,
        check=_bootstrap_check,
    ),
    "statistics.stationary_bootstrap_ci[automatic block length]": ConstantCase(
        lambda v: qs.stationary_bootstrap_ci(
            np.full(_T, v), np.mean, rng=np.random.default_rng(0), n_boot=50
        ),
        "raises",
    ),
    "time_series.arima_fit": ConstantCase(
        lambda v: qt.arima_fit(np.full(_T, v), 1, 0, 1), "raises"
    ),
    "time_series.arima_fit[d=1]": ConstantCase(
        lambda v: qt.arima_fit(np.full(_T, v), 1, 1, 0), "raises"
    ),
    "time_series.arima_forecast": ConstantCase(
        lambda v: qt.arima_forecast(np.full(_T, v), np.array([0.5]), np.array([0.2]), 0, 3),
        "defined",
        reason="the coefficients are given; the forecast is a finite linear recursion",
    ),
    "time_series.arima_residuals": ConstantCase(
        lambda v: qt.arima_residuals(np.full(_T, v), np.array([0.5]), np.array([0.2]), 0),
        "defined",
        reason="the coefficients are given; the residuals are a finite linear filter",
    ),
    "time_series.granger_causality_test[y]": ConstantCase(
        lambda v: qt.granger_causality_test(np.full(_T, v), _WALK, 2), "raises"
    ),
    "time_series.granger_causality_test[x]": ConstantCase(
        lambda v: qt.granger_causality_test(_WALK, np.full(_T, v), 2), "raises"
    ),
    "time_series.ljung_box_test": ConstantCase(
        lambda v: qt.ljung_box_test(np.full(_T, v), 5), "raises"
    ),
    "time_series.select_arima_order": ConstantCase(
        lambda v: qt.select_arima_order(np.full(_T, v), 1, 1, 1), "raises"
    ),
    "time_series.select_var_lag_order": ConstantCase(
        lambda v: qt.select_var_lag_order(_with_column(v, 1), 3), "raises"
    ),
    "time_series.var_fit": ConstantCase(lambda v: qt.var_fit(_with_column(v, 0), 1), "raises"),
    "time_series.var_forecast": ConstantCase(
        lambda v: qt.var_forecast(np.full((_T, 2), v), _VAR_COEFS, 1, 3),
        "defined",
        reason="the coefficients are given; the forecast is a finite linear recursion",
    ),
    "risk.egarch_11_variance": ConstantCase(
        lambda v: qr.egarch_11_variance(np.full(_T, v), -0.1, 0.1, -0.05, 0.95, 1e-4),
        "defined",
        reason=_GIVEN_PARAMS,
    ),
    "risk.egarch_11_variance[backcast start]": ConstantCase(
        lambda v: qr.egarch_11_variance(np.full(_T, v), -0.1, 0.1, -0.05, 0.95),
        "defined",
        reason="the backcast of a non-zero constant is positive",
        values=(0.01, 3.7),
    ),
    "risk.egarch_11_variance[backcast of zeros]": ConstantCase(
        lambda v: qr.egarch_11_variance(np.full(_T, v), -0.1, 0.1, -0.05, 0.95),
        "raises",
        values=(0.0,),
    ),
    "risk.egarch_fit": ConstantCase(lambda v: qr.egarch_fit(np.full(_T, v)), "raises"),
    "risk.ewma_variance": ConstantCase(
        lambda v: qr.ewma_variance(np.full(_T, v), 0.94),
        "defined",
        reason="an exponentially weighted mean of squares is finite and non-negative",
    ),
    "risk.fit_garch_11": ConstantCase(lambda v: qr.fit_garch_11(np.full(_T, v)), "raises"),
    "risk.garch_11_variance": ConstantCase(
        lambda v: qr.garch_11_variance(np.full(_T, v), 1e-6, 0.1, 0.8),
        "defined",
        reason=_GIVEN_PARAMS,
    ),
    "risk.gjr_garch_11_variance": ConstantCase(
        lambda v: qr.gjr_garch_11_variance(np.full(_T, v), 1e-6, 0.05, 0.1, 0.8),
        "defined",
        reason=_GIVEN_PARAMS,
    ),
    "risk.gjr_garch_fit": ConstantCase(lambda v: qr.gjr_garch_fit(np.full(_T, v)), "raises"),
    "portfolio.ledoit_wolf_shrinkage": ConstantCase(
        lambda v: qpf.ledoit_wolf_shrinkage(np.full((_T, 3), v)),
        "defined",
        reason="the sample covariance of constant columns is 0, or rounding residue near 0, and "
        "so is its shrunk estimate",
    ),
    "regime.hmm_decode": ConstantCase(
        lambda v: qg.hmm_decode(
            np.full(_T, v), _TRANSITION, _HMM_MEANS, _HMM_VARIANCES, _HMM_START
        ),
        "defined",
        reason="the model parameters are given; every path has a finite log-likelihood",
    ),
    "regime.hmm_fit": ConstantCase(
        lambda v: qg.hmm_fit(np.full(_T, v), 2),
        "defined",
        reason="every state's mean is the constant and its variance sits at the 1e-8 floor: "
        "the states are not distinguishable",
        check=_hmm_fit_check,
    ),
    "regime.hmm_predict_proba": ConstantCase(
        lambda v: qg.hmm_predict_proba(
            np.full(_T, v), _TRANSITION, _HMM_MEANS, _HMM_VARIANCES, _HMM_START
        ),
        "defined",
        reason="the model parameters are given; the filtered probabilities are finite",
    ),
    "filtering.dynamic_hedge_ratio[x]": ConstantCase(
        lambda v: qf.dynamic_hedge_ratio(_WALK, np.full(_T, v)),
        "defined",
        reason="the Kalman recursion with a positive observation variance stays finite; with a "
        "constant x the ratio tracks y / x",
    ),
    "filtering.dynamic_hedge_ratio[y]": ConstantCase(
        lambda v: qf.dynamic_hedge_ratio(np.full(_T, v), _WALK),
        "defined",
        reason="the Kalman recursion with a positive observation variance stays finite",
    ),
    "performance.deflated_sharpe_ratio": ConstantCase(
        lambda v: qp.deflated_sharpe_ratio(np.full(_T, v), [0.1, 0.2, 0.3], n_effective=3.0),
        "raises",
    ),
    "performance.sharpe_se_mertens": ConstantCase(
        lambda v: qp.sharpe_se_mertens(np.full(_T, v), periods_per_year=252), "raises"
    ),
}

# Qualified name -> why it has no series input to hold constant.
CONSTANT_EXCLUDED: dict[str, str] = {
    "time_series.impulse_response": "fitted VAR coefficients and a residual covariance",
    "portfolio.black_litterman": "a covariance matrix, weights and views",
    "portfolio.risk_parity_weights": "a covariance matrix",
    "portfolio.unconstrained_mean_variance_weights": "expected returns and a covariance matrix",
    "performance.component_var": "weights and a covariance matrix",
    "pricing.option_roll_schedule": "dates",
    "pricing.protective_put_sizing": "scalar option and portfolio inputs",
    "core.simulate_cir_paths": "scalar model parameters",
    "core.simulate_garch_t_price_paths": "scalar model parameters",
    "core.simulate_gbm_paths": "scalar model parameters",
    "core.simulate_gbm_price_paths": "scalar model parameters",
    "core.simulate_heston_price_paths": "scalar model parameters",
    "core.simulate_jump_diffusion_price_paths": "scalar model parameters",
    "core.simulate_ou_paths": "scalar model parameters",
    "core.cir_fit": (
        "a positive constant raises CIRNoMeanReversionError, a RuntimeError other repos catch by "
        "name, and 0 is rejected as not positive; test_cir_fit_constant_series_raises pins both"
    ),
}


def _multi_value_functions() -> set[str]:
    """Qualified names of every public function not annotated to return `float` or `int`."""
    found: set[str] = set()
    for info in pkgutil.iter_modules(quantcore.__path__):
        if not info.ispkg:
            continue
        package = importlib.import_module(f"quantcore.{info.name}")
        for name in getattr(package, "__all__", ()):
            obj = getattr(package, name)
            if not inspect.isfunction(obj):
                continue
            returns = inspect.signature(obj).return_annotation
            if returns not in ("float", "int", float, int):
                found.add(f"{info.name}.{name}")
    return found


def _numeric_members(result: object) -> dict[str, FloatArray]:
    """The numeric members of a result, by field name, NamedTuple field or tuple index."""
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        items = [(f.name, getattr(result, f.name)) for f in dataclasses.fields(result)]
    elif isinstance(result, tuple):
        names = getattr(result, "_fields", None) or [str(i) for i in range(len(result))]
        items = list(zip(names, result, strict=True))
    else:
        items = [("value", result)]
    members: dict[str, FloatArray] = {}
    for name, value in items:
        array = np.asarray(value)
        if array.dtype.kind in "fiu":
            members[name] = array.astype(np.float64)
    return members


def constant_problem(case: ConstantCase, v: float) -> str | None:
    """None if `case.call(v)` behaves as `case` says; else what went wrong."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            result = case.call(v)
    except UndefinedResultError as exc:
        return None if case.outcome == "raises" else f"raised UndefinedResultError: {exc}"
    except Exception as exc:
        return f"raised {type(exc).__name__}: {exc}"
    if case.outcome == "raises":
        return f"returned {result!r} instead of raising UndefinedResultError"
    members = _numeric_members(result)
    missing = set(case.nan_members) - set(members)
    if missing:
        return f"has no member(s) {sorted(missing)}"
    for name, array in members.items():
        if name in case.nan_members:
            if not np.all(np.isnan(array)):
                return f"member {name} is {array!r}, documented as NaN"
        elif not np.all(np.isfinite(array)):
            return f"member {name} is {array!r}, not finite"
    if case.check is not None:
        try:
            case.check(result, v)
        except AssertionError as exc:
            return f"check failed: {exc}"
    return None


def test_every_multi_value_function_is_in_the_constant_battery_or_excluded() -> None:
    # The battery may also hold a single-number function whose constant case needs a system
    # input (select_var_lag_order).
    battery = {name.split("[")[0] for name in CONSTANT_CASES}
    assert not battery & set(CONSTANT_EXCLUDED)
    listed = (battery | set(CONSTANT_EXCLUDED)) - _single_number_functions()
    assert listed == _multi_value_functions()


@pytest.mark.parametrize("v", [0.01, 3.7])
def test_cir_fit_constant_series_raises(v: float) -> None:
    from quantcore.core import cir_fit
    from quantcore.core.stochastic import CIRNoMeanReversionError

    with pytest.raises(CIRNoMeanReversionError):
        cir_fit(np.full(_T, v), 1.0 / 252.0)


def test_cir_fit_all_zero_series_is_rejected_as_not_positive() -> None:
    from quantcore.core import cir_fit

    with pytest.raises(ValueError, match="strictly positive"):
        cir_fit(np.zeros(_T), 1.0 / 252.0)


def test_every_constant_case_is_consistent() -> None:
    battery = {name.split("[")[0] for name in CONSTANT_CASES}
    for name in battery:
        values = {
            v
            for key, case in CONSTANT_CASES.items()
            if key.split("[")[0] == name
            for v in case.values
        }
        assert values >= set(_CONSTANTS), name
    for name, case in CONSTANT_CASES.items():
        assert bool(case.nan_members) == (case.outcome == "nan"), name
        assert bool(case.reason) == (case.outcome == "defined"), name


_CONSTANT_CASES = [
    pytest.param(case, v, id=f"{name}-{v}")
    for name, case in CONSTANT_CASES.items()
    for v in case.values
]


@pytest.mark.parametrize(("case", "v"), _CONSTANT_CASES)
def test_constant_input_raises_or_matches_its_documented_result(
    case: ConstantCase, v: float
) -> None:
    assert constant_problem(case, v) is None


def test_constant_checker_reports_a_planted_finite_wrong_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CONSTANT_CASES["statistics.engle_granger_test[y]"]
    assert constant_problem(case, 0.01) is None
    monkeypatch.setattr(qs, "engle_granger_test", lambda y, x: (-1.2, 0.797, -3.4))
    reported = constant_problem(case, 0.01)
    assert reported is not None and "instead of raising" in reported


def test_constant_checker_reports_a_finite_member_documented_as_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CONSTANT_CASES["statistics.ols"]
    assert constant_problem(case, 0.01) is None
    real_ols = qs.ols

    def finite_r_squared(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        coefficients, residuals, _ = real_ols(y, x)
        return coefficients, residuals, -146.7

    monkeypatch.setattr(qs, "ols", finite_r_squared)
    assert constant_problem(case, 0.01) == "member 2 is array(-146.7), documented as NaN"


def test_constant_checker_reports_a_non_finite_defined_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CONSTANT_CASES["portfolio.ledoit_wolf_shrinkage"]
    assert constant_problem(case, 3.7) is None
    monkeypatch.setattr(qpf, "ledoit_wolf_shrinkage", lambda r: np.full((3, 3), np.nan))
    reported = constant_problem(case, 3.7)
    assert reported is not None and "not finite" in reported


def test_constant_checker_reports_a_plain_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CONSTANT_CASES["statistics.adf_test"]

    def plain_value_error(*_args: object, **_kwargs: object) -> tuple[float, float]:
        raise ValueError("constant series")

    monkeypatch.setattr(qs, "adf_test", plain_value_error)
    assert constant_problem(case, 0.0) == "raised ValueError: constant series"
