"""Calls every public quantcore function on fixed inputs and prints a digest of each result.

`tests/unit/test_cross_process_determinism.py` runs this script in separate processes, with a cold
and a warm Numba cache and with different BLAS thread counts. For a fixed BLAS thread
configuration every digest must agree. Across thread counts every digest must agree except those
of `BLAS_THREAD_SENSITIVE`, whose values must agree within the test's tolerance. A call that
raises is recorded by its exception type and message, which must agree too.

Usage: `python tests/determinism_battery.py` prints a JSON object with `digests` (name -> sha256)
and `values` (name -> the result's floats as `float.hex()` strings, for the names in
`BLAS_THREAD_SENSITIVE`). The variable `QC23_ONLY` (comma-separated names) runs only those calls.
The name `planted_unseeded` is not run unless it is named there: it draws from an unseeded
generator, for the test's negative control.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import struct
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

import quantcore.core as core
import quantcore.execution as execution
import quantcore.filtering as filtering
import quantcore.performance as performance
import quantcore.portfolio as portfolio
import quantcore.pricing as pricing
import quantcore.regime as regime
import quantcore.risk as risk
import quantcore.statistics as statistics
import quantcore.time_series as time_series

PLANTED = "planted_unseeded"

# The fits `arch` computes run scipy's SLSQP, whose arithmetic goes through BLAS. On OpenBLAS a
# single thread rounds differently from several, and the optimizer carries that last-bit
# difference into the fitted parameters, so these results depend on the BLAS thread count.
BLAS_THREAD_SENSITIVE = frozenset({"risk.egarch_fit", "risk.fit_garch_11", "risk.gjr_garch_fit"})


def _encode(value: Any, out: list[bytes]) -> None:
    """Append a canonical byte encoding of `value`: type tags, shapes and exact float bits."""
    if isinstance(value, np.ndarray):
        out.append(f"ndarray{value.dtype.str}{value.shape}".encode())
        out.append(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bool, np.bool_)):
        out.append(b"bool1" if value else b"bool0")
    elif isinstance(value, (int, np.integer)):
        out.append(f"int{int(value)}".encode())
    elif isinstance(value, (float, np.floating)):
        out.append(b"float" + struct.pack("<d", float(value)))
    elif isinstance(value, str):
        out.append(b"str" + value.encode())
    elif isinstance(value, dt.date):
        out.append(b"date" + value.isoformat().encode())
    elif value is None:
        out.append(b"none")
    elif isinstance(value, (tuple, list)):
        out.append(f"seq{len(value)}".encode())
        for item in value:
            _encode(item, out)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        out.append(type(value).__name__.encode())
        for field in dataclasses.fields(value):
            out.append(field.name.encode())
            _encode(getattr(value, field.name), out)
    else:
        raise TypeError(f"no encoding for {type(value).__name__}")


def digest(value: Any) -> str:
    out: list[bytes] = []
    _encode(value, out)
    return hashlib.sha256(b"\x00".join(out)).hexdigest()


def _inputs() -> dict[str, Any]:
    rng = np.random.default_rng(20261110)
    n = 500
    returns = 0.0004 + 0.01 * rng.standard_normal(n)
    benchmark = 0.0003 + 0.009 * rng.standard_normal(n)
    panel = 0.0002 + 0.01 * rng.standard_normal((n, 60))
    # GARCH(1,1) returns, so the fits have something to find.
    z = rng.standard_normal(1500)
    garch = np.empty_like(z)
    variance = 1e-4
    for t, shock in enumerate(z):
        garch[t] = np.sqrt(variance) * shock
        variance = 2e-6 + 0.08 * garch[t] ** 2 + 0.9 * variance
    x_level = 100.0 + np.cumsum(rng.standard_normal(n))
    y_level = 1.5 * x_level + rng.standard_normal(n)
    ou = np.empty(n)
    ou[0] = 1.0
    for t in range(1, n):
        ou[t] = 0.9 * ou[t - 1] + 0.1 * rng.standard_normal()
    arma = np.empty(n)
    arma[0] = 0.0
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        arma[t] = 0.5 * arma[t - 1] + shocks[t] + 0.3 * shocks[t - 1]
    system = np.empty((n, 3))
    system[0] = 0.0
    for t in range(1, n):
        system[t] = 0.4 * system[t - 1] + rng.standard_normal(3)
    regressors = np.column_stack([np.ones(2000), rng.standard_normal((2000, 49))])
    response = regressors @ rng.standard_normal(50) + rng.standard_normal(2000)
    cov = np.cov(panel, rowvar=False)
    return {
        "returns": returns,
        "benchmark": benchmark,
        "panel": panel,
        "cov": cov,
        "garch": garch,
        "x_level": x_level,
        "y_level": y_level,
        "ou": ou,
        "arma": arma,
        "system": system,
        "regressors": regressors,
        "response": response,
        "two_state": np.concatenate([0.01 * z[:250], 0.03 * z[250:500]]),
    }


def _calls(d: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    r, g = d["returns"], d["garch"]
    cov5 = d["cov"][:5, :5]
    weights60 = np.full(60, 1.0 / 60.0)
    cir = core.simulate_cir_paths(0.03, 2.0, 0.04, 0.1, 1.0, 1, 250, seed=11)[0]
    hmm = regime.hmm_fit(d["two_state"], 2)
    trans, means, variances, initial = hmm
    arima = time_series.arima_fit(d["arma"], 1, 0, 1)
    var_coef, var_sigma = time_series.var_fit(d["system"], 2)
    _, ols_resid, _ = statistics.ols(d["response"], d["regressors"])
    return {
        "core.cir_fit": lambda: core.cir_fit(cir, 1.0 / 250.0),
        "core.simulate_cir_paths": lambda: core.simulate_cir_paths(
            0.03, 2.0, 0.04, 0.1, 1.0, 50, 100, seed=7
        ),
        "core.simulate_garch_t_price_paths": lambda: core.simulate_garch_t_price_paths(
            np.random.default_rng(1), 20, 100, 1.0 / 252.0
        ),
        "core.simulate_gbm_paths": lambda: core.simulate_gbm_paths(
            100.0, 0.03, 0.2, 1.0, 100, 50, seed=7
        ),
        "core.simulate_gbm_price_paths": lambda: core.simulate_gbm_price_paths(
            np.random.default_rng(2), 20, 100, 1.0 / 252.0
        ),
        "core.simulate_heston_price_paths": lambda: core.simulate_heston_price_paths(
            np.random.default_rng(3), 20, 100, 1.0 / 252.0
        ),
        "core.simulate_jump_diffusion_price_paths": (
            lambda: core.simulate_jump_diffusion_price_paths(
                np.random.default_rng(4), 20, 100, 1.0 / 252.0
            )
        ),
        "core.simulate_ou_paths": lambda: core.simulate_ou_paths(
            2.0, 0.0, 0.3, 0.1, 1.0, 50, 100, seed=7
        ),
        "execution.margin_financing_cost": lambda: execution.margin_financing_cost(1e6, 0.05, 30),
        "execution.short_borrow_cost": lambda: execution.short_borrow_cost(1e6, 0.02, 30),
        "execution.square_root_market_impact": lambda: execution.square_root_market_impact(
            1e5, 1e7, 0.02
        ),
        "filtering.dynamic_hedge_ratio": lambda: filtering.dynamic_hedge_ratio(
            d["y_level"], d["x_level"]
        ),
        "performance.calmar_ratio": lambda: performance.calmar_ratio(r),
        "performance.common_sense_ratio": lambda: performance.common_sense_ratio(r),
        "performance.component_var": lambda: performance.component_var(weights60, d["cov"], 0.95),
        "performance.cpc_index": lambda: performance.cpc_index(r),
        "performance.deflated_sharpe_ratio": lambda: performance.deflated_sharpe_ratio(
            r, np.linspace(-0.02, 0.06, 10), n_effective=5.0
        ),
        "performance.expected_max_sharpe": lambda: performance.expected_max_sharpe(10.0, 0.01),
        "performance.gain_to_pain_ratio": lambda: performance.gain_to_pain_ratio(r),
        "performance.hit_rate": lambda: performance.hit_rate(r),
        "performance.information_ratio": lambda: performance.information_ratio(r, d["benchmark"]),
        "performance.maximum_drawdown": lambda: performance.maximum_drawdown(r),
        "performance.minimum_track_record_length": (
            lambda: performance.minimum_track_record_length(r, 0.0, 0.05)
        ),
        "performance.outlier_loss_ratio": lambda: performance.outlier_loss_ratio(r),
        "performance.outlier_win_ratio": lambda: performance.outlier_win_ratio(r),
        "performance.probabilistic_sharpe_ratio": lambda: performance.probabilistic_sharpe_ratio(r),
        "performance.profit_factor": lambda: performance.profit_factor(r),
        "performance.risk_of_ruin": lambda: performance.risk_of_ruin(r),
        "performance.serenity_index": lambda: performance.serenity_index(r),
        "performance.sharpe_ratio": lambda: performance.sharpe_ratio(r),
        "performance.sharpe_se_mertens": lambda: performance.sharpe_se_mertens(
            r, periods_per_year=252
        ),
        "performance.sortino_ratio": lambda: performance.sortino_ratio(r),
        "performance.ulcer_index": lambda: performance.ulcer_index(r),
        "performance.ulcer_performance_index": lambda: performance.ulcer_performance_index(r),
        "portfolio.black_litterman": lambda: portfolio.black_litterman(
            cov5,
            np.full(5, 0.2),
            np.array([[1.0, -1.0, 0.0, 0.0, 0.0]]),
            np.array([0.02]),
            np.array([[0.001]]),
            2.5,
        ),
        "portfolio.ledoit_wolf_shrinkage": lambda: portfolio.ledoit_wolf_shrinkage(d["panel"]),
        "portfolio.risk_parity_weights": lambda: portfolio.risk_parity_weights(d["cov"][:20, :20]),
        "portfolio.unconstrained_mean_variance_weights": (
            lambda: portfolio.unconstrained_mean_variance_weights(
                d["panel"].mean(axis=0), d["cov"], 3.0
            )
        ),
        "pricing.black_scholes_call": lambda: pricing.black_scholes_call(100, 105, 0.03, 0.2, 0.5),
        "pricing.black_scholes_put": lambda: pricing.black_scholes_put(100, 105, 0.03, 0.2, 0.5),
        "pricing.bs_delta": lambda: pricing.bs_delta(100, 105, 0.03, 0.2, 0.5, "call"),
        "pricing.bs_gamma": lambda: pricing.bs_gamma(100, 105, 0.03, 0.2, 0.5),
        "pricing.bs_theta": lambda: pricing.bs_theta(100, 105, 0.03, 0.2, 0.5, "put"),
        "pricing.bs_vega": lambda: pricing.bs_vega(100, 105, 0.03, 0.2, 0.5),
        "pricing.option_roll_schedule": lambda: pricing.option_roll_schedule(
            dt.date(2026, 1, 5),
            30,
            [dt.date(2026, 1, 16), dt.date(2026, 2, 20), dt.date(2026, 3, 20)],
        ),
        "pricing.protective_put_sizing": lambda: pricing.protective_put_sizing(
            1e6, 100.0, 95.0, 0.2, 0.25, 0.03, 0.9, 0.02
        ),
        "regime.hmm_decode": lambda: regime.hmm_decode(
            d["two_state"], trans, means, variances, initial
        ),
        "regime.hmm_fit": lambda: regime.hmm_fit(d["two_state"], 2),
        "regime.hmm_predict_proba": lambda: regime.hmm_predict_proba(
            d["two_state"], trans, means, variances, initial
        ),
        "risk.conditional_value_at_risk": lambda: [
            risk.conditional_value_at_risk(r, 0.95, method=m)
            for m in ("gaussian", "historical", "cornish_fisher")
        ],
        "risk.egarch_11_variance": lambda: risk.egarch_11_variance(g, -0.2, 0.1, -0.05, 0.98),
        "risk.egarch_fit": lambda: risk.egarch_fit(g),
        "risk.ewma_variance": lambda: risk.ewma_variance(g, 0.94),
        "risk.fit_garch_11": lambda: risk.fit_garch_11(g),
        "risk.garch_11_variance": lambda: risk.garch_11_variance(g, 2e-6, 0.08, 0.9),
        "risk.gjr_garch_11_variance": lambda: risk.gjr_garch_11_variance(g, 2e-6, 0.05, 0.05, 0.9),
        "risk.gjr_garch_fit": lambda: risk.gjr_garch_fit(g),
        "risk.value_at_risk": lambda: [
            risk.value_at_risk(r, 0.95, method=m)
            for m in ("gaussian", "historical", "cornish_fisher")
        ],
        "risk.variance_backcast": lambda: risk.variance_backcast(g),
        "statistics.adf_test": lambda: statistics.adf_test(d["x_level"], autolag="aic"),
        "statistics.arch_lm_test": lambda: statistics.arch_lm_test(g),
        "statistics.cscv_pbo": lambda: statistics.cscv_pbo(d["panel"][:, :8], n_splits=8),
        "statistics.effective_n_eigen": lambda: statistics.effective_n_eigen(d["panel"]),
        "statistics.engle_granger_test": lambda: statistics.engle_granger_test(
            d["y_level"], d["x_level"]
        ),
        "statistics.factor_loadings": lambda: statistics.factor_loadings(
            d["panel"][:, 0], d["panel"][:, 1:4]
        ),
        "statistics.fama_macbeth_regression": lambda: statistics.fama_macbeth_regression(
            d["panel"][:, :30], np.linspace(0.5, 1.5, 60).reshape(30, 2), n_lags_nw=3
        ),
        "statistics.hac_t_test": lambda: statistics.hac_t_test(r),
        "statistics.jarque_bera_test": lambda: statistics.jarque_bera_test(r),
        "statistics.kpss_test": lambda: statistics.kpss_test(r),
        "statistics.newey_west_cov": lambda: statistics.newey_west_cov(
            d["regressors"], ols_resid, 5
        ),
        "statistics.newey_west_optimal_lags": lambda: statistics.newey_west_optimal_lags(ols_resid),
        "statistics.ols": lambda: statistics.ols(d["response"], d["regressors"]),
        "statistics.optimal_block_length": lambda: statistics.optimal_block_length(r),
        "statistics.ou_half_life": lambda: statistics.ou_half_life(d["ou"]),
        "statistics.spa_test": lambda: statistics.spa_test(
            np.abs(d["benchmark"]),
            np.abs(d["panel"][:, :4]),
            rng=np.random.default_rng(5),
            block_size=10,
            reps=200,
            studentize=True,
        ),
        "statistics.spread_zscore": lambda: statistics.spread_zscore(d["ou"], 20),
        "statistics.stationary_bootstrap_ci": lambda: statistics.stationary_bootstrap_ci(
            r, np.mean, rng=np.random.default_rng(6), n_boot=200, method="bca"
        ),
        "time_series.arima_fit": lambda: arima,
        "time_series.arima_forecast": lambda: time_series.arima_forecast(
            d["arma"], arima[0], arima[1], 0, 5
        ),
        "time_series.arima_residuals": lambda: time_series.arima_residuals(
            d["arma"], arima[0], arima[1], 0
        ),
        "time_series.granger_causality_test": lambda: time_series.granger_causality_test(
            d["system"][:, 0], d["system"][:, 1], 2
        ),
        "time_series.impulse_response": lambda: time_series.impulse_response(
            var_coef, var_sigma, 2, 5
        ),
        "time_series.ljung_box_test": lambda: time_series.ljung_box_test(r, 10),
        "time_series.select_arima_order": lambda: time_series.select_arima_order(
            d["arma"], 2, 1, 2
        ),
        "time_series.select_var_lag_order": lambda: time_series.select_var_lag_order(
            d["system"], 4
        ),
        "time_series.var_fit": lambda: (var_coef, var_sigma),
        "time_series.var_forecast": lambda: time_series.var_forecast(d["system"], var_coef, 2, 5),
        PLANTED: lambda: np.random.default_rng().standard_normal(3),
    }


def _floats(value: Any, out: list[float]) -> None:
    """Append every float in `value` (a float, an array, a sequence or a dataclass) to `out`."""
    if isinstance(value, np.ndarray):
        out.extend(float(v) for v in value.ravel())
    elif isinstance(value, (float, np.floating)):
        out.append(float(value))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _floats(item, out)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _floats(getattr(value, field.name), out)


def run(only: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Digest of each call's result, or of its exception type and message if it raises.

    Returns:
        `{"digests": {name: sha256}, "values": {name: [float.hex(), ...]}}`, with `values` for the
        names in `BLAS_THREAD_SENSITIVE` that did not raise.
    """
    calls = _calls(_inputs())
    names = sorted(only) if only is not None else sorted(set(calls) - {PLANTED})
    digests: dict[str, str] = {}
    values: dict[str, list[str]] = {}
    for name in names:
        try:
            result = calls[name]()
        except Exception as error:
            digests[name] = "raised " + digest((type(error).__name__, str(error)))
            continue
        digests[name] = digest(result)
        if name in BLAS_THREAD_SENSITIVE:
            floats: list[float] = []
            _floats(result, floats)
            values[name] = [v.hex() for v in floats]
    return {"digests": digests, "values": values}


def main() -> None:
    only = os.environ.get("QC23_ONLY")
    selected = set(only.split(",")) if only else None
    json.dump(run(selected), sys.stdout, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
