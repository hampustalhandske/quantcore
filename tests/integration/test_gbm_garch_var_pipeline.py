"""Integration: a realistic cross-module pipeline -- simulate asset paths
(core.sde_solver), derive log-returns, fit conditional variance
(risk.volatility), and feed the result into risk metrics (risk.var).

This doesn't have a single closed-form target (unlike the Black-Scholes
comparisons), so the assertions are structural/invariant-based: shapes line
up, outputs stay finite and non-negative, and the CVaR >= VaR relationship
continues to hold when the input returns come from simulated GBM rather
than a hand-constructed array. This is meant to catch integration bugs
(e.g. mismatched array orientations, off-by-one length errors between
modules) that per-module unit tests with hand-picked arrays wouldn't catch.
"""

from __future__ import annotations

import numpy as np

from quantcore.core.sde_solver import simulate_gbm_paths
from quantcore.risk.var import conditional_value_at_risk, value_at_risk
from quantcore.risk.volatility import garch_11_variance


def test_gbm_log_returns_through_garch_and_var_pipeline() -> None:
    spot, rate, vol, t = 100.0, 0.03, 0.25, 1.0
    num_steps, num_paths = 252, 1

    paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=555,
    )
    price_path = paths[0]
    assert price_path.shape == (num_steps + 1,)

    log_returns = np.diff(np.log(price_path))
    assert log_returns.shape == (num_steps,)

    mean_adjusted = log_returns - log_returns.mean()

    # Stationary GARCH(1,1) parameterization (alpha + beta < 1).
    omega, alpha, beta = 1e-6, 0.08, 0.88
    conditional_variance = garch_11_variance(mean_adjusted, omega=omega, alpha=alpha, beta=beta)
    assert conditional_variance.shape == mean_adjusted.shape
    assert np.all(conditional_variance >= 0.0)
    assert np.all(np.isfinite(conditional_variance))

    for confidence_level in (0.95, 0.99):
        var = value_at_risk(mean_adjusted, confidence_level)
        cvar = conditional_value_at_risk(mean_adjusted, confidence_level)
        assert np.isfinite(var)
        assert np.isfinite(cvar)
        assert cvar >= var


def test_pipeline_reproducible_end_to_end_with_fixed_seed() -> None:
    def run_pipeline() -> tuple[float, float]:
        paths = simulate_gbm_paths(
            spot=100.0,
            rate=0.02,
            volatility=0.3,
            time_to_maturity=1.0,
            num_steps=100,
            num_paths=1,
            seed=42,
        )
        log_returns = np.diff(np.log(paths[0]))
        mean_adjusted = log_returns - log_returns.mean()
        var = value_at_risk(mean_adjusted, 0.95)
        cvar = conditional_value_at_risk(mean_adjusted, 0.95)
        return var, cvar

    result1 = run_pipeline()
    result2 = run_pipeline()
    assert result1 == result2
