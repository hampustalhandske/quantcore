"""Integration: raw GBM path simulation, averaged and discounted by hand,
should also recover the Black-Scholes price.

This is a second, more end-to-end version of the Monte Carlo convergence
check: rather than going through monte_carlo_call_price/put_price, it drives
quantcore.core.sde_solver.simulate_gbm_paths directly and does the
payoff/discounting/standard-error arithmetic in the test itself. This
exercises core -> pricing numerical agreement without relying on the pricing
module's own implementation of the discounting step.
"""

from __future__ import annotations

import numpy as np

from quantcore.core.sde_solver import simulate_gbm_paths
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put

STDERR_TOLERANCE = 4.0
NUM_PATHS = 50_000
NUM_STEPS = 100
SEED = 777


def test_gbm_terminal_values_recover_black_scholes_call_price() -> None:
    spot, strike, rate, vol, t = 100.0, 100.0, 0.05, 0.2, 1.0

    paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_steps=NUM_STEPS,
        num_paths=NUM_PATHS,
        seed=SEED,
    )
    terminal = paths[:, -1]
    payoffs = np.maximum(terminal - strike, 0.0)
    discounted_payoffs = np.exp(-rate * t) * payoffs

    mc_price = discounted_payoffs.mean()
    mc_stderr = discounted_payoffs.std(ddof=1) / np.sqrt(NUM_PATHS)

    bs_price = black_scholes_call(spot, strike, rate, vol, t)
    assert abs(mc_price - bs_price) <= STDERR_TOLERANCE * mc_stderr


def test_gbm_terminal_values_recover_black_scholes_put_price() -> None:
    spot, strike, rate, vol, t = 100.0, 95.0, 0.03, 0.25, 0.5

    paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_steps=NUM_STEPS,
        num_paths=NUM_PATHS,
        seed=SEED,
    )
    terminal = paths[:, -1]
    payoffs = np.maximum(strike - terminal, 0.0)
    discounted_payoffs = np.exp(-rate * t) * payoffs

    mc_price = discounted_payoffs.mean()
    mc_stderr = discounted_payoffs.std(ddof=1) / np.sqrt(NUM_PATHS)

    bs_price = black_scholes_put(spot, strike, rate, vol, t)
    assert abs(mc_price - bs_price) <= STDERR_TOLERANCE * mc_stderr


def test_gbm_terminal_value_mean_matches_risk_neutral_expectation() -> None:
    # Under the risk-neutral measure, E[S_T] = S_0 * e^{rT} exactly (this is
    # the defining property of GBM risk-neutral drift, independent of any
    # option payoff). Check the Monte Carlo average of simulated terminal
    # values matches this closed form within sampling error.
    spot, rate, vol, t = 100.0, 0.05, 0.2, 1.0

    paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_steps=NUM_STEPS,
        num_paths=NUM_PATHS,
        seed=SEED,
    )
    terminal = paths[:, -1]
    sample_mean = terminal.mean()
    sample_stderr = terminal.std(ddof=1) / np.sqrt(NUM_PATHS)

    expected_mean = spot * np.exp(rate * t)
    assert abs(sample_mean - expected_mean) <= STDERR_TOLERANCE * sample_stderr
