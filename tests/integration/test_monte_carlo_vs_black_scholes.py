"""Integration: Monte Carlo pricers must converge to the closed-form
Black-Scholes price for the same GBM dynamics.

Per CLAUDE.md: "Every Monte Carlo method must have a closed-form or
reference test to validate against where one exists." Black-Scholes is the
known closed-form solution against which monte_carlo_call_price /
monte_carlo_put_price are validated here, using a fixed seed and a large
enough path count that the check is not flaky: the MC estimate must land
within a small multiple of its own reported standard error of the
closed-form price.
"""

from __future__ import annotations

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.monte_carlo import monte_carlo_call_price, monte_carlo_put_price

# Number of standard errors of slack allowed between the MC estimate and the
# closed-form price. For a correctly implemented, unbiased estimator this
# should hold with overwhelming probability (~1 in 15,000 chance of a false
# failure per test, using the normal-approximation of the CLT for the MC
# average).
STDERR_TOLERANCE = 4.0
NUM_PATHS = 50_000
NUM_STEPS = 100
SEED = 2024


def test_monte_carlo_call_converges_to_black_scholes() -> None:
    spot, strike, rate, vol, t = 100.0, 100.0, 0.05, 0.2, 1.0

    bs_price = black_scholes_call(spot, strike, rate, vol, t)
    mc_price, mc_stderr = monte_carlo_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_paths=NUM_PATHS,
        num_steps=NUM_STEPS,
        seed=SEED,
    )

    assert abs(mc_price - bs_price) <= STDERR_TOLERANCE * mc_stderr


def test_monte_carlo_put_converges_to_black_scholes() -> None:
    spot, strike, rate, vol, t = 100.0, 95.0, 0.03, 0.25, 0.5

    bs_price = black_scholes_put(spot, strike, rate, vol, t)
    mc_price, mc_stderr = monte_carlo_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_paths=NUM_PATHS,
        num_steps=NUM_STEPS,
        seed=SEED,
    )

    assert abs(mc_price - bs_price) <= STDERR_TOLERANCE * mc_stderr


def test_monte_carlo_itm_call_converges_to_black_scholes() -> None:
    # Deep in-the-money case, to check convergence isn't a fluke of the
    # at-the-money example.
    spot, strike, rate, vol, t = 120.0, 90.0, 0.04, 0.3, 0.75

    bs_price = black_scholes_call(spot, strike, rate, vol, t)
    mc_price, mc_stderr = monte_carlo_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=vol,
        time_to_maturity=t,
        num_paths=NUM_PATHS,
        num_steps=NUM_STEPS,
        seed=SEED,
    )

    assert abs(mc_price - bs_price) <= STDERR_TOLERANCE * mc_stderr
