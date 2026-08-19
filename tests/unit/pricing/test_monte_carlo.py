"""Tests for Monte Carlo option pricing.

Per CLAUDE.md, every Monte Carlo method must have a closed-form reference to
validate against; the actual convergence-to-Black-Scholes check lives in
tests/integration/test_monte_carlo_vs_black_scholes.py. This file exercises
the public contract of monte_carlo_call_price / monte_carlo_put_price in
isolation: shapes/types, invariants implied by the pricing formula

    V_0 = exp(-r * T) * E[payoff(S_T)]

and boundary/degenerate cases that are exactly solvable by hand (zero
volatility collapses the GBM to a deterministic path).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantcore.pricing.monte_carlo import monte_carlo_call_price, monte_carlo_put_price

# ---------------------------------------------------------------------------
# Non-negativity / well-formedness invariants (payoffs are non-negative, so
# the discounted average and its standard error must be too).
# ---------------------------------------------------------------------------


def test_call_price_and_stderr_are_nonnegative() -> None:
    price, stderr = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=2000,
        num_steps=50,
        seed=1,
    )
    assert price >= 0.0
    assert stderr >= 0.0
    assert math.isfinite(price)
    assert math.isfinite(stderr)


def test_put_price_and_stderr_are_nonnegative() -> None:
    price, stderr = monte_carlo_put_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=2000,
        num_steps=50,
        seed=1,
    )
    assert price >= 0.0
    assert stderr >= 0.0
    assert math.isfinite(price)
    assert math.isfinite(stderr)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_price() -> None:
    kwargs = dict(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=500,
        num_steps=20,
        seed=42,
    )
    price1, stderr1 = monte_carlo_call_price(**kwargs)
    price2, stderr2 = monte_carlo_call_price(**kwargs)
    assert price1 == price2
    assert stderr1 == stderr2


def test_no_seed_still_returns_valid_price() -> None:
    # seed=None should not error; just check the contract still holds.
    price, stderr = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=500,
        num_steps=20,
        seed=None,
    )
    assert price >= 0.0
    assert stderr >= 0.0


# ---------------------------------------------------------------------------
# Degenerate / exactly-solvable cases: with volatility=0 the GBM has no
# diffusion term, so the terminal price is deterministic and the MC estimate
# must equal the discounted intrinsic payoff exactly (up to floating point),
# with (near) zero standard error since there is no sampling variance.
# ---------------------------------------------------------------------------


def test_zero_volatility_call_is_deterministic() -> None:
    spot, strike, rate, t, num_steps = 100.0, 90.0, 0.05, 1.0, 50
    price, stderr = monte_carlo_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=0.0,
        time_to_maturity=t,
        num_paths=100,
        num_steps=num_steps,
        seed=7,
    )
    # Zero volatility collapses the GBM to the module's documented
    # Euler-Maruyama recursion S_{k+1} = S_k * (1 + r * dt), not the
    # continuous-compounding solution S_0 * exp(r * T) -- see the identical
    # zero-volatility case in tests/unit/core/test_sde_solver.py.
    dt = t / num_steps
    terminal = spot * (1.0 + rate * dt) ** num_steps
    expected = math.exp(-rate * t) * max(terminal - strike, 0.0)
    assert price == pytest.approx(expected, abs=1e-6)
    assert stderr == pytest.approx(0.0, abs=1e-6)


def test_zero_volatility_put_is_deterministic() -> None:
    spot, strike, rate, t, num_steps = 100.0, 110.0, 0.05, 1.0, 50
    price, stderr = monte_carlo_put_price(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=0.0,
        time_to_maturity=t,
        num_paths=100,
        num_steps=num_steps,
        seed=7,
    )
    # See test_zero_volatility_call_is_deterministic: discrete, not
    # continuous, compounding is the correct closed form here.
    dt = t / num_steps
    terminal = spot * (1.0 + rate * dt) ** num_steps
    expected = math.exp(-rate * t) * max(strike - terminal, 0.0)
    assert price == pytest.approx(expected, abs=1e-6)
    assert stderr == pytest.approx(0.0, abs=1e-6)


def test_zero_volatility_deep_otm_call_is_worthless() -> None:
    # Deterministic terminal value below the strike -> exactly zero payoff.
    price, stderr = monte_carlo_call_price(
        spot=50.0,
        strike=1000.0,
        rate=0.01,
        volatility=0.0,
        time_to_maturity=1.0,
        num_paths=50,
        num_steps=10,
        seed=3,
    )
    assert price == pytest.approx(0.0, abs=1e-9)
    assert stderr == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Boundary values that must still "just work" (not raise).
# ---------------------------------------------------------------------------


def test_num_paths_one_does_not_crash() -> None:
    price, _stderr = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=1,
        num_steps=10,
        seed=1,
        antithetic=False,
    )
    assert price >= 0.0
    assert math.isfinite(price)


def test_num_steps_one_does_not_crash() -> None:
    price, stderr = monte_carlo_put_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=200,
        num_steps=1,
        seed=1,
    )
    assert price >= 0.0
    assert stderr >= 0.0


# ---------------------------------------------------------------------------
# Invalid inputs: counts of paths/steps that cannot yield a defined estimate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_paths", [0, -1])
def test_non_positive_num_paths_raises(num_paths: int) -> None:
    with pytest.raises(ValueError):
        monte_carlo_call_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_paths=num_paths,
            num_steps=10,
            seed=1,
        )


@pytest.mark.parametrize("num_steps", [0, -1])
def test_non_positive_num_steps_raises(num_steps: int) -> None:
    with pytest.raises(ValueError):
        monte_carlo_put_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_paths=10,
            num_steps=num_steps,
            seed=1,
        )


# ---------------------------------------------------------------------------
# Variance reduction: antithetic variates are a monotone transform of the
# same draws paired with their mirror image. For a monotone payoff (call and
# put payoffs are both piecewise-monotone in S_T) this is proven (Boyle,
# Broadie & Glasserman, 1997) to not increase estimator variance versus
# plain Monte Carlo with the same path count. We allow a small slack factor
# to avoid flakiness from sampling noise while still checking the direction
# of the effect.
# ---------------------------------------------------------------------------


def test_antithetic_variates_do_not_increase_standard_error() -> None:
    kwargs = dict(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=5000,
        num_steps=50,
        seed=123,
    )
    _price_plain, stderr_plain = monte_carlo_call_price(**kwargs, antithetic=False)
    _price_anti, stderr_anti = monte_carlo_call_price(**kwargs, antithetic=True)
    assert stderr_anti <= stderr_plain * 1.1


# ---------------------------------------------------------------------------
# Put payoff bound: a European put can never be worth more than the
# discounted strike (K * e^{-rT}), since the maximum payoff max(K - S_T, 0)
# is bounded by K and the estimate is a discounted average of that payoff.
# ---------------------------------------------------------------------------


def test_put_price_never_exceeds_discounted_strike() -> None:
    strike, rate, t = 100.0, 0.05, 1.0
    price, _stderr = monte_carlo_put_price(
        spot=100.0,
        strike=strike,
        rate=rate,
        volatility=0.5,
        time_to_maturity=t,
        num_paths=2000,
        num_steps=50,
        seed=9,
    )
    assert price <= strike * math.exp(-rate * t) + 1e-9


def test_returns_tuple_of_floats() -> None:
    result = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=10,
        num_steps=5,
        seed=1,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    price, stderr = result
    assert isinstance(price, float)
    assert isinstance(stderr, float)


def test_output_array_dtype_is_numpy_compatible() -> None:
    # np.isscalar style sanity check that outputs behave like plain floats
    # (not, e.g., 0-d arrays with surprising broadcasting behavior downstream).
    price, stderr = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=10,
        num_steps=5,
        seed=1,
    )
    assert np.isscalar(price) or isinstance(price, float)
    assert np.isscalar(stderr) or isinstance(stderr, float)
