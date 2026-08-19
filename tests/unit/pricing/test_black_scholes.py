"""Tests for Black-Scholes closed-form pricing.

These are the reference values other pricers (Monte Carlo, PDE solvers) in
this package get validated against.
"""

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put


def test_call_matches_known_value() -> None:
    # Classic textbook example: S=100, K=100, r=0.05, sigma=0.2, T=1
    # Known Black-Scholes call price ≈ 10.4506
    price = black_scholes_call(
        spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0
    )
    assert abs(price - 10.4506) < 1e-3


def test_put_call_parity() -> None:
    call = black_scholes_call(
        spot=100.0, strike=95.0, rate=0.03, volatility=0.25, time_to_maturity=0.5
    )
    put = black_scholes_put(
        spot=100.0, strike=95.0, rate=0.03, volatility=0.25, time_to_maturity=0.5
    )
    # Put-call parity: C - P = S - K * e^(-rT)
    import numpy as np

    expected_diff = 100.0 - 95.0 * np.exp(-0.03 * 0.5)
    assert abs((call - put) - expected_diff) < 1e-9


def test_zero_time_to_maturity_is_intrinsic_value() -> None:
    price = black_scholes_call(
        spot=110.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=0.0
    )
    assert price == 10.0
