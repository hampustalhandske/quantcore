"""Closed-form Black-Scholes option pricing.

Reference implementation in pure NumPy/SciPy. This is the "ground truth" that
Monte Carlo pricers in this package are validated against — see
tests/pricing/test_black_scholes.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float:
    """Price a European call option under Black-Scholes.

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).

    Returns:
        The fair price of the call option.
    """
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")
    if time_to_maturity == 0.0:
        return max(spot - strike, 0.0)

    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (
        volatility * np.sqrt(time_to_maturity)
    )
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    return float(spot * norm.cdf(d1) - strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2))


def black_scholes_put(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
) -> float:
    """Price a European put option under Black-Scholes (via put-call parity)."""
    call_price = black_scholes_call(spot, strike, rate, volatility, time_to_maturity)
    return float(call_price - spot + strike * np.exp(-rate * time_to_maturity))
