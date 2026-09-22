"""Closed-form Black-Scholes(-Merton) option pricing.

Reference implementation in pure NumPy/SciPy. This is the "ground truth" that
Monte Carlo pricers in this package are validated against — see
tests/pricing/test_black_scholes.py.

`dividend_yield` (q) generalizes Black-Scholes (1973) to Black-Scholes-
Merton: a continuously compounded dividend (or, for FX/futures, a
cost-of-carry) yield, entering d1/d2 as `r - q` in place of `r`, and
discounting the spot leg by `exp(-q*T)`:

    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Call = S*exp(-qT)*Phi(d1) - K*exp(-rT)*Phi(d2)
    Put  = K*exp(-rT)*Phi(-d2) - S*exp(-qT)*Phi(-d1)

`dividend_yield=0.0` (the default) reduces exactly to the original
Black-Scholes (1973) formulas.

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." *Journal of Political Economy*, 81(3), 637-654.
    Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal
    of Economics and Management Science*, 4(1), 141-183. (Continuous
    dividend yield extension.)
    See docs/REFERENCES.md.
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
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call option under Black-Scholes-Merton.

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes
            (1973). See module docstring.

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

    d1 = (
        np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    return float(
        spot * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1)
        - strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2)
    )


def black_scholes_put(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European put option under Black-Scholes-Merton (via put-call parity).

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes
            (1973). See module docstring.
    """
    call_price = black_scholes_call(
        spot, strike, rate, volatility, time_to_maturity, dividend_yield
    )
    # Put-call parity with a dividend yield: C - P = S*exp(-qT) - K*exp(-rT).
    return float(
        call_price
        - spot * np.exp(-dividend_yield * time_to_maturity)
        + strike * np.exp(-rate * time_to_maturity)
    )
