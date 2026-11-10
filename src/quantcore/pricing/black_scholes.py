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

Every input must be finite. With `volatility = 0` and `T > 0` the price is
its sigma -> 0+ limit, the discounted forward intrinsic value, computed
directly:

    Call = max(S*exp(-qT) - K*exp(-rT), 0)
    Put  = max(K*exp(-rT) - S*exp(-qT), 0)

which is 0 when the forward sits exactly at the strike. A price that
overflows float64 raises `quantcore.UndefinedResultError` (a `ValueError`).

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." *Journal of Political Economy*, 81(3), 637-654.
    Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal
    of Economics and Management Science*, 4(1), 141-183. (Continuous
    dividend yield extension.)
    See docs/REFERENCES.md.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from quantcore.errors import _finite_result


def _validate_finite_inputs(**inputs: float) -> None:
    """Raise ValueError naming the first of `inputs` that is NaN or infinite."""
    for name, value in inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _validate_price_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float,
) -> None:
    _validate_finite_inputs(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        dividend_yield=dividend_yield,
    )
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")


def _discounted_legs(
    spot: float, strike: float, rate: float, time_to_maturity: float, dividend_yield: float
) -> tuple[float, float]:
    """`(S*exp(-qT), K*exp(-rT))`; either is `inf` when the discount factor overflows."""
    with np.errstate(over="ignore"):
        spot_leg = float(spot * np.exp(-dividend_yield * time_to_maturity))
        strike_leg = float(strike * np.exp(-rate * time_to_maturity))
    return spot_leg, strike_leg


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

        With `volatility` 0 and `time_to_maturity` > 0, the limit
        `max(S*exp(-qT) - K*exp(-rT), 0)`.

    Raises:
        ValueError: An input is not finite, `spot` or `strike` is not
            strictly positive, or `volatility` or `time_to_maturity` is
            negative.
        UndefinedResultError: The price overflows float64.
    """
    _validate_price_inputs(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    if time_to_maturity == 0.0:
        return max(spot - strike, 0.0)
    if volatility == 0.0:
        spot_leg, strike_leg = _discounted_legs(
            spot, strike, rate, time_to_maturity, dividend_yield
        )
        return _finite_result(max(spot_leg - strike_leg, 0.0), "call price")

    d1 = (
        np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    price = float(
        spot * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1)
        - strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2)
    )
    return _finite_result(price, "call price")


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

    Returns:
        The fair price of the put option. With `volatility` 0 and
        `time_to_maturity` > 0, the limit `max(K*exp(-rT) - S*exp(-qT), 0)`.

    Raises:
        ValueError: Any input condition of `black_scholes_call`.
        UndefinedResultError: The price overflows float64.
    """
    _validate_price_inputs(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    if volatility == 0.0 and time_to_maturity > 0.0:
        spot_leg, strike_leg = _discounted_legs(
            spot, strike, rate, time_to_maturity, dividend_yield
        )
        return _finite_result(max(strike_leg - spot_leg, 0.0), "put price")
    call_price = black_scholes_call(
        spot, strike, rate, volatility, time_to_maturity, dividend_yield
    )
    # Put-call parity with a dividend yield: C - P = S*exp(-qT) - K*exp(-rT).
    price = float(
        call_price
        - spot * np.exp(-dividend_yield * time_to_maturity)
        + strike * np.exp(-rate * time_to_maturity)
    )
    return _finite_result(price, "put price")
