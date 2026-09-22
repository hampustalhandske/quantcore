"""Black-Scholes-Merton option Greeks and implied volatility inversion.

    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Delta_call = exp(-qT)*Phi(d1)          Delta_put = -exp(-qT)*Phi(-d1)
    Gamma       = exp(-qT)*phi(d1) / (S*sigma*sqrt(T))
    Vega        = S*exp(-qT)*phi(d1)*sqrt(T)  (per unit of vol, i.e. divide
                  by 100 for per-% move)
    Theta_call  = -S*exp(-qT)*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*Phi(d2)
                  + q*S*exp(-qT)*Phi(d1)
    Theta_put   = -S*exp(-qT)*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*Phi(-d2)
                  - q*S*exp(-qT)*Phi(-d1)
    Rho_call    =  K*T*exp(-rT)*Phi(d2)
    Rho_put     = -K*T*exp(-rT)*Phi(-d2)

    Phi = standard normal CDF, phi = standard normal PDF, q = dividend
    yield / cost-of-carry adjustment. q=0.0 (the default everywhere below)
    reduces every formula exactly to plain Black-Scholes (1973). Rho's
    closed form doesn't reference q explicitly, but its value still shifts
    with q through d2 = d1 - sigma*sqrt(T), since d1 is a function of r - q.

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." *Journal of Political Economy*, 81(3), 637-654.

    Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal
    of Economics and Management Science*, 4(1), 141-183. (Continuous
    dividend yield extension used by `dividend_yield` throughout.)

    Brent, R.P. (1973). *Algorithms for Minimization Without Derivatives*.
    Prentice-Hall. (Root-finding used by `implied_volatility`.)

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put

_VALID_OPTION_TYPES = ("call", "put")


def _validate_common_inputs(
    spot: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")


def _validate_option_type(option_type: str) -> None:
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_OPTION_TYPES}, got {option_type!r}")


def _d1_d2(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    t: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:

    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * t) / (
        volatility * np.sqrt(t)
    )
    d2 = d1 - volatility * np.sqrt(t)
    return float(d1), float(d2)


def _validate_bs_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> None:
    _validate_common_inputs(spot, strike, volatility, time_to_maturity)
    _validate_option_type(option_type)


def bs_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton Delta: sensitivity of option price to a 1-unit move in spot.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes.

    Returns:
        Delta_call = exp(-qT)*Phi(d1), Delta_put = exp(-qT)*(Phi(d1) - 1).
    """
    _validate_bs_delta(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_delta(spot, strike, rate, volatility, time_to_maturity, option_type, dividend_yield)


def _bs_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    if time_to_maturity == 0.0:
        in_the_money = spot > strike
        if option_type == "call":
            return 1.0 if in_the_money else 0.0
        return -1.0 if not in_the_money else 0.0

    discount = np.exp(-dividend_yield * time_to_maturity)
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    if option_type == "call":
        return float(discount * norm.cdf(d1))
    return float(discount * (norm.cdf(d1) - 1.0))


def _validate_bs_gamma(
    spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float
) -> None:
    _validate_common_inputs(spot, strike, volatility, time_to_maturity)


def bs_gamma(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton Gamma: sensitivity of Delta to a 1-unit move in spot.

    Same for call and put.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes.

    Returns:
        Gamma = exp(-qT)*phi(d1) / (S*sigma*sqrt(T)).
    """
    _validate_bs_gamma(spot, strike, rate, volatility, time_to_maturity)
    return _bs_gamma(spot, strike, rate, volatility, time_to_maturity, dividend_yield)


def _bs_gamma(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:

    if time_to_maturity == 0.0 or volatility == 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    discount = np.exp(-dividend_yield * time_to_maturity)
    return float(discount * norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_maturity)))


def _validate_bs_vega(
    spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float
) -> None:
    _validate_common_inputs(spot, strike, volatility, time_to_maturity)


def bs_vega(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton Vega: sensitivity of option price to a 1-unit move in volatility.

    Same for call and put.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes.

    Returns:
        Vega = S*exp(-qT)*phi(d1)*sqrt(T), per unit (100 vol points) of
        volatility.
    """
    _validate_bs_vega(spot, strike, rate, volatility, time_to_maturity)
    return _bs_vega(spot, strike, rate, volatility, time_to_maturity, dividend_yield)


def _bs_vega(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    discount = np.exp(-dividend_yield * time_to_maturity)
    return float(spot * discount * norm.pdf(d1) * np.sqrt(time_to_maturity))


def _validate_bs_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> None:
    _validate_common_inputs(spot, strike, volatility, time_to_maturity)
    _validate_option_type(option_type)


def bs_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton Theta, per calendar day (annualized Theta divided by 365).

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes.

    Returns:
        Theta per calendar day (time decay of option value).
    """
    _validate_bs_theta(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_theta(spot, strike, rate, volatility, time_to_maturity, option_type, dividend_yield)


def _bs_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    rate_discount = np.exp(-rate * time_to_maturity)
    div_discount = np.exp(-dividend_yield * time_to_maturity)
    common = -spot * div_discount * norm.pdf(d1) * volatility / (2.0 * np.sqrt(time_to_maturity))
    if option_type == "call":
        theta_annual = (
            common
            - rate * strike * rate_discount * norm.cdf(d2)
            + dividend_yield * spot * div_discount * norm.cdf(d1)
        )
    else:
        theta_annual = (
            common
            + rate * strike * rate_discount * norm.cdf(-d2)
            - dividend_yield * spot * div_discount * norm.cdf(-d1)
        )
    return float(theta_annual / 365.0)


def _validate_bs_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> None:
    _validate_common_inputs(spot, strike, volatility, time_to_maturity)
    _validate_option_type(option_type)


def bs_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton Rho: sensitivity of option price to a 1-unit move in the risk-free rate.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). The closed-form expression's shape
            (K*T*exp(-rT)*Phi(d2)) doesn't reference q directly, but its
            *value* still depends on q through d2 = d1 - sigma*sqrt(T),
            since d1 is a function of r - q.

    Returns:
        Rho_call = K*T*exp(-rT)*Phi(d2), Rho_put = -K*T*exp(-rT)*Phi(-d2).
    """
    _validate_bs_rho(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_rho(spot, strike, rate, volatility, time_to_maturity, option_type, dividend_yield)


def _bs_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    _, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity, dividend_yield)
    discount = np.exp(-rate * time_to_maturity)
    if option_type == "call":
        return float(strike * time_to_maturity * discount * norm.cdf(d2))
    return float(-strike * time_to_maturity * discount * norm.cdf(-d2))


_IV_LOWER_BOUND = 1e-9
_IV_UPPER_BOUND = 10.0
# Below this vega (relative to spot), the price equation has no reliably
# identifiable root: for deep ITM/OTM options the price is essentially flat
# in sigma across most of [_IV_LOWER_BOUND, _IV_UPPER_BOUND] at double
# precision, so Brent's method can "converge" to a sigma indistinguishable
# from the search boundary even though the true sigma is elsewhere entirely
# -- see `_implied_volatility`'s post-hoc check.
_IV_MIN_RELATIVE_VEGA = 1e-8


def _validate_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    option_type: str,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")
    _validate_option_type(option_type)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    option_type: str,
    tol: float = 1e-8,
    max_iter: int = 100,
    dividend_yield: float = 0.0,
) -> float:
    """Invert the Black-Scholes-Merton formula to recover the implied volatility.

    Args:
        market_price: Observed option price.
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".
        tol: Root-finding tolerance for `scipy.optimize.brentq`.
        max_iter: Maximum number of `brentq` iterations.
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Black-Scholes.

    Returns:
        The volatility sigma such that the Black-Scholes-Merton price
        equals `market_price`.

    Raises:
        ValueError: If no root exists in [1e-9, 10.0] (e.g. `market_price` is
            below intrinsic value or otherwise outside no-arbitrage bounds),
            or if the root found is not reliably identifiable because vega
            is negligible there (deep ITM/OTM and/or very short expiry: the
            price is essentially flat in sigma, so a boundary value would
            otherwise be returned as if it were a real answer).
    """
    _validate_implied_volatility(market_price, spot, strike, rate, time_to_maturity, option_type)
    return _implied_volatility(
        market_price,
        spot,
        strike,
        rate,
        time_to_maturity,
        option_type,
        tol,
        max_iter,
        dividend_yield,
    )


def _implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    option_type: str,
    tol: float,
    max_iter: int,
    dividend_yield: float = 0.0,
) -> float:
    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma: float) -> float:
        return pricer(spot, strike, rate, sigma, time_to_maturity, dividend_yield) - market_price

    lo, hi = objective(_IV_LOWER_BOUND), objective(_IV_UPPER_BOUND)
    if lo * hi > 0.0:
        raise ValueError(
            "implied_volatility: no root in [1e-9, 10.0] -- market_price is outside "
            "no-arbitrage bounds for the given parameters"
        )
    sigma = float(brentq(objective, _IV_LOWER_BOUND, _IV_UPPER_BOUND, xtol=tol, maxiter=max_iter))

    vega = _bs_vega(spot, strike, rate, sigma, time_to_maturity, dividend_yield)
    if vega < _IV_MIN_RELATIVE_VEGA * spot:
        raise ValueError(
            "implied_volatility: no reliably identifiable root -- vega is negligible "
            f"at the candidate sigma={sigma:.6g} (deep ITM/OTM and/or very short expiry "
            "make the option price essentially insensitive to volatility here, so no "
            "root-finder can recover a meaningful implied volatility)"
        )
    return sigma
