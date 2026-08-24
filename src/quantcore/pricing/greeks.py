"""Black-Scholes option Greeks and implied volatility inversion.

    d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Delta_call = Phi(d1)            Delta_put  = Phi(d1) - 1
    Gamma       = phi(d1) / (S*sigma*sqrt(T))
    Vega        = S*phi(d1)*sqrt(T)  (per unit of vol, i.e. divide by 100 for per-% move)
    Theta_call  = -S*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*Phi(d2)
    Theta_put   = -S*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*Phi(-d2)
    Rho_call    =  K*T*exp(-rT)*Phi(d2)
    Rho_put     = -K*T*exp(-rT)*Phi(-d2)

    Phi = standard normal CDF, phi = standard normal PDF.

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." *Journal of Political Economy*, 81(3), 637-654.

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
    spot: float, strike: float, rate: float, volatility: float, t: float
) -> tuple[float, float]:

    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
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
) -> float:
    """Black-Scholes Delta: sensitivity of option price to a 1-unit move in spot.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".

    Returns:
        Delta_call = Phi(d1), Delta_put = Phi(d1) - 1.
    """
    _validate_bs_delta(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_delta(spot, strike, rate, volatility, time_to_maturity, option_type)


def _bs_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> float:
    if time_to_maturity == 0.0:
        in_the_money = spot > strike
        if option_type == "call":
            return 1.0 if in_the_money else 0.0
        return -1.0 if not in_the_money else 0.0

    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


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
) -> float:
    """Black-Scholes Gamma: sensitivity of Delta to a 1-unit move in spot (same for call/put).

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).

    Returns:
        Gamma = phi(d1) / (S*sigma*sqrt(T)).
    """
    _validate_bs_gamma(spot, strike, rate, volatility, time_to_maturity)
    return _bs_gamma(spot, strike, rate, volatility, time_to_maturity)


def _bs_gamma(
    spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float
) -> float:

    if time_to_maturity == 0.0 or volatility == 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_maturity)))


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
) -> float:
    """Black-Scholes Vega: sensitivity of option price to a 1-unit move in volatility.

    Same for call and put.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).

    Returns:
        Vega = S*phi(d1)*sqrt(T), per unit (100 vol points) of volatility.
    """
    _validate_bs_vega(spot, strike, rate, volatility, time_to_maturity)
    return _bs_vega(spot, strike, rate, volatility, time_to_maturity)


def _bs_vega(
    spot: float, strike: float, rate: float, volatility: float, time_to_maturity: float
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    return float(spot * norm.pdf(d1) * np.sqrt(time_to_maturity))


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
) -> float:
    """Black-Scholes Theta, per calendar day (annualized Theta divided by 365).

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".

    Returns:
        Theta per calendar day (time decay of option value).
    """
    _validate_bs_theta(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_theta(spot, strike, rate, volatility, time_to_maturity, option_type)


def _bs_theta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    d1, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    discount = np.exp(-rate * time_to_maturity)
    common = -spot * norm.pdf(d1) * volatility / (2.0 * np.sqrt(time_to_maturity))
    if option_type == "call":
        theta_annual = common - rate * strike * discount * norm.cdf(d2)
    else:
        theta_annual = common + rate * strike * discount * norm.cdf(-d2)
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
) -> float:
    """Black-Scholes Rho: sensitivity of option price to a 1-unit move in the risk-free rate.

    Args:
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".

    Returns:
        Rho_call = K*T*exp(-rT)*Phi(d2), Rho_put = -K*T*exp(-rT)*Phi(-d2).
    """
    _validate_bs_rho(spot, strike, rate, volatility, time_to_maturity, option_type)
    return _bs_rho(spot, strike, rate, volatility, time_to_maturity, option_type)


def _bs_rho(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str,
) -> float:

    if time_to_maturity == 0.0:
        return 0.0
    _, d2 = _d1_d2(spot, strike, rate, volatility, time_to_maturity)
    discount = np.exp(-rate * time_to_maturity)
    if option_type == "call":
        return float(strike * time_to_maturity * discount * norm.cdf(d2))
    return float(-strike * time_to_maturity * discount * norm.cdf(-d2))


_IV_LOWER_BOUND = 1e-9
_IV_UPPER_BOUND = 10.0


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
) -> float:
    """Invert the Black-Scholes formula to recover the implied volatility.

    Args:
        market_price: Observed option price.
        spot: Current price of the underlying asset (S).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        time_to_maturity: Time to expiry in years (T).
        option_type: "call" or "put".
        tol: Root-finding tolerance for `scipy.optimize.brentq`.
        max_iter: Maximum number of `brentq` iterations.

    Returns:
        The volatility sigma such that the Black-Scholes price equals `market_price`.

    Raises:
        ValueError: If no root exists in [1e-9, 10.0] (e.g. `market_price` is
            below intrinsic value or otherwise outside no-arbitrage bounds).
    """
    _validate_implied_volatility(market_price, spot, strike, rate, time_to_maturity, option_type)
    return _implied_volatility(
        market_price, spot, strike, rate, time_to_maturity, option_type, tol, max_iter
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
) -> float:
    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma: float) -> float:
        return pricer(spot, strike, rate, sigma, time_to_maturity) - market_price

    lo, hi = objective(_IV_LOWER_BOUND), objective(_IV_UPPER_BOUND)
    if lo * hi > 0.0:
        raise ValueError(
            "implied_volatility: no root in [1e-9, 10.0] -- market_price is outside "
            "no-arbitrage bounds for the given parameters"
        )
    return float(brentq(objective, _IV_LOWER_BOUND, _IV_UPPER_BOUND, xtol=tol, maxiter=max_iter))
