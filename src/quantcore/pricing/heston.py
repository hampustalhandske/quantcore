"""Heston stochastic-volatility option pricing via the COS method.

Heston characteristic function (Heston 1993), stable "little trap"
parameterization (Albrecher et al. 2007), for the log-return X_T = ln(S_T/S0):

    d(u)   = sqrt((kappa - i*rho*xi*u)^2 + xi^2*(i*u + u^2))
    g(u)   = (kappa - i*rho*xi*u - d) / (kappa - i*rho*xi*u + d)
    C(u)   = i*u*r*T + (kappa*theta/xi^2) *
             ((kappa - i*rho*xi*u - d)*T - 2*ln((1 - g*exp(-d*T)) / (1 - g)))
    D(u)   = (kappa - i*rho*xi*u - d)/xi^2 * (1 - exp(-d*T)) / (1 - g*exp(-d*T))
    phi(u) = exp(C(u) + D(u)*v0)

COS method (Fang & Oosterlee 2008) for a European call, with x = ln(S0/K)
and truncation range [a, b] given by cumulants c1, c2:

    price = K*exp(-r*T) * sum_{k=0}^{N-1}' Re[phi(u_k)*exp(i*u_k*(x-a))] * U_k
    u_k = k*pi/(b-a)
    U_k = (2/(b-a)) * (chi_k(0,b) - psi_k(0,b))

where chi_k, psi_k are the standard analytic payoff-coefficient integrals and
the prime on the sum denotes the k=0 term being weighted by one half.

References:
    Heston, S.L. (1993), "A Closed-Form Solution for Options with Stochastic
    Volatility..." *Review of Financial Studies*, 6(2), 327-343.

    Fang, F. and Oosterlee, C.W. (2008), "A Novel Pricing Method for European
    Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on
    Scientific Computing*, 31(2), 826-848.

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_heston_cos_call(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_terms: int,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")
    if v0 <= 0.0:
        raise ValueError("v0 must be strictly positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive")
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive")
    if xi < 0.0:
        raise ValueError("xi must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")
    if n_terms <= 0:
        raise ValueError("n_terms must be a positive integer")


def heston_cos_call(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_terms: int = 128,
) -> float:
    """Price a European call under the Heston model via the COS method.

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        time_to_maturity: Time to expiry in years (T).
        v0: Initial variance.
        kappa: Mean-reversion speed of variance.
        theta: Long-run variance.
        xi: Vol-of-vol.
        rho: Correlation between the spot and variance Brownian motions.
        n_terms: Number of cosine-series terms.

    Returns:
        The Heston call price.

    References:
        Heston (1993); Fang & Oosterlee (2008). See docs/REFERENCES.md.
    """
    _validate_heston_cos_call(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms
    )
    return _heston_cos_call(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms
    )


def _heston_char_func(
    u: npt.NDArray[np.complex128],
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> npt.NDArray[np.complex128]:
    """Heston characteristic function of X_T = ln(S_T/S0), "little trap" form."""
    if xi == 0.0:
        # Degenerate zero-vol-of-vol case: V_t == v0 deterministically, so
        # ln(S_T/S0) is Gaussian with the constant-variance GBM moments.
        mean = (rate - 0.5 * v0) * time_to_maturity
        var = v0 * time_to_maturity
        return np.exp(1j * u * mean - 0.5 * var * u**2)

    xi_bar = kappa - 1j * rho * xi * u
    d = np.sqrt(xi_bar**2 + xi**2 * (1j * u + u**2))
    g = (xi_bar - d) / (xi_bar + d)

    exp_neg_dt = np.exp(-d * time_to_maturity)
    c = (kappa * theta / xi**2) * (
        (xi_bar - d) * time_to_maturity - 2.0 * np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    )
    d_term = ((xi_bar - d) / xi**2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt))

    return np.exp(1j * u * rate * time_to_maturity + c + d_term * v0)


def _heston_cos_call(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_terms: int,
) -> float:
    if time_to_maturity == 0.0:
        return max(spot - strike, 0.0)

    c1 = (
        rate * time_to_maturity
        + (1.0 - np.exp(-kappa * time_to_maturity)) * (theta - v0) / (2.0 * kappa)
        - 0.5 * theta * time_to_maturity
    )
    c2 = (1.0 / (8.0 * kappa**3)) * (
        xi
        * time_to_maturity
        * kappa
        * np.exp(-kappa * time_to_maturity)
        * (v0 - theta)
        * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - np.exp(-kappa * time_to_maturity)) * (16.0 * theta - 8.0 * v0)
        + 2.0
        * theta
        * kappa
        * time_to_maturity
        * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
        + xi**2
        * (
            (theta - 2.0 * v0) * np.exp(-2.0 * kappa * time_to_maturity)
            + theta * (6.0 * np.exp(-kappa * time_to_maturity) - 7.0)
            + 2.0 * v0
        )
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - np.exp(-kappa * time_to_maturity))
    )

    a = c1 - 12.0 * np.sqrt(abs(c2))
    b = c1 + 12.0 * np.sqrt(abs(c2))

    x = np.log(spot / strike)
    k = np.arange(n_terms, dtype=np.float64)
    u = k * np.pi / (b - a)

    phi = _heston_char_func(
        u.astype(np.complex128), rate, time_to_maturity, v0, kappa, theta, xi, rho
    )
    re_term = np.real(phi * np.exp(1j * u * (x - a)))

    d_range, c_range = b, 0.0
    denom = 1.0 + u**2
    cos_d = np.cos(u * (d_range - a))
    cos_c = np.cos(u * (c_range - a))
    sin_d = np.sin(u * (d_range - a))
    sin_c = np.sin(u * (c_range - a))
    chi_k = (1.0 / denom) * (
        cos_d * np.exp(d_range)
        - cos_c * np.exp(c_range)
        + u * sin_d * np.exp(d_range)
        - u * sin_c * np.exp(c_range)
    )
    psi_k = np.empty(n_terms, dtype=np.float64)
    psi_k[0] = d_range - c_range
    psi_k[1:] = (sin_d[1:] - sin_c[1:]) / u[1:]

    u_k = (2.0 / (b - a)) * (chi_k - psi_k)

    weights = np.ones(n_terms)
    weights[0] = 0.5
    price = strike * np.exp(-rate * time_to_maturity) * np.sum(weights * re_term * u_k)

    return float(price)
