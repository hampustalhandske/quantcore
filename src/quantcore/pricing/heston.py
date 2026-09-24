"""Heston stochastic-volatility option pricing via the COS method.

Heston characteristic function (Heston 1993), stable "little trap"
parameterization (Albrecher et al. 2007), for the log-return X_T = ln(S_T/S0),
under the risk-neutral drift b = r - q (q = continuous dividend yield /
cost-of-carry adjustment, 0.0 by default -- reduces to plain Heston with
drift r):

    d(u)   = sqrt((kappa - i*rho*xi*u)^2 + xi^2*(i*u + u^2))
    g(u)   = (kappa - i*rho*xi*u - d) / (kappa - i*rho*xi*u + d)
    C(u)   = i*u*b*T + (kappa*theta/xi^2) *
             ((kappa - i*rho*xi*u - d)*T - 2*ln((1 - g*exp(-d*T)) / (1 - g)))
    D(u)   = (kappa - i*rho*xi*u - d)/xi^2 * (1 - exp(-d*T)) / (1 - g*exp(-d*T))
    phi(u) = exp(C(u) + D(u)*v0)

COS method (Fang & Oosterlee 2008) in the variable y = ln(S_T/K), whose
characteristic function is phi(u)*exp(i*u*x) with x = ln(S0/K) the
log-moneyness. The put is the quantity expanded, because its payoff
K*(1 - e^y)^+ is bounded on the left tail, whereas the call payoff grows
like e^y on the right tail and turns any truncation of that tail into a
price error (Fang & Oosterlee 2008, Section 3.3):

    put  = K*exp(-r*T) * sum_{k=0}^{N-1}' Re[phi(u_k)*exp(i*u_k*(x-a))] * V_k
    u_k  = k*pi/(b-a)
    V_k  = (2/(b-a)) * (psi_k(a, min(b, 0)) - chi_k(a, min(b, 0)))
    call = put + S0*exp(-q*T) - K*exp(-r*T)          (put-call parity)

where chi_k, psi_k are the standard analytic payoff-coefficient integrals
and the prime on the sum denotes the k=0 term being weighted by one half.
The truncation range is centred on the log-moneyness and sized by the
cumulants of the log-return (Fang & Oosterlee 2008, eq. 49):

    [a, b] = x + c1 -/+ L * sqrt(c2 + sqrt(c4)),   L = 14

c1 and c2 are exact closed forms (`_heston_cumulants`); c4 is obtained from
the cumulant-generating function ln(phi(-i*s)) by a central finite
difference, which is ample for sizing the range. Including c4 widens the
range exactly when the log-return is fat-tailed (vol-of-vol large relative
to the Feller bound), which is what keeps far-from-the-money and
short-dated prices accurate.

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
    n_terms: int = 2048,
    dividend_yield: float = 0.0,
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
        n_terms: Number of cosine-series terms. The default (2048) resolves
            the tail-adaptive truncation range to ~1e-8 relative across
            moneyness 0.5-3, maturities 0.02-2y and vol-of-vol up to 1.0;
            fewer than ~1024 terms loses accuracy on short-dated or
            fat-tailed (high vol-of-vol) cases because the range is sized
            by the fourth cumulant and can be wide there.
        dividend_yield: Continuously compounded dividend yield / cost-of-
            carry adjustment (q). 0.0 (default) is plain Heston, drift r.

    Returns:
        The Heston call price, obtained from the COS put price by put-call
        parity `put + spot*exp(-q*T) - strike*exp(-r*T)`.

    References:
        Heston (1993); Fang & Oosterlee (2008). See docs/REFERENCES.md.
    """
    _validate_heston_cos_call(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms
    )
    return _heston_cos_call(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms, dividend_yield
    )


def heston_cos_put(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_terms: int = 2048,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European put under the Heston model via the COS method.

    Args: see `heston_cos_call`.

    Returns:
        The Heston put price. This is the quantity the cosine expansion
        computes directly; `heston_cos_call` is derived from it by put-call
        parity, so `call - put == spot*exp(-q*T) - strike*exp(-r*T)` holds
        to round-off.

    References:
        Heston (1993); Fang & Oosterlee (2008). See docs/REFERENCES.md.
    """
    _validate_heston_cos_call(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms
    )
    return _heston_cos_put(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms, dividend_yield
    )


def _heston_char_func(
    u: npt.NDArray[np.complex128],
    drift: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> npt.NDArray[np.complex128]:
    """Heston characteristic function of X_T = ln(S_T/S0), "little trap" form.

    `drift` is the risk-neutral drift b = r - q.
    """
    if xi == 0.0:
        # Degenerate zero-vol-of-vol case: the variance path is the
        # deterministic mean-reverting curve v_t = theta + (v0 - theta) *
        # exp(-kappa*t), so ln(S_T/S0) is Gaussian with variance equal to
        # the integrated variance over [0, T].
        integrated_var = _heston_expected_integrated_variance(time_to_maturity, v0, kappa, theta)
        mean = drift * time_to_maturity - 0.5 * integrated_var
        return np.exp(1j * u * mean - 0.5 * integrated_var * u**2)

    xi_bar = kappa - 1j * rho * xi * u
    d = np.sqrt(xi_bar**2 + xi**2 * (1j * u + u**2))
    g = (xi_bar - d) / (xi_bar + d)

    exp_neg_dt = np.exp(-d * time_to_maturity)
    c = (kappa * theta / xi**2) * (
        (xi_bar - d) * time_to_maturity - 2.0 * np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    )
    d_term = ((xi_bar - d) / xi**2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt))

    return np.exp(1j * u * drift * time_to_maturity + c + d_term * v0)


def _heston_expected_integrated_variance(
    time_to_maturity: float, v0: float, kappa: float, theta: float
) -> float:
    """E[integral_0^T v_t dt] under CIR variance dynamics."""
    return float(
        theta * time_to_maturity + (v0 - theta) * (1.0 - np.exp(-kappa * time_to_maturity)) / kappa
    )


_COS_TRUNCATION_L = 14.0
_CGF_FD_STEP = 0.05


def _heston_cumulants(
    drift: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> tuple[float, float, float]:
    """First, second and fourth cumulants of the log-return ln(S_T/S0).

    c1 and c2 are exact. c2 is Var[-I/2 + M] with I the integrated variance
    and M the spot Brownian integral: E[I] + Var[I]/4 - (rho/xi) *
    (Cov[I, v_T] + kappa*Var[I]), evaluated in closed form from the CIR
    covariance function. (This differs from the c2 printed in Fang &
    Oosterlee's Table 11, whose xi^2 * theta term reads 6e^{-kT} - 7 where
    the variance actually requires 4e^{-kT} - 5; the printed expression
    understates the variance whenever xi > 0.)

    c4 is the fourth derivative at 0 of the cumulant-generating function
    K(s) = ln(phi(-i*s)), by a five-point central difference with step
    `_CGF_FD_STEP`, accurate to ~1% -- ample for a truncation range. A
    non-finite or negative estimate is treated as 0, which reduces the range
    to the c2-only form.
    """
    t = time_to_maturity
    e1 = np.exp(-kappa * t)
    e2 = np.exp(-2.0 * kappa * t)
    integrated_var = _heston_expected_integrated_variance(t, v0, kappa, theta)
    c1 = drift * t - 0.5 * integrated_var
    c2 = (1.0 / (8.0 * kappa**3)) * (
        xi * t * kappa * e1 * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - e1) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * t * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
        + xi**2 * ((theta - 2.0 * v0) * e2 + theta * (4.0 * e1 - 5.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - e1)
    )
    if xi == 0.0:
        return float(c1), float(c2), 0.0

    h = _CGF_FD_STEP
    s_grid = np.array([-2.0 * h, -h, 0.0, h, 2.0 * h])
    with np.errstate(all="ignore"):
        cgf = np.log(
            _heston_char_func(
                (-1j * s_grid).astype(np.complex128), drift, t, v0, kappa, theta, xi, rho
            )
        ).real
        c4 = float((cgf[0] - 4.0 * cgf[1] + 6.0 * cgf[2] - 4.0 * cgf[3] + cgf[4]) / h**4)
    if not np.isfinite(c4) or c4 < 0.0:
        c4 = 0.0
    return float(c1), float(c2), c4


def _heston_cos_put(
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
    dividend_yield: float = 0.0,
) -> float:
    if time_to_maturity == 0.0:
        return max(strike - spot, 0.0)

    drift = rate - dividend_yield
    c1, c2, c4 = _heston_cumulants(drift, time_to_maturity, v0, kappa, theta, xi, rho)

    # Expansion variable y = ln(S_T/K) = x + ln(S_T/S0): the range is the
    # log-return's, shifted by the log-moneyness so it is centred on the
    # distribution of y and the payoff kink at y = 0 sits where it belongs.
    x = np.log(spot / strike)
    half_width = _COS_TRUNCATION_L * np.sqrt(abs(c2) + np.sqrt(c4))
    a = x + c1 - half_width
    b = x + c1 + half_width

    # Put payoff K*(1 - e^y)^+ is supported on y < 0, i.e. on [a, min(b, 0)].
    c_range, d_range = a, min(b, 0.0)
    if c_range >= d_range:
        return 0.0

    k = np.arange(n_terms, dtype=np.float64)
    u = k * np.pi / (b - a)

    phi = _heston_char_func(
        u.astype(np.complex128), drift, time_to_maturity, v0, kappa, theta, xi, rho
    )
    re_term = np.real(phi * np.exp(1j * u * (x - a)))

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

    v_k = (2.0 / (b - a)) * (psi_k - chi_k)

    weights = np.ones(n_terms)
    weights[0] = 0.5
    price = strike * np.exp(-rate * time_to_maturity) * np.sum(weights * re_term * v_k)

    return float(price)


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
    dividend_yield: float = 0.0,
) -> float:
    if time_to_maturity == 0.0:
        return max(spot - strike, 0.0)
    put_price = _heston_cos_put(
        spot, strike, rate, time_to_maturity, v0, kappa, theta, xi, rho, n_terms, dividend_yield
    )
    return float(
        put_price
        + spot * np.exp(-dividend_yield * time_to_maturity)
        - strike * np.exp(-rate * time_to_maturity)
    )
