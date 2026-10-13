"""The quantcore `pricing` surface beyond the Black-Scholes-Merton price, delta and vega.

Gamma, theta and rho, `implied_volatility` (Brent on the BSM price), Heston COS pricing, Monte
Carlo pricing, and option lifecycle and sizing helpers. Each is registered as a
`ConformanceTarget` pairing a quantcore adapter (a thin call into `quantcore.pricing`) with an
independent oracle (the BSM closed form, QuantLib's Heston engine, or a hand-derived formula).
Importing this module registers all of them.

Conventions the oracles share with quantcore: theta is per calendar day (the annualised closed
form over 365), rho is per unit of the rate (not per percentage point), and every P&L or payoff
is that of a long holder, so a short position's is its negation.
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import QuantLib as _QuantLib
from scipy.optimize import brentq

import quantcore.pricing as _qc_pricing
from conformance.compare import (
    ArrayComparator,
    Comparator,
    ConformanceTarget,
    ScalarComparator,
    Tolerance,
    TupleComparator,
)
from conformance.targets.pricing import OptionRight, _d1_d2, _norm_cdf, _norm_pdf

__all__: list[str] = []

type FloatArray = npt.NDArray[np.float64]

# bs_gamma


def _bs_gamma_reference(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    """BSM closed-form gamma for `t > 0`; the oracle for `bs_gamma`."""
    d1, _ = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    return math.exp(-q * t) * _norm_pdf(d1) / (s * sigma * math.sqrt(t))


def _bs_gamma_quantcore(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    """`quantcore.pricing.bs_gamma` adapter, `t > 0` only."""
    return float(_qc_pricing.bs_gamma(s, k, r, sigma, t, q))


_BS_GAMMA = ConformanceTarget(
    name="bs_gamma",
    tier=1,
    oracle="Black-Scholes-Merton closed-form gamma on math.erf; second oracle "
    "py_vollib.black_scholes_merton.greeks.analytical.gamma",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_bs_gamma_quantcore,
    reference=_bs_gamma_reference,
)


# bs_theta


def _bs_theta_reference(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """BSM closed-form theta per calendar day for `t > 0`; the oracle for `bs_theta`."""
    d1, d2 = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    rate_discount = math.exp(-r * t)
    div_discount = math.exp(-q * t)
    common = -s * div_discount * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(t))
    if right == "call":
        theta_annual = (
            common - r * k * rate_discount * _norm_cdf(d2) + q * s * div_discount * _norm_cdf(d1)
        )
    else:
        theta_annual = (
            common + r * k * rate_discount * _norm_cdf(-d2) - q * s * div_discount * _norm_cdf(-d1)
        )
    return theta_annual / 365.0


def _bs_theta_quantcore(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """`quantcore.pricing.bs_theta` adapter, `t > 0` only."""
    return float(_qc_pricing.bs_theta(s, k, r, sigma, t, right, q))


_BS_THETA = ConformanceTarget(
    name="bs_theta",
    tier=1,
    oracle="Black-Scholes-Merton closed-form theta per calendar day on math.erf; second oracle "
    "py_vollib.black_scholes_merton.greeks.analytical.theta (per calendar day, same convention "
    "as quantcore -- no rescaling needed)",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_bs_theta_quantcore,
    reference=_bs_theta_reference,
)


# bs_rho


def _bs_rho_reference(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """BSM closed-form rho for `t > 0`, per unit of `r`; the oracle for `bs_rho`."""
    _, d2 = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    discount = math.exp(-r * t)
    if right == "call":
        return k * t * discount * _norm_cdf(d2)
    return -k * t * discount * _norm_cdf(-d2)


def _bs_rho_quantcore(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """`quantcore.pricing.bs_rho` adapter, `t > 0` only."""
    return float(_qc_pricing.bs_rho(s, k, r, sigma, t, right, q))


_BS_RHO = ConformanceTarget(
    name="bs_rho",
    tier=1,
    oracle="Black-Scholes-Merton closed-form rho per unit rate move on math.erf; second oracle "
    "py_vollib.black_scholes_merton.greeks.analytical.rho "
    "(per 1 PERCENTAGE POINT of rate, i.e. quantcore's value / 100 -- multiply py_vollib's "
    "result by 100 to compare)",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_bs_rho_quantcore,
    reference=_bs_rho_reference,
)


# implied_volatility

_IV_LOWER_BOUND = 1e-9
_IV_UPPER_BOUND = 10.0
# Same threshold quantcore documents (`_IV_MIN_RELATIVE_VEGA`): below this vega relative to
# spot the price is flat in sigma and a root-finder's answer is not identifiable.
_IV_MIN_RELATIVE_VEGA = 1e-8


def _bsm_price_closed_form(
    s: float, k: float, t: float, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """BSM closed-form price for `t > 0`, the pricing step shared by this module's oracles."""
    d1, d2 = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    if right == "call":
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def _implied_volatility_reference(
    market_price: float, s: float, k: float, t: float, *, r: float, q: float, right: OptionRight
) -> float:
    """Brent's method on the BSM closed-form price; the oracle for `implied_volatility`."""

    def objective(sigma: float) -> float:
        return _bsm_price_closed_form(s, k, t, r, q, sigma, right) - market_price

    lo, hi = objective(_IV_LOWER_BOUND), objective(_IV_UPPER_BOUND)
    if lo * hi > 0.0:
        raise ValueError(
            "implied_volatility: no root in [1e-9, 10.0] -- market_price is outside "
            "no-arbitrage bounds for the given parameters"
        )
    sigma = float(brentq(objective, _IV_LOWER_BOUND, _IV_UPPER_BOUND, xtol=1e-8, maxiter=100))
    vega = (
        s * math.exp(-q * t) * _norm_pdf(_d1_d2(s, k, t, r=r, q=q, sigma=sigma)[0]) * math.sqrt(t)
    )
    if vega < _IV_MIN_RELATIVE_VEGA * s:
        raise ValueError(
            "implied_volatility: no reliably identifiable root -- vega is negligible "
            f"at the candidate sigma={sigma:.6g}"
        )
    return sigma


def _implied_volatility_quantcore(
    market_price: float, s: float, k: float, t: float, *, r: float, q: float, right: OptionRight
) -> float:
    """`quantcore.pricing.implied_volatility` adapter."""
    return float(_qc_pricing.implied_volatility(market_price, s, k, r, t, right, dividend_yield=q))


_IMPLIED_VOLATILITY = ConformanceTarget(
    name="implied_volatility",
    tier=1,
    oracle="Brent's method (bracket [1e-9, 10], xtol 1e-8) on the Black-Scholes-Merton closed "
    "form; second oracle py_vollib.black_scholes_merton.implied_volatility.implied_volatility",
    citation="Black, F. and Scholes, M. (1973); Brent, R.P. (1973).",
    compare=ScalarComparator(),
    quantcore=_implied_volatility_quantcore,
    reference=_implied_volatility_reference,
)


# heston_cos_call / heston_cos_put


def _heston_quantlib_price(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    right: OptionRight,
) -> float:
    """Price a European Heston option with QuantLib's `AnalyticHestonEngine`.

    `t` becomes a whole number of calendar days, floored at 1, so a tiny positive `t` never
    collapses to `0` under QuantLib's day-count rounding.
    """
    calc_date = _QuantLib.Date(1, 1, 2024)
    _QuantLib.Settings.instance().evaluationDate = calc_date
    day_count = _QuantLib.Actual365Fixed()
    maturity = calc_date + max(1, round(t * 365))
    option_type = _QuantLib.Option.Call if right == "call" else _QuantLib.Option.Put
    payoff = _QuantLib.PlainVanillaPayoff(option_type, k)
    exercise = _QuantLib.EuropeanExercise(maturity)
    option = _QuantLib.VanillaOption(payoff, exercise)
    spot_handle = _QuantLib.QuoteHandle(_QuantLib.SimpleQuote(s))
    rate_ts = _QuantLib.YieldTermStructureHandle(_QuantLib.FlatForward(calc_date, r, day_count))
    div_ts = _QuantLib.YieldTermStructureHandle(_QuantLib.FlatForward(calc_date, q, day_count))
    process = _QuantLib.HestonProcess(rate_ts, div_ts, spot_handle, v0, kappa, theta, xi, rho)
    engine = _QuantLib.AnalyticHestonEngine(_QuantLib.HestonModel(process))
    option.setPricingEngine(engine)
    return float(option.NPV())


def _heston_cos_call_reference(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """`QuantLib.AnalyticHestonEngine` adapter; the oracle behind `heston_cos_call`."""
    if t == 0.0:
        return max(s - k, 0.0)
    return _heston_quantlib_price(s, k, t, r, q, v0, kappa, theta, xi, rho, "call")


def _heston_cos_put_reference(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """`QuantLib.AnalyticHestonEngine` adapter; the oracle behind `heston_cos_put`."""
    if t == 0.0:
        return max(k - s, 0.0)
    return _heston_quantlib_price(s, k, t, r, q, v0, kappa, theta, xi, rho, "put")


def _heston_cos_call_quantcore(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """`quantcore.pricing.heston_cos_call` adapter."""
    return float(
        _qc_pricing.heston_cos_call(s, k, r, t, v0, kappa, theta, xi, rho, dividend_yield=q)
    )


def _heston_cos_put_quantcore(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """`quantcore.pricing.heston_cos_put` adapter."""
    return float(
        _qc_pricing.heston_cos_put(s, k, r, t, v0, kappa, theta, xi, rho, dividend_yield=q)
    )


_HESTON_COS_CALL = ConformanceTarget(
    name="heston_cos_call",
    tier=2,
    oracle="QuantLib AnalyticHestonEngine; Fang & Oosterlee (2008), Table 4/Table 5 published "
    "reference values (tests/fixtures/fang_oosterlee_2008.json) are checked in a separate test",
    citation="Heston, S.L. (1993); Fang, F. and Oosterlee, C.W. (2008).",
    compare=ScalarComparator(),
    quantcore=_heston_cos_call_quantcore,
    reference=_heston_cos_call_reference,
)

_HESTON_COS_PUT = ConformanceTarget(
    name="heston_cos_put",
    tier=2,
    oracle="QuantLib AnalyticHestonEngine",
    citation="Heston, S.L. (1993); Fang, F. and Oosterlee, C.W. (2008).",
    compare=ScalarComparator(),
    quantcore=_heston_cos_put_quantcore,
    reference=_heston_cos_put_reference,
)


# monte_carlo_call_price / monte_carlo_put_price

# Standard-error multiple judging a Monte Carlo estimate against the BSM closed form: k=4 keeps
# the false-positive rate for a well-behaved case below ~1e-4 while catching systematic mispricing.
_MONTE_CARLO_K_STD_ERRORS = 4.0


def _norm_cdf_tail(x: float) -> float:
    """Standard normal CDF via `math.erfc`, with full relative precision in the left tail.

    `_norm_cdf` is built on `1 + erf` and rounds to exactly `0.0` below about `x = -8.3`. A tail
    probability that gets multiplied by a squared strike and then square-rooted needs its
    leading digits, which this form keeps.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _monte_carlo_true_std_error(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    sigma: float,
    num_paths: int,
    antithetic: bool,
    right: OptionRight,
) -> float:
    """Standard deviation of quantcore's Monte Carlo price estimate under BSM dynamics.

    The estimate is `exp(-r*t)` times the plain mean of `num_paths` payoffs. quantcore steps the
    log price, which is exact for geometric Brownian motion at any step count, so one path's
    terminal price is `S(Z) = F * exp(-v^2/2 + v*Z)` with `Z ~ N(0, 1)`, `F = s * exp(r*t)` and
    `v = sigma * sqrt(t)`. With `w = +1` for a call and `-1` for a put, its payoff `g` has

        E[g]   = w * (F*N(w*d1) - k*N(w*d2))
        E[g^2] = F^2*exp(v^2)*N(w*(d1 + v)) - 2*k*F*N(w*d1) + k^2*N(w*d2)

    Without `antithetic` the paths are independent, so the sum of the payoffs has variance
    `num_paths * Var[g]`.

    With `antithetic`, quantcore draws `ceil(num_paths / 2)` sets of Brownian increments, appends
    their mirror images and keeps the first `num_paths` paths. That is `num_paths // 2`
    independent pairs `(Z, -Z)` and, when `num_paths` is odd, one more path whose mirror was cut
    off. The two payoffs of a pair are both positive only for `|Z| < b = w*d2`, and
    `S(Z) * S(-Z) = F^2 * exp(-v^2)` does not depend on `Z`, so their cross moment is closed form
    as well:

        E[g(Z)*g(-Z)] = (F^2*exp(-v^2) + k^2)*P(|Z| < b) - 2*k*F*(N(b - v) - N(-b - v))

    for `b > 0`, and `0` for `b <= 0`. The sum of the payoffs then has variance

        2 * (num_paths // 2) * (Var[g] + Cov[g(Z), g(-Z)]) + (num_paths % 2) * Var[g]

    These are equalities for the sampling scheme described; no term is bounded or approximated.

    Args:
        s: Spot price, positive.
        k: Strike, positive.
        t: Time to expiry in years, non-negative.
        r: Continuously compounded risk-free rate.
        sigma: Annualised volatility, non-negative.
        num_paths: Number of simulated paths, positive.
        antithetic: Whether the paths are drawn as mirrored pairs.
        right: `"call"` or `"put"`.

    Returns:
        The standard deviation of the price estimate. `0.0` when `sigma * sqrt(t)` is zero,
        where every path pays the same amount. The variances are differences of terms of size
        `F^2`; a difference that rounding turns negative is returned as `0.0`.
    """
    total_vol = sigma * math.sqrt(t)
    if total_vol == 0.0:
        return 0.0
    forward = s * math.exp(r * t)
    forward_sq = forward * forward
    vol_sq = total_vol * total_vol
    d1, d2 = _d1_d2(s, k, t, r=r, q=0.0, sigma=sigma)
    sign = 1.0 if right == "call" else -1.0

    mean = sign * (forward * _norm_cdf_tail(sign * d1) - k * _norm_cdf_tail(sign * d2))
    second_moment = (
        forward_sq * math.exp(vol_sq) * _norm_cdf_tail(sign * (d1 + total_vol))
        - 2.0 * k * forward * _norm_cdf_tail(sign * d1)
        + k * k * _norm_cdf_tail(sign * d2)
    )
    variance = max(second_moment - mean * mean, 0.0)

    n_pairs = num_paths // 2 if antithetic else 0
    n_unpaired = num_paths - 2 * n_pairs
    covariance = 0.0
    if n_pairs > 0:
        both_pay_bound = sign * d2
        cross_moment = 0.0
        if both_pay_bound > 0.0:
            prob_both_pay = math.erf(both_pay_bound / math.sqrt(2.0))
            forward_mass_both_pay = _norm_cdf_tail(both_pay_bound - total_vol) - _norm_cdf_tail(
                -both_pay_bound - total_vol
            )
            cross_moment = (
                forward_sq * math.exp(-vol_sq) + k * k
            ) * prob_both_pay - 2.0 * k * forward * forward_mass_both_pay
        covariance = cross_moment - mean * mean

    variance_of_sum = 2.0 * n_pairs * (variance + covariance) + n_unpaired * variance
    return math.exp(-r * t) * math.sqrt(max(variance_of_sum, 0.0)) / num_paths


@dataclass(frozen=True)
class _MonteCarloComparator:
    """`Comparator` for a Monte Carlo `(price, std_error)` against `(closed_form, true_std_error)`.

    Checks `|price - closed_form| <= tol.atol + k * max(std_error, true_std_error)` with
    `k = _MONTE_CARLO_K_STD_ERRORS`; `tol.rtol` is unused. `std_error` is the sample estimate the
    pricer reports. `true_std_error` is the standard deviation the estimator actually has, which
    the oracle returns: for `num_paths` payoffs `g` drawn as `num_paths // 2` mirrored pairs
    `(Z, -Z)` plus one unpaired path when `num_paths` is odd (no pairs without antithetic
    variates),

        true_std_error = exp(-r*t) / num_paths
            * sqrt(2 * (num_paths // 2) * (Var[g] + Cov[g(Z), g(-Z)]) + (num_paths % 2) * Var[g])

    The moments are exact closed forms under geometric Brownian motion, the covariance included,
    because the product of a path's terminal price and its mirror's is a constant;
    `_monte_carlo_true_std_error` gives them. Nothing in the band is an upper or lower bound.

    Each standard error covers a case the other cannot:

    - The reported one is computed from the same paths as the price. When no path finishes in
      the money every payoff is zero and the pricer returns price `0.0` with standard error
      `0.0`, so a band built on it alone has no width while the closed-form price is positive.
      The true standard error does not collapse. With `p` the chance that one path pays,
      Cauchy-Schwarz gives `E[g]^2 <= p * E[g^2]`, and from it
      `closed_form <= true_std_error * sqrt(num_paths * p / (1 - 2*p))` for an out-of-the-money
      option priced with antithetic variates. A `(0.0, 0.0)` estimate is therefore out of band
      only if `num_paths * p > 16 * (1 - 2*p)`, and then the chance that no path pays is below
      `(1 + 32/num_paths)^(-num_paths/2)`, about `1e-7` at 20,000 paths. (In the money, one
      path of every mirrored pair pays, so the case does not arise. Without antithetic variates
      the same holds with `1 - p` for `1 - 2*p` and `(1 + 16/num_paths)^(-num_paths)`.)
    - The true one gives the spread of the estimate but not its shape. When `j` paths pay and
      `j` is small, the estimate is far from normal and can exceed the closed form by many true
      standard errors. It cannot exceed `sqrt(j)` reported standard errors, though, because the
      sum of `j` payoffs is at most `sqrt(j)` times the root of the sum of their squares; the
      reported standard error covers the overshoot while fewer than 16 paths pay.

    Where many paths pay, the two agree to within the sampling noise of a standard deviation
    (about 1 percent at 20,000 paths), so the band is `k` true standard errors and a bias larger
    than that is rejected however small a standard error the pricer reports.
    """

    def is_finite(self, value: tuple[float, float]) -> bool:
        """Return whether both the price and the standard error are finite."""
        price, std_error = value
        return math.isfinite(price) and math.isfinite(std_error)

    def max_errors(
        self, actual: tuple[float, float], expected: tuple[float, float]
    ) -> tuple[float, float]:
        """Return `(abs_error, rel_error)` of the price component only."""
        actual_price, _ = actual
        expected_price, _ = expected
        abs_error = abs(actual_price - expected_price)
        rel_error = abs_error / max(abs(expected_price), 1e-300)
        return abs_error, rel_error

    def assert_close(
        self, actual: tuple[float, float], expected: tuple[float, float], tol: Tolerance
    ) -> None:
        """Raise `AssertionError` unless the price is within `k` std. errors of the closed form.

        The standard error used is the larger of the reported one (`actual[1]`) and the true one
        (`expected[1]`).
        """
        actual_price, actual_std_error = actual
        expected_price, true_std_error = expected
        allowed = tol.atol + _MONTE_CARLO_K_STD_ERRORS * max(actual_std_error, true_std_error)
        if abs(actual_price - expected_price) > allowed:
            raise AssertionError(
                f"actual={actual!r} expected={expected!r} "
                f"k={_MONTE_CARLO_K_STD_ERRORS} allowed_gap={allowed!r}"
            )


def _monte_carlo_call_price_reference(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    """BSM closed-form call price and the standard deviation of a Monte Carlo estimate of it.

    `num_steps` and `seed` only match the adapter's signature: neither changes the distribution
    of the estimate.
    """
    true_std_error = _monte_carlo_true_std_error(
        s, k, t, r=r, sigma=sigma, num_paths=num_paths, antithetic=antithetic, right="call"
    )
    return _bsm_price_closed_form(s, k, t, r, 0.0, sigma, "call"), true_std_error


def _monte_carlo_put_price_reference(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    """BSM closed-form put price and the standard deviation of a Monte Carlo estimate of it.

    `num_steps` and `seed` only match the adapter's signature: neither changes the distribution
    of the estimate.
    """
    true_std_error = _monte_carlo_true_std_error(
        s, k, t, r=r, sigma=sigma, num_paths=num_paths, antithetic=antithetic, right="put"
    )
    return _bsm_price_closed_form(s, k, t, r, 0.0, sigma, "put"), true_std_error


def _monte_carlo_call_price_quantcore(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    """`quantcore.pricing.monte_carlo_call_price` adapter."""
    price, std_error = _qc_pricing.monte_carlo_call_price(
        s, k, r, sigma, t, num_paths, num_steps, seed, antithetic
    )
    return float(price), float(std_error)


def _monte_carlo_put_price_quantcore(
    s: float,
    k: float,
    t: float,
    *,
    r: float,
    sigma: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    """`quantcore.pricing.monte_carlo_put_price` adapter."""
    price, std_error = _qc_pricing.monte_carlo_put_price(
        s, k, r, sigma, t, num_paths, num_steps, seed, antithetic
    )
    return float(price), float(std_error)


_MONTE_CARLO_COMPARATOR: Comparator[tuple[float, float]] = _MonteCarloComparator()

_MONTE_CARLO_CALL_PRICE = ConformanceTarget(
    name="monte_carlo_call_price",
    tier=2,
    oracle="Black-Scholes-Merton closed form "
    "(same GBM dynamics), judged within k=4 times the larger of the reported standard error and "
    "the estimator's closed-form standard deviation",
    citation="Boyle, P. (1977); Black, F. and Scholes, M. (1973).",
    compare=_MONTE_CARLO_COMPARATOR,
    quantcore=_monte_carlo_call_price_quantcore,
    reference=_monte_carlo_call_price_reference,
)

_MONTE_CARLO_PUT_PRICE = ConformanceTarget(
    name="monte_carlo_put_price",
    tier=2,
    oracle="Black-Scholes-Merton closed form, "
    "judged within k=4 times the larger of the reported standard error and the estimator's "
    "closed-form standard deviation",
    citation="Boyle, P. (1977); Black, F. and Scholes, M. (1973).",
    compare=_MONTE_CARLO_COMPARATOR,
    quantcore=_monte_carlo_put_price_quantcore,
    reference=_monte_carlo_put_price_reference,
)


# option_expiry_payoff


def _option_expiry_payoff_reference(
    spot_at_expiry: float,
    strike: float,
    premium_paid: float,
    contracts: int,
    multiplier: float,
    right: OptionRight,
) -> float:
    """Hand formula behind `option_expiry_payoff`: `contracts * multiplier * (intrinsic - premium)`.

    Assumes a LONG holder; a short position has the exactly opposite P&L, so negate the result.
    """
    intrinsic = (
        max(spot_at_expiry - strike, 0.0) if right == "call" else max(strike - spot_at_expiry, 0.0)
    )
    return contracts * multiplier * (intrinsic - premium_paid)


def _option_expiry_payoff_quantcore(
    spot_at_expiry: float,
    strike: float,
    premium_paid: float,
    contracts: int,
    multiplier: float,
    right: OptionRight,
) -> float:
    """`quantcore.pricing.option_expiry_payoff` adapter."""
    return float(
        _qc_pricing.option_expiry_payoff(
            spot_at_expiry, strike, premium_paid, contracts, multiplier, right
        )
    )


_OPTION_EXPIRY_PAYOFF = ConformanceTarget(
    name="option_expiry_payoff",
    tier=2,
    oracle="hand-derived intrinsic-value formula",
    citation="Standard listed-option settlement convention; see "
    "quantcore.pricing.options_positions.",
    compare=ScalarComparator(),
    quantcore=_option_expiry_payoff_quantcore,
    reference=_option_expiry_payoff_reference,
)


# option_position_pnl


def _option_position_pnl_reference(
    entry_price: float,
    contracts: int,
    multiplier: float,
    spot_path: FloatArray,
    strike: float,
    expiry: float,
    iv_path: FloatArray,
    *,
    r: float,
    right: OptionRight,
) -> FloatArray:
    """Daily BSM repricing along a path; the oracle for `option_position_pnl`.

    `spot_path`/`iv_path` run from entry (index 0) to expiry (last index) inclusive;
    time-to-expiry is assumed to decay linearly from `expiry` to `0` across those points.
    """
    n = spot_path.size
    steps = n - 1
    time_to_expiry = np.linspace(expiry, 0.0, n) if steps > 0 else np.array([0.0])
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        t_i = float(time_to_expiry[i])
        if t_i == 0.0:
            intrinsic = (
                max(float(spot_path[i]) - strike, 0.0)
                if right == "call"
                else max(strike - float(spot_path[i]), 0.0)
            )
            values[i] = intrinsic
        else:
            values[i] = _bsm_price_closed_form(
                float(spot_path[i]), strike, t_i, r, 0.0, float(iv_path[i]), right
            )
    return np.asarray(contracts * multiplier * (values - entry_price), dtype=np.float64)


def _option_position_pnl_quantcore(
    entry_price: float,
    contracts: int,
    multiplier: float,
    spot_path: FloatArray,
    strike: float,
    expiry: float,
    iv_path: FloatArray,
    *,
    r: float,
    right: OptionRight,
) -> FloatArray:
    """`quantcore.pricing.option_position_pnl` adapter."""
    return np.asarray(
        _qc_pricing.option_position_pnl(
            entry_price, contracts, multiplier, spot_path, strike, expiry, iv_path, r, right
        ),
        dtype=np.float64,
    )


_OPTION_POSITION_PNL = ConformanceTarget(
    name="option_position_pnl",
    tier=2,
    oracle="hand-derived daily BSM repricing",
    citation="Black, F. and Scholes, M. (1973); see quantcore.pricing.options_positions.",
    compare=ArrayComparator(),
    quantcore=_option_position_pnl_quantcore,
    reference=_option_position_pnl_reference,
)


# option_roll_schedule


def _option_roll_schedule_reference(
    current_date: date, target_tenor_days: int, listed_expiries: tuple[date, ...]
) -> date:
    """Hand formula behind `option_roll_schedule`: nearest listed expiry to the target tenor."""
    target_date = current_date + timedelta(days=target_tenor_days)
    return min(listed_expiries, key=lambda d: abs((d - target_date).days))


def _option_roll_schedule_quantcore(
    current_date: date, target_tenor_days: int, listed_expiries: tuple[date, ...]
) -> date:
    """`quantcore.pricing.option_roll_schedule` adapter."""
    result: date = _qc_pricing.option_roll_schedule(
        current_date, target_tenor_days, list(listed_expiries)
    )
    return result


@dataclass(frozen=True)
class _DateComparator:
    """`Comparator` for a function returning a single `datetime.date`."""

    def is_finite(self, value: date) -> bool:
        """A `date` is always well-defined; always `True`."""
        return True

    def max_errors(self, actual: date, expected: date) -> tuple[float, float]:
        """Absolute error in whole days; relative error is not meaningful for a date."""
        days = abs((actual - expected).days)
        return float(days), float(days)

    def assert_close(self, actual: date, expected: date, tol: Tolerance) -> None:
        """Raise `AssertionError` unless the two dates are identical."""
        if actual != expected:
            raise AssertionError(f"actual={actual!r} expected={expected!r} tol={tol!r}")


_OPTION_ROLL_SCHEDULE = ConformanceTarget(
    name="option_roll_schedule",
    tier=2,
    oracle="hand-derived nearest-expiry selection",
    citation="Standard listed-option roll convention; see quantcore.pricing.options_positions.",
    compare=_DateComparator(),
    quantcore=_option_roll_schedule_quantcore,
    reference=_option_roll_schedule_reference,
)


# protective_put_sizing

_CONTRACT_MULTIPLIER = 100


def _protective_put_sizing_formula(
    portfolio_value: float,
    s: float,
    k: float,
    iv: float,
    t: float,
    r: float,
    target_protection_pct: float,
    budget_pct: float,
    put_price: float,
) -> tuple[float, float, float, float]:
    """Sizing arithmetic given an already-computed put price; contracts are returned as `float`."""
    notional_to_protect = portfolio_value * target_protection_pct
    shares_to_hedge = notional_to_protect / s
    contracts = math.ceil(shares_to_hedge / _CONTRACT_MULTIPLIER)

    premium_cost = contracts * _CONTRACT_MULTIPLIER * put_price
    budget = budget_pct * portfolio_value
    if premium_cost > budget and put_price > 0.0:
        contracts = max(math.floor(budget / (_CONTRACT_MULTIPLIER * put_price)), 0)
        premium_cost = contracts * _CONTRACT_MULTIPLIER * put_price

    protected_notional = contracts * _CONTRACT_MULTIPLIER * s
    protection_pct_achieved = (
        min(protected_notional / portfolio_value, 1.0) if contracts > 0 else 0.0
    )
    if contracts > 0:
        premium_per_share = premium_cost / (contracts * _CONTRACT_MULTIPLIER)
        breakeven = k - premium_per_share
    else:
        breakeven = s
    return float(contracts), premium_cost, breakeven, protection_pct_achieved


def _protective_put_sizing_reference(
    portfolio_value: float,
    s: float,
    k: float,
    iv: float,
    t: float,
    *,
    r: float,
    target_protection_pct: float,
    budget_pct: float,
) -> tuple[float, float, float, float]:
    """Hand formula behind `protective_put_sizing`, pricing the put via the closed-form BSM."""
    put_price = _bsm_price_closed_form(s, k, t, r, 0.0, iv, "put") if t > 0.0 else max(k - s, 0.0)
    return _protective_put_sizing_formula(
        portfolio_value, s, k, iv, t, r, target_protection_pct, budget_pct, put_price
    )


def _protective_put_sizing_quantcore(
    portfolio_value: float,
    s: float,
    k: float,
    iv: float,
    t: float,
    *,
    r: float,
    target_protection_pct: float,
    budget_pct: float,
) -> tuple[float, float, float, float]:
    """`quantcore.pricing.protective_put_sizing` adapter, unpacked to a plain tuple."""
    result = _qc_pricing.protective_put_sizing(
        portfolio_value, s, k, iv, t, r, target_protection_pct, budget_pct
    )
    return (
        float(result.contracts),
        result.premium_cost,
        result.breakeven,
        result.protection_pct_achieved,
    )


_PROTECTIVE_PUT_SIZING = ConformanceTarget(
    name="protective_put_sizing",
    tier=2,
    oracle="hand-derived sizing formula",
    citation="Standard portfolio-insurance sizing; see quantcore.pricing.options_positions.",
    compare=TupleComparator(
        ScalarComparator(), ScalarComparator(), ScalarComparator(), ScalarComparator()
    ),
    quantcore=_protective_put_sizing_quantcore,
    reference=_protective_put_sizing_reference,
)
