"""The quantcore `pricing` surface beyond the Black-Scholes-Merton price, delta and vega.

Gamma, theta and rho, `implied_volatility` (Brent on the BSM price), and option lifecycle and
sizing helpers. Each is registered as a
`ConformanceTarget` pairing a quantcore adapter (a thin call into `quantcore.pricing`) with an
independent oracle (the BSM closed form or a hand-derived formula).
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
from scipy.optimize import brentq

import quantcore.pricing as _qc_pricing
from conformance.compare import (
    ArrayComparator,
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
