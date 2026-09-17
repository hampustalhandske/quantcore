"""Convex hedge sizing and option lifecycle math for listed-option positions.

Builds on the closed-form Black-Scholes pricer to answer the practical
questions a listed-options desk needs once a pricing model is in place:
how many put contracts to buy for a target level of portfolio protection,
what a long option position settles to at expiry, how its mark-to-market
P&L evolves day by day as it decays toward expiry, and which listed
expiry to roll into for a target tenor.

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
    Liabilities." *Journal of Political Economy*, 81(3), 637-654.

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put

_VALID_OPTION_TYPES = ("call", "put")
_CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class ProtectivePutSizing:
    """Result of sizing a protective put overlay against a portfolio.

    Attributes:
        contracts: Number of put contracts to buy (each covering
            `_CONTRACT_MULTIPLIER` shares of the underlying).
        premium_cost: Total premium paid for `contracts`, in dollars.
        breakeven: Per-share underlying price at expiry below which the
            combined stock+put position starts losing money net of the
            premium paid (approximately `strike - premium_per_share`).
        protection_pct_achieved: Fraction of `portfolio_value` actually
            covered by `contracts` puts. Equals the requested
            `target_protection_pct` unless the premium budget was
            insufficient, in which case it is lower -- see
            `protective_put_sizing`.
    """

    contracts: int
    premium_cost: float
    breakeven: float
    protection_pct_achieved: float


def _validate_protective_put_sizing(
    portfolio_value: float,
    spot: float,
    strike: float,
    iv: float,
    time_to_expiry: float,
    target_protection_pct: float,
    budget_pct: float,
) -> None:
    if portfolio_value <= 0.0:
        raise ValueError("portfolio_value must be strictly positive")
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if iv < 0.0:
        raise ValueError("iv must be non-negative")
    if time_to_expiry < 0.0:
        raise ValueError("time_to_expiry must be non-negative")
    if not (0.0 < target_protection_pct <= 1.0):
        raise ValueError("target_protection_pct must be in (0, 1]")
    if not (0.0 < budget_pct < 1.0):
        raise ValueError("budget_pct must be in (0, 1)")


def protective_put_sizing(
    portfolio_value: float,
    spot: float,
    strike: float,
    iv: float,
    time_to_expiry: float,
    r: float,
    target_protection_pct: float,
    budget_pct: float,
) -> ProtectivePutSizing:
    """Size a protective put overlay for a target downside-protection level.

    Standard portfolio-insurance sizing: the notional to protect is
    `portfolio_value * target_protection_pct`, converted to a number of
    underlying shares via `notional / spot`, then rounded up to whole
    contracts (each contract covers `_CONTRACT_MULTIPLIER` shares). If the
    resulting premium exceeds the dollar budget (`budget_pct *
    portfolio_value`), contracts are scaled *down* (floored) to fit the
    budget rather than raising an error -- this is a deliberate tradeoff:
    a budget-constrained hedge is still useful, just partial, and the
    caller needs the achieved protection level (not just the target) to
    decide whether the trade-off is acceptable. That achieved level is
    reported in `protection_pct_achieved`.

    Args:
        portfolio_value: Total portfolio value to protect, in dollars.
        spot: Current price of the underlying asset (S).
        strike: Put strike price (K).
        iv: Implied volatility of the underlying (sigma), annualized.
        time_to_expiry: Time to expiry in years (T).
        r: Risk-free interest rate, continuously compounded.
        target_protection_pct: Fraction of `portfolio_value` to protect,
            e.g. 0.90 protects 90% of the portfolio against a decline.
        budget_pct: Maximum premium spend as a fraction of
            `portfolio_value`, e.g. 0.02 allows up to 2% of portfolio value
            in premium.

    Returns:
        A `ProtectivePutSizing` with the number of contracts, total premium
        cost, per-share breakeven, and the protection level actually
        achieved (which may be below `target_protection_pct` if
        budget-constrained).

    Raises:
        ValueError: If any input fails validation (see field descriptions).
    """
    _validate_protective_put_sizing(
        portfolio_value, spot, strike, iv, time_to_expiry, target_protection_pct, budget_pct
    )
    return _protective_put_sizing(
        portfolio_value, spot, strike, iv, time_to_expiry, r, target_protection_pct, budget_pct
    )


def _protective_put_sizing(
    portfolio_value: float,
    spot: float,
    strike: float,
    iv: float,
    time_to_expiry: float,
    r: float,
    target_protection_pct: float,
    budget_pct: float,
) -> ProtectivePutSizing:
    notional_to_protect = portfolio_value * target_protection_pct
    shares_to_hedge = notional_to_protect / spot
    contracts = math.ceil(shares_to_hedge / _CONTRACT_MULTIPLIER)

    put_price = black_scholes_put(spot, strike, r, iv, time_to_expiry)
    premium_cost = contracts * _CONTRACT_MULTIPLIER * put_price

    budget = budget_pct * portfolio_value
    if premium_cost > budget and put_price > 0.0:
        max_affordable_contracts = math.floor(budget / (_CONTRACT_MULTIPLIER * put_price))
        contracts = max(max_affordable_contracts, 0)
        premium_cost = contracts * _CONTRACT_MULTIPLIER * put_price

    protected_notional = contracts * _CONTRACT_MULTIPLIER * spot
    protection_pct_achieved = (
        min(protected_notional / portfolio_value, 1.0) if contracts > 0 else 0.0
    )

    if contracts > 0:
        premium_per_share = premium_cost / (contracts * _CONTRACT_MULTIPLIER)
        breakeven = strike - premium_per_share
    else:
        breakeven = spot

    return ProtectivePutSizing(
        contracts=contracts,
        premium_cost=premium_cost,
        breakeven=breakeven,
        protection_pct_achieved=protection_pct_achieved,
    )


def _validate_option_type(option_type: str) -> None:
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_OPTION_TYPES}, got {option_type!r}")


def _validate_option_expiry_payoff(
    spot_at_expiry: float,
    strike: float,
    contracts: int,
    multiplier: float,
    option_type: str,
) -> None:
    if spot_at_expiry < 0.0:
        raise ValueError("spot_at_expiry must be non-negative")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if contracts < 0:
        raise ValueError("contracts must be non-negative")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be strictly positive")
    _validate_option_type(option_type)


def option_expiry_payoff(
    spot_at_expiry: float,
    strike: float,
    premium_paid: float,
    contracts: int,
    multiplier: float,
    option_type: str,
) -> float:
    """Compute the terminal net P&L of a long option position at expiry.

    Assumes a LONG option holder (buyer): net P&L is
    `contracts * multiplier * (intrinsic_value - premium_paid)`. A short
    position has the opposite sign and is not what this function computes
    -- callers holding a short position should negate the result.

    Args:
        spot_at_expiry: Underlying price at expiry.
        strike: Option strike price (K).
        premium_paid: Premium paid per share when the position was opened
            (i.e. the option price at entry, not the total premium).
        contracts: Number of contracts held (non-negative).
        multiplier: Contract multiplier (e.g. 100 shares per contract for
            standard US equity options).
        option_type: "call" or "put".

    Returns:
        Net dollar P&L at expiry: `contracts * multiplier * (intrinsic - premium_paid)`.

    Raises:
        ValueError: If any input fails validation (see field descriptions).
    """
    _validate_option_expiry_payoff(spot_at_expiry, strike, contracts, multiplier, option_type)
    return _option_expiry_payoff(
        spot_at_expiry, strike, premium_paid, contracts, multiplier, option_type
    )


def _option_expiry_payoff(
    spot_at_expiry: float,
    strike: float,
    premium_paid: float,
    contracts: int,
    multiplier: float,
    option_type: str,
) -> float:
    if option_type == "call":
        intrinsic = max(spot_at_expiry - strike, 0.0)
    else:
        intrinsic = max(strike - spot_at_expiry, 0.0)
    return float(contracts * multiplier * (intrinsic - premium_paid))


def _validate_option_position_pnl(
    entry_price: float,
    contracts: int,
    multiplier: float,
    spot_path: npt.NDArray[np.float64],
    strike: float,
    expiry: float,
    iv_path: npt.NDArray[np.float64],
    option_type: str,
) -> None:
    if contracts < 0:
        raise ValueError("contracts must be non-negative")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be strictly positive")
    if spot_path.ndim != 1 or spot_path.size == 0:
        raise ValueError("spot_path must be a non-empty 1D array")
    if iv_path.shape != spot_path.shape:
        raise ValueError("iv_path must have the same shape as spot_path")
    if np.any(spot_path <= 0.0):
        raise ValueError("spot_path values must be strictly positive")
    if np.any(iv_path < 0.0):
        raise ValueError("iv_path values must be non-negative")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if expiry < 0.0:
        raise ValueError("expiry must be non-negative")
    _validate_option_type(option_type)


def option_position_pnl(
    entry_price: float,
    contracts: int,
    multiplier: float,
    spot_path: npt.NDArray[np.float64],
    strike: float,
    expiry: float,
    iv_path: npt.NDArray[np.float64],
    r: float,
    option_type: str,
) -> npt.NDArray[np.float64]:
    """Compute daily mark-to-market P&L for a long option position over a path.

    `spot_path` and `iv_path` are assumed to be daily-frequency arrays of
    consecutive trading days from entry (index 0) to expiry (last index),
    inclusive. Time-to-expiry is assumed to decay *linearly* from `expiry`
    down to 0 over `len(spot_path) - 1` steps (i.e. calendar/trading-day
    spacing is treated as uniform) -- this is an approximation for real
    calendars with weekends/holidays but is standard for a daily P&L
    series. Each day's option value is repriced independently via
    `black_scholes_call`/`black_scholes_put` at that day's spot, IV, and
    time-to-expiry; this is a per-day repricing (not a recursive/stateful
    simulation), so it is implemented with a vectorized/list-comprehension
    NumPy loop rather than Numba, consistent with how Greeks are computed
    elsewhere in this package via repeated SciPy-backed calls. At the final
    day (time-to-expiry == 0), `black_scholes_call`/`black_scholes_put`
    already collapse to intrinsic value, so no special-casing is needed for
    expiry-day P&L.

    Args:
        entry_price: Price paid per share when the position was opened.
        contracts: Number of contracts held (non-negative).
        multiplier: Contract multiplier (e.g. 100).
        spot_path: Daily underlying prices from entry to expiry, shape (n,).
        strike: Option strike price (K).
        expiry: Time to expiry in years at entry (T), i.e. at `spot_path[0]`.
        iv_path: Daily implied volatilities, same shape as `spot_path`.
        r: Risk-free interest rate, continuously compounded.
        option_type: "call" or "put".

    Returns:
        Array of shape (n,) with `contracts * multiplier * (option_value[i] - entry_price)`
        for each day i.

    Raises:
        ValueError: If any input fails validation (see field descriptions).
    """
    _validate_option_position_pnl(
        entry_price, contracts, multiplier, spot_path, strike, expiry, iv_path, option_type
    )
    return _option_position_pnl(
        entry_price, contracts, multiplier, spot_path, strike, expiry, iv_path, r, option_type
    )


def _option_position_pnl(
    entry_price: float,
    contracts: int,
    multiplier: float,
    spot_path: npt.NDArray[np.float64],
    strike: float,
    expiry: float,
    iv_path: npt.NDArray[np.float64],
    r: float,
    option_type: str,
) -> npt.NDArray[np.float64]:
    n = spot_path.size
    steps = n - 1
    time_to_expiry = np.linspace(expiry, 0.0, n) if steps > 0 else np.array([0.0], dtype=np.float64)
    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    option_values = np.array(
        [
            pricer(float(spot_path[i]), strike, r, float(iv_path[i]), float(time_to_expiry[i]))
            for i in range(n)
        ],
        dtype=np.float64,
    )
    return np.asarray(contracts * multiplier * (option_values - entry_price), dtype=np.float64)


def _validate_option_roll_schedule(
    target_tenor_days: int,
    listed_expiries: npt.NDArray[np.object_] | Sequence[date],
) -> None:
    if target_tenor_days <= 0:
        raise ValueError("target_tenor_days must be strictly positive")
    if len(listed_expiries) == 0:
        raise ValueError("listed_expiries must be non-empty")


def option_roll_schedule(
    current_date: date,
    target_tenor_days: int,
    listed_expiries: npt.NDArray[np.object_] | Sequence[date],
) -> date:
    """Select the listed expiry closest to a target tenor from today.

    Real listed options only trade on a fixed set of exchange-listed
    expiry dates (e.g. weekly/monthly cycles), not on arbitrary calendar
    dates. This selects, from the given calendar of tradeable expiries,
    the one nearest to `current_date + target_tenor_days` by absolute
    day difference (ties broken by whichever date is returned first by
    Python's stable `min`, i.e. the earlier one in `listed_expiries`).

    Args:
        current_date: The date the roll decision is made from.
        target_tenor_days: Desired tenor in calendar days from `current_date`
            (e.g. 30, 60, 90). Must be strictly positive.
        listed_expiries: Non-empty collection of actually-listed expiry
            dates to choose from.

    Returns:
        The element of `listed_expiries` closest to
        `current_date + timedelta(days=target_tenor_days)`.

    Raises:
        ValueError: If `target_tenor_days <= 0` or `listed_expiries` is empty.
    """
    _validate_option_roll_schedule(target_tenor_days, listed_expiries)
    return _option_roll_schedule(current_date, target_tenor_days, listed_expiries)


def _option_roll_schedule(
    current_date: date,
    target_tenor_days: int,
    listed_expiries: npt.NDArray[np.object_] | Sequence[date],
) -> date:
    from datetime import timedelta

    target_date = current_date + timedelta(days=target_tenor_days)
    return min(listed_expiries, key=lambda d: abs((d - target_date).days))
