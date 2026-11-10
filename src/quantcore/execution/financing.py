"""Carry/financing cost formulas for holding levered or short positions.

    Simple annual carry (day-count accrual):
      cost = notional * rate_annual * (days_held / 365.0)

    Futures roll yield (implied annualized carry from the term structure):
      roll_yield = ((near_price - far_price) / far_price) * (365.0 / days_to_roll)
      positive => backwardation (near > far), negative => contango (near < far)

References:
    These are standard day-count carry/accrual conventions (simple interest,
    actual/365) with no single canonical paper; see docs/REFERENCES.md.

Every input must be finite; a cost that overflows float64 raises
`quantcore.UndefinedResultError` (a `ValueError`).
"""

from __future__ import annotations

import math

from quantcore.errors import _finite_result


def _validate_simple_annual_carry(notional: float, rate_annual: float, days_held: int) -> None:
    for name, value in (
        ("notional", notional),
        ("rate_annual", rate_annual),
        ("days_held", days_held),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if notional < 0.0:
        raise ValueError("notional must be non-negative")
    if rate_annual < 0.0:
        raise ValueError("rate_annual must be non-negative")
    if days_held < 0:
        raise ValueError("days_held must be non-negative")


def _simple_annual_carry(notional: float, rate_annual: float, days_held: int) -> float:
    return notional * rate_annual * (days_held / 365.0)


def short_borrow_cost(notional: float, borrow_rate_annual: float, days_held: int) -> float:
    """Cost of borrowing shares to hold a short position for `days_held` days.

    Args:
        notional: Absolute notional value of the shorted position (must be
            non-negative).
        borrow_rate_annual: Annualized stock borrow rate, e.g. 0.02 for 2%
            (must be non-negative).
        days_held: Number of calendar days the short is held (must be
            non-negative).

    Returns:
        Borrow cost in the same currency units as `notional`.

    Raises:
        ValueError: An input is negative or not finite.
        UndefinedResultError: The cost overflows float64.
    """
    _validate_simple_annual_carry(notional, borrow_rate_annual, days_held)
    return _finite_result(
        _simple_annual_carry(notional, borrow_rate_annual, days_held), "short borrow cost"
    )


def margin_financing_cost(notional: float, financing_rate_annual: float, days_held: int) -> float:
    """Interest charge for financing a levered/margin position for `days_held` days.

    Uses the same day-count carry arithmetic as `short_borrow_cost` but is
    kept as a distinct function since it prices a conceptually different
    cost (leverage/margin interest on a financed position, not a stock
    borrow fee on a short) that downstream callers attribute separately.

    Args:
        notional: Absolute notional value financed on margin (must be
            non-negative).
        financing_rate_annual: Annualized financing/margin interest rate,
            e.g. 0.05 for 5% (must be non-negative).
        days_held: Number of calendar days the position is financed (must be
            non-negative).

    Returns:
        Financing cost in the same currency units as `notional`.

    Raises:
        ValueError: An input is negative or not finite.
        UndefinedResultError: The cost overflows float64.
    """
    _validate_simple_annual_carry(notional, financing_rate_annual, days_held)
    return _finite_result(
        _simple_annual_carry(notional, financing_rate_annual, days_held), "margin financing cost"
    )
