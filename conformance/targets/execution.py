"""Tier-1 execution costs: square-root market impact, short-borrow cost, margin-financing cost.

`impact_bps = coefficient * daily_vol * sqrt(|trade_size| / adv) * 10_000`, an unsigned
magnitude. Borrow and financing are simple annual carry on ACT/365:
`cost = notional * rate_annual * days_held / 365`.

The three functions named in `__all__` are conformance-side input validators, not part of
quantcore's API: each checks that its arguments are finite and correctly signed, raising
`ValueError` otherwise, and then calls the matching quantcore adapter.
`conformance/test_tier1_execution.py` uses them to pin down which inputs are contract
violations, next to its record of what raw quantcore does on such inputs.
"""

import math

import quantcore.execution as qc_execution
from conformance.compare import ConformanceTarget, ScalarComparator

__all__ = [
    "margin_financing_cost",
    "short_borrow_cost",
    "square_root_market_impact",
]

_DAYS_PER_YEAR = 365.0


def _validate_finite(name: str, value: float) -> None:
    """Raise `ValueError` naming `name` unless `value` is a finite number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _validate_positive(name: str, value: float) -> None:
    """Raise `ValueError` naming `name` unless `value` is a finite number `> 0`."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _validate_non_negative(name: str, value: float) -> None:
    """Raise `ValueError` naming `name` unless `value` is a finite number `>= 0`."""
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


def _square_root_market_impact_quantcore(
    trade_size: float, adv: float, daily_vol: float, coefficient: float
) -> float:
    return float(qc_execution.square_root_market_impact(trade_size, adv, daily_vol, coefficient))


def _square_root_market_impact_reference(
    trade_size: float, adv: float, daily_vol: float, coefficient: float
) -> float:
    return coefficient * daily_vol * math.sqrt(abs(trade_size) / adv) * 10_000.0


_SQUARE_ROOT_MARKET_IMPACT = ConformanceTarget(
    name="square_root_market_impact",
    tier=1,
    oracle="hand-derived closed form",
    citation=(
        "Almgren, R. and Chriss, N. (2000), 'Optimal Execution of Portfolio Transactions.' "
        "Journal of Risk, 3, 5-40."
    ),
    compare=ScalarComparator(),
    quantcore=_square_root_market_impact_quantcore,
    reference=_square_root_market_impact_reference,
)


def square_root_market_impact(
    trade_size: float, adv: float, daily_vol: float, coefficient: float = 1.0
) -> float:
    """Temporary market impact of a trade under the square-root impact law, in basis points.

    ``impact_bps = coefficient * daily_vol * sqrt(|trade_size| / adv) * 10_000``. The result is
    always a non-negative magnitude; it does not carry the sign of the trade direction, the
    caller signs it as needed.

    Args:
        trade_size: Signed or unsigned size of the trade, in the same units as `adv`. Only
            `abs(trade_size)` affects the result.
        adv: Average daily volume, in the same units as `trade_size`. Must be positive.
        daily_vol: Daily return volatility, e.g. `0.02` for 2%. Must be non-negative.
        coefficient: Empirical scaling constant calibrated per market/asset class. Must be
            non-negative.

    Returns:
        Estimated temporary market impact magnitude, in basis points.

    Raises:
        ValueError: `trade_size` is not finite, `adv` is not positive and finite, `daily_vol` is
            negative or not finite, or `coefficient` is negative or not finite.

    References:
        Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio Transactions."
        Journal of Risk, 3, 5-40.
    """
    _validate_finite("trade_size", trade_size)
    _validate_positive("adv", adv)
    _validate_non_negative("daily_vol", daily_vol)
    _validate_non_negative("coefficient", coefficient)
    return _square_root_market_impact_quantcore(trade_size, adv, daily_vol, coefficient)


def _short_borrow_cost_quantcore(
    notional: float, borrow_rate_annual: float, days_held: float
) -> float:
    return float(qc_execution.short_borrow_cost(notional, borrow_rate_annual, days_held))


def _short_borrow_cost_reference(
    notional: float, borrow_rate_annual: float, days_held: float
) -> float:
    return notional * borrow_rate_annual * (days_held / _DAYS_PER_YEAR)


_SHORT_BORROW_COST = ConformanceTarget(
    name="short_borrow_cost",
    tier=1,
    oracle="hand-derived closed form",
    citation="ACT/365 simple-interest day-count accrual convention.",
    compare=ScalarComparator(),
    quantcore=_short_borrow_cost_quantcore,
    reference=_short_borrow_cost_reference,
)


def short_borrow_cost(notional: float, borrow_rate_annual: float, days_held: float) -> float:
    """Cost of borrowing shares to hold a short position for `days_held` days, ACT/365.

    ``cost = notional * borrow_rate_annual * (days_held / 365)``.

    Args:
        notional: Absolute notional value of the shorted position. Must be non-negative.
        borrow_rate_annual: Annualised stock borrow rate, e.g. `0.02` for 2%. Must be
            non-negative.
        days_held: Number of calendar days the short is held. Must be non-negative.

    Returns:
        Borrow cost, in the same currency units as `notional`.

    Raises:
        ValueError: `notional`, `borrow_rate_annual` or `days_held` is negative or not finite.

    References:
        ACT/365 is a standard day-count accrual convention; see ISDA day-count fraction
        definitions.
    """
    _validate_non_negative("notional", notional)
    _validate_non_negative("borrow_rate_annual", borrow_rate_annual)
    _validate_non_negative("days_held", days_held)
    return _short_borrow_cost_quantcore(notional, borrow_rate_annual, days_held)


def _margin_financing_cost_quantcore(
    notional: float, financing_rate_annual: float, days_held: float
) -> float:
    return float(qc_execution.margin_financing_cost(notional, financing_rate_annual, days_held))


def _margin_financing_cost_reference(
    notional: float, financing_rate_annual: float, days_held: float
) -> float:
    return notional * financing_rate_annual * (days_held / _DAYS_PER_YEAR)


_MARGIN_FINANCING_COST = ConformanceTarget(
    name="margin_financing_cost",
    tier=1,
    oracle="hand-derived closed form",
    citation="ACT/365 simple-interest day-count accrual convention.",
    compare=ScalarComparator(),
    quantcore=_margin_financing_cost_quantcore,
    reference=_margin_financing_cost_reference,
)


def margin_financing_cost(notional: float, financing_rate_annual: float, days_held: float) -> float:
    """Interest charge for financing a levered/margin position for `days_held` days, ACT/365.

    ``cost = notional * financing_rate_annual * (days_held / 365)``. Uses the same day-count
    carry arithmetic as `short_borrow_cost` but prices a conceptually different cost (leverage
    interest on a financed position, not a stock borrow fee on a short), kept as a distinct
    function so downstream callers attribute the two separately.

    Args:
        notional: Absolute notional value financed on margin. Must be non-negative.
        financing_rate_annual: Annualised financing/margin interest rate, e.g. `0.05` for 5%.
            Must be non-negative.
        days_held: Number of calendar days the position is financed. Must be non-negative.

    Returns:
        Financing cost, in the same currency units as `notional`.

    Raises:
        ValueError: `notional`, `financing_rate_annual` or `days_held` is negative or not finite.

    References:
        ACT/365 is a standard day-count accrual convention; see ISDA day-count fraction
        definitions.
    """
    _validate_non_negative("notional", notional)
    _validate_non_negative("financing_rate_annual", financing_rate_annual)
    _validate_non_negative("days_held", days_held)
    return _margin_financing_cost_quantcore(notional, financing_rate_annual, days_held)
