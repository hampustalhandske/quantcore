"""Market impact cost formulas.

    Square-root market impact law (Almgren-Chriss family):
      impact_bps = impact_coefficient * daily_vol * sqrt(|trade_size| / adv) * 10000

    impact scales with daily volatility (return units) times the square root
    of the participation rate (trade size over average daily volume, ADV),
    scaled to basis points by an empirical, market/asset-class-specific
    coefficient supplied by the caller.

References:
    Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio
    Transactions," Journal of Risk, 3, 5-40. Gatheral, J. (2010),
    "No-Dynamic-Arbitrage and Market Impact," Quantitative Finance, 10(7),
    749-759. See docs/REFERENCES.md.
"""

from __future__ import annotations

import math


def _validate_square_root_market_impact(
    trade_size: float,
    adv: float,
    daily_vol: float,
    impact_coefficient: float,
) -> None:
    if adv <= 0.0:
        raise ValueError("adv must be strictly positive")
    if daily_vol < 0.0:
        raise ValueError("daily_vol must be non-negative")
    if impact_coefficient < 0.0:
        raise ValueError("impact_coefficient must be non-negative")


def square_root_market_impact(
    trade_size: float,
    adv: float,
    daily_vol: float,
    impact_coefficient: float = 1.0,
) -> float:
    """Temporary market impact of a trade under the square-root impact law.

    The result is always a non-negative magnitude in basis points; it does
    not carry the sign of the trade direction, matching this repo's
    convention elsewhere (e.g. `value_at_risk`) of returning magnitudes for
    the caller to sign as needed.

    Args:
        trade_size: Signed or unsigned size of the trade, in the same units
            as `adv`. Only `abs(trade_size)` affects the result.
        adv: Average daily volume, in the same units as `trade_size` (must
            be strictly positive).
        daily_vol: Daily return volatility, e.g. 0.02 for 2% (must be
            non-negative).
        impact_coefficient: Empirical scaling constant calibrated per
            market/asset class (must be non-negative). Defaults to 1.0.

    Returns:
        Estimated temporary market impact magnitude, in basis points.
    """
    _validate_square_root_market_impact(trade_size, adv, daily_vol, impact_coefficient)
    return _square_root_market_impact(trade_size, adv, daily_vol, impact_coefficient)


def _square_root_market_impact(
    trade_size: float,
    adv: float,
    daily_vol: float,
    impact_coefficient: float,
) -> float:
    participation_rate = abs(trade_size) / adv
    return impact_coefficient * daily_vol * math.sqrt(participation_rate) * 10000.0
