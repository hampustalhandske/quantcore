"""Rate conversions shared by the conformance targets and their tests."""


def annual_rate_to_per_period(rate_annual: float, periods_per_year: int) -> float:
    """De-annualise a rate geometrically to a per-period rate.

    Uses the convention of ``empyrical``/``quantstats``: ``(1 + rate_annual) ** (1 / p) - 1``, so
    that compounding the per-period rate ``p`` times reproduces ``rate_annual`` exactly.

    Args:
        rate_annual: Annualised rate, e.g. ``0.02`` for 2%.
        periods_per_year: Number of periods per year, e.g. ``252`` for daily data.

    Returns:
        The equivalent per-period rate.
    """
    return float((1.0 + rate_annual) ** (1.0 / periods_per_year) - 1.0)
