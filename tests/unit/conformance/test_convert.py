"""Tests for conformance.convert: the annual-to-per-period rate conversion."""

import pytest

# hypothesis is installed by the `oracle` extra only; skip this module without it.
pytest.importorskip("hypothesis")

from conformance.convert import annual_rate_to_per_period
from hypothesis import given, settings
from hypothesis import strategies as st


def test_annual_rate_to_per_period_hand_values_and_zero() -> None:
    """annual_rate_to_per_period matches a geometric hand computation and is 0 at 0% annual rate."""
    assert annual_rate_to_per_period(0.0, 12) == 0.0
    assert annual_rate_to_per_period(0.0, 252) == 0.0
    got = annual_rate_to_per_period(0.05, 12)
    expected = (1.05) ** (1.0 / 12.0) - 1.0
    assert got == pytest.approx(expected, abs=1e-15)


@given(
    rate_annual=st.floats(min_value=-0.99, max_value=10.0, allow_nan=False, allow_infinity=False),
    periods_per_year=st.integers(min_value=1, max_value=366),
)
@settings(max_examples=100, deadline=None)
def test_annual_rate_to_per_period_compounding_round_trips(rate_annual, periods_per_year) -> None:
    """Compounding the per-period rate periods_per_year times recovers the annual rate to 1e-12."""
    per_period = annual_rate_to_per_period(rate_annual, periods_per_year)
    recovered = (1.0 + per_period) ** periods_per_year - 1.0
    assert recovered == pytest.approx(rate_annual, abs=1e-12, rel=1e-12)
