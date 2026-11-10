"""Tests for carry/financing cost formulas: borrow cost, margin financing."""

from __future__ import annotations

import math

import pytest

from quantcore import UndefinedResultError
from quantcore.execution.financing import (
    margin_financing_cost,
    short_borrow_cost,
)


class TestShortBorrowCost:
    def test_known_answer(self) -> None:
        actual = short_borrow_cost(notional=100_000.0, borrow_rate_annual=0.03, days_held=30)
        expected = 100_000.0 * 0.03 * (30 / 365.0)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_zero_days_is_zero_cost(self) -> None:
        assert short_borrow_cost(notional=100_000.0, borrow_rate_annual=0.03, days_held=0) == 0.0

    def test_scales_linearly_with_days_held(self) -> None:
        one_day = short_borrow_cost(notional=50_000.0, borrow_rate_annual=0.04, days_held=1)
        ten_days = short_borrow_cost(notional=50_000.0, borrow_rate_annual=0.04, days_held=10)
        assert ten_days == pytest.approx(10 * one_day, rel=1e-9)

    @pytest.mark.parametrize(
        ("notional", "borrow_rate_annual", "days_held"),
        [
            (-1.0, 0.03, 10),
            (100.0, -0.01, 10),
            (100.0, 0.03, -1),
        ],
    )
    def test_invalid_inputs(
        self, notional: float, borrow_rate_annual: float, days_held: int
    ) -> None:
        with pytest.raises(ValueError):
            short_borrow_cost(notional, borrow_rate_annual, days_held)


class TestMarginFinancingCost:
    def test_known_answer(self) -> None:
        actual = margin_financing_cost(notional=250_000.0, financing_rate_annual=0.05, days_held=90)
        expected = 250_000.0 * 0.05 * (90 / 365.0)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_matches_short_borrow_cost_for_equal_inputs(self) -> None:
        assert margin_financing_cost(100_000.0, 0.03, 30) == short_borrow_cost(100_000.0, 0.03, 30)

    @pytest.mark.parametrize(
        ("notional", "financing_rate_annual", "days_held"),
        [
            (-1.0, 0.05, 10),
            (100.0, -0.02, 10),
            (100.0, 0.05, -1),
        ],
    )
    def test_invalid_inputs(
        self, notional: float, financing_rate_annual: float, days_held: int
    ) -> None:
        with pytest.raises(ValueError):
            margin_financing_cost(notional, financing_rate_annual, days_held)


@pytest.mark.parametrize("cost", [short_borrow_cost, margin_financing_cost])
class TestUndefinedAndNonFinite:
    @pytest.mark.parametrize(
        "args", [(math.nan, 0.02, 10), (1e6, math.inf, 10), (math.inf, 0.02, 10)]
    )
    def test_non_finite_inputs_are_plain_value_errors(self, cost, args) -> None:
        with pytest.raises(ValueError, match="must be finite") as raised:
            cost(*args)
        assert not isinstance(raised.value, UndefinedResultError)

    def test_overflowing_cost_raises_undefined(self, cost) -> None:
        with pytest.raises(UndefinedResultError, match="not finite") as raised:
            cost(1e308, 1e10, 365)
        assert isinstance(raised.value, ValueError)

    def test_zero_cost_is_defined(self, cost) -> None:
        assert cost(0.0, 0.02, 10) == 0.0
