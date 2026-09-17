"""Tests for carry/financing cost formulas: borrow cost, roll yield, margin financing."""

from __future__ import annotations

import pytest

from quantcore.execution.financing import (
    futures_roll_yield,
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


class TestFuturesRollYield:
    def test_known_answer(self) -> None:
        actual = futures_roll_yield(near_price=102.0, far_price=100.0, days_to_roll=30)
        expected = ((102.0 - 100.0) / 100.0) * (365.0 / 30)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_backwardation_is_positive(self) -> None:
        actual = futures_roll_yield(near_price=105.0, far_price=100.0, days_to_roll=60)
        assert actual > 0.0

    def test_contango_is_negative(self) -> None:
        actual = futures_roll_yield(near_price=98.0, far_price=100.0, days_to_roll=60)
        assert actual < 0.0

    def test_equal_prices_is_zero(self) -> None:
        actual = futures_roll_yield(near_price=100.0, far_price=100.0, days_to_roll=30)
        assert actual == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("near_price", "far_price", "days_to_roll"),
        [
            (0.0, 100.0, 30),
            (-1.0, 100.0, 30),
            (100.0, 0.0, 30),
            (100.0, -1.0, 30),
            (100.0, 100.0, 0),
            (100.0, 100.0, -1),
        ],
    )
    def test_invalid_inputs(self, near_price: float, far_price: float, days_to_roll: int) -> None:
        with pytest.raises(ValueError):
            futures_roll_yield(near_price, far_price, days_to_roll)
