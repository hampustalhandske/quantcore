"""Tests for the square-root market impact law."""

from __future__ import annotations

import math

import pytest

from quantcore.execution.market_impact import square_root_market_impact


class TestSquareRootMarketImpact:
    def test_known_answer(self) -> None:
        actual = square_root_market_impact(
            trade_size=10_000.0, adv=1_000_000.0, daily_vol=0.02, impact_coefficient=0.5
        )
        expected = 0.5 * 0.02 * math.sqrt(10_000.0 / 1_000_000.0) * 10000.0
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_default_impact_coefficient_is_one(self) -> None:
        actual = square_root_market_impact(trade_size=5_000.0, adv=500_000.0, daily_vol=0.01)
        expected = 1.0 * 0.01 * math.sqrt(5_000.0 / 500_000.0) * 10000.0
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_negative_trade_size_uses_magnitude(self) -> None:
        positive = square_root_market_impact(trade_size=10_000.0, adv=1_000_000.0, daily_vol=0.02)
        negative = square_root_market_impact(trade_size=-10_000.0, adv=1_000_000.0, daily_vol=0.02)
        assert negative == pytest.approx(positive, rel=1e-12)
        assert negative >= 0.0

    def test_zero_trade_size_is_zero_impact(self) -> None:
        actual = square_root_market_impact(trade_size=0.0, adv=1_000_000.0, daily_vol=0.02)
        assert actual == 0.0

    def test_increases_with_trade_size_magnitude(self) -> None:
        small = square_root_market_impact(trade_size=1_000.0, adv=1_000_000.0, daily_vol=0.02)
        large = square_root_market_impact(trade_size=10_000.0, adv=1_000_000.0, daily_vol=0.02)
        assert large > small

    def test_increases_with_daily_vol(self) -> None:
        low_vol = square_root_market_impact(trade_size=10_000.0, adv=1_000_000.0, daily_vol=0.01)
        high_vol = square_root_market_impact(trade_size=10_000.0, adv=1_000_000.0, daily_vol=0.05)
        assert high_vol > low_vol

    def test_decreases_with_adv(self) -> None:
        small_adv = square_root_market_impact(trade_size=10_000.0, adv=500_000.0, daily_vol=0.02)
        large_adv = square_root_market_impact(trade_size=10_000.0, adv=5_000_000.0, daily_vol=0.02)
        assert large_adv < small_adv

    @pytest.mark.parametrize(
        ("trade_size", "adv", "daily_vol", "impact_coefficient"),
        [
            (10_000.0, 0.0, 0.02, 1.0),
            (10_000.0, -1.0, 0.02, 1.0),
            (10_000.0, 1_000_000.0, -0.01, 1.0),
            (10_000.0, 1_000_000.0, 0.02, -0.1),
        ],
    )
    def test_invalid_inputs(
        self, trade_size: float, adv: float, daily_vol: float, impact_coefficient: float
    ) -> None:
        with pytest.raises(ValueError):
            square_root_market_impact(trade_size, adv, daily_vol, impact_coefficient)
