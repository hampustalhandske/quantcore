"""Tests for protective-put sizing and the option roll schedule.

Tolerances: dollar-value assertions use `pytest.approx(..., rel=1e-6)` unless noted.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantcore.pricing.black_scholes import black_scholes_put
from quantcore.pricing.options_positions import (
    ProtectivePutSizing,
    option_roll_schedule,
    protective_put_sizing,
)


class TestProtectivePutSizing:
    def test_hand_verifiable_sizing_within_budget(self) -> None:
        result = protective_put_sizing(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.2,
            time_to_expiry=0.5,
            r=0.03,
            target_protection_pct=0.90,
            budget_pct=0.20,
        )
        notional_to_protect = 1_000_000.0 * 0.90
        shares_to_hedge = notional_to_protect / 100.0
        expected_contracts = int(np.ceil(shares_to_hedge / 100))
        assert result.contracts == expected_contracts

        put_price = black_scholes_put(100.0, 95.0, 0.03, 0.2, 0.5)
        expected_premium = expected_contracts * 100 * put_price
        assert result.premium_cost == pytest.approx(expected_premium, rel=1e-6)
        assert result.protection_pct_achieved == pytest.approx(0.9, rel=1e-2)

    def test_returns_frozen_dataclass(self) -> None:
        result = protective_put_sizing(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.2,
            time_to_expiry=0.5,
            r=0.03,
            target_protection_pct=0.90,
            budget_pct=0.20,
        )
        assert isinstance(result, ProtectivePutSizing)
        with pytest.raises(AttributeError):
            result.contracts = 999  # type: ignore[misc]

    def test_budget_constrained_scales_contracts_down(self) -> None:
        unconstrained = protective_put_sizing(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.5,
            time_to_expiry=2.0,
            r=0.03,
            target_protection_pct=1.0,
            budget_pct=0.99,
        )
        constrained = protective_put_sizing(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.5,
            time_to_expiry=2.0,
            r=0.03,
            target_protection_pct=1.0,
            budget_pct=0.01,
        )
        assert constrained.contracts < unconstrained.contracts
        assert constrained.protection_pct_achieved < 1.0
        budget = 0.01 * 1_000_000.0
        assert constrained.premium_cost <= budget + 1e-6

    def test_breakeven_below_strike_for_long_put_hedge(self) -> None:
        result = protective_put_sizing(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.2,
            time_to_expiry=0.5,
            r=0.03,
            target_protection_pct=0.90,
            budget_pct=0.20,
        )
        assert result.breakeven < 95.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(portfolio_value=0.0),
            dict(spot=0.0),
            dict(strike=-1.0),
            dict(iv=-0.1),
            dict(time_to_expiry=-1.0),
            dict(target_protection_pct=0.0),
            dict(target_protection_pct=1.5),
            dict(budget_pct=0.0),
            dict(budget_pct=1.0),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, float]) -> None:
        base = dict(
            portfolio_value=1_000_000.0,
            spot=100.0,
            strike=95.0,
            iv=0.2,
            time_to_expiry=0.5,
            r=0.03,
            target_protection_pct=0.90,
            budget_pct=0.20,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            protective_put_sizing(**base)  # type: ignore[arg-type]


class TestOptionRollSchedule:
    def test_selects_nearest_listed_expiry(self) -> None:
        current = date(2026, 1, 1)
        listed = [
            date(2026, 1, 15),
            date(2026, 2, 20),
            date(2026, 3, 20),
            date(2026, 4, 17),
        ]
        result = option_roll_schedule(current, target_tenor_days=60, listed_expiries=listed)
        assert result == date(2026, 2, 20)

    def test_exact_match_returned(self) -> None:
        current = date(2026, 1, 1)
        target = date(2026, 1, 31)
        listed = [date(2026, 1, 10), target, date(2026, 2, 15)]
        result = option_roll_schedule(current, target_tenor_days=30, listed_expiries=listed)
        assert result == target

    def test_empty_calendar_raises(self) -> None:
        with pytest.raises(ValueError):
            option_roll_schedule(date(2026, 1, 1), target_tenor_days=30, listed_expiries=[])

    def test_non_positive_tenor_raises(self) -> None:
        with pytest.raises(ValueError):
            option_roll_schedule(
                date(2026, 1, 1),
                target_tenor_days=0,
                listed_expiries=[date(2026, 2, 1)],
            )
