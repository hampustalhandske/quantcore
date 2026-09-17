"""Tests for convex hedge sizing and option lifecycle math.

Tolerances: dollar-value assertions use `pytest.approx(..., rel=1e-6)` unless
noted; qualitative decay checks use strict monotonicity since Black-Scholes
theta has a definite sign for OTM/ATM options with no dividends.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.options_positions import (
    ProtectivePutSizing,
    option_expiry_payoff,
    option_position_pnl,
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


class TestOptionExpiryPayoff:
    def test_deep_itm_put_equals_intrinsic_minus_premium(self) -> None:
        pnl = option_expiry_payoff(
            spot_at_expiry=50.0,
            strike=100.0,
            premium_paid=2.0,
            contracts=3,
            multiplier=100.0,
            option_type="put",
        )
        intrinsic = 100.0 - 50.0
        expected = 3 * 100.0 * (intrinsic - 2.0)
        assert pnl == pytest.approx(expected)

    def test_deep_itm_call_equals_intrinsic_minus_premium(self) -> None:
        pnl = option_expiry_payoff(
            spot_at_expiry=150.0,
            strike=100.0,
            premium_paid=5.0,
            contracts=2,
            multiplier=100.0,
            option_type="call",
        )
        intrinsic = 150.0 - 100.0
        expected = 2 * 100.0 * (intrinsic - 5.0)
        assert pnl == pytest.approx(expected)

    def test_otm_option_loses_full_premium(self) -> None:
        pnl = option_expiry_payoff(
            spot_at_expiry=100.0,
            strike=150.0,
            premium_paid=3.0,
            contracts=1,
            multiplier=100.0,
            option_type="call",
        )
        assert pnl == pytest.approx(-300.0)

    def test_zero_contracts_yields_zero_pnl(self) -> None:
        pnl = option_expiry_payoff(
            spot_at_expiry=100.0,
            strike=100.0,
            premium_paid=3.0,
            contracts=0,
            multiplier=100.0,
            option_type="call",
        )
        assert pnl == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(spot_at_expiry=-1.0),
            dict(strike=0.0),
            dict(contracts=-1),
            dict(multiplier=0.0),
            dict(option_type="straddle"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, float]) -> None:
        base = dict(
            spot_at_expiry=100.0,
            strike=100.0,
            premium_paid=3.0,
            contracts=1,
            multiplier=100.0,
            option_type="call",
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            option_expiry_payoff(**base)  # type: ignore[arg-type]


class TestOptionPositionPnl:
    def test_matches_manual_bs_repricing(self) -> None:
        n = 5
        spot_path = np.full(n, 100.0)
        iv_path = np.full(n, 0.2)
        pnl = option_position_pnl(
            entry_price=black_scholes_call(100.0, 100.0, 0.03, 0.2, 0.5),
            contracts=2,
            multiplier=100.0,
            spot_path=spot_path,
            strike=100.0,
            expiry=0.5,
            iv_path=iv_path,
            r=0.03,
            option_type="call",
        )
        times = np.linspace(0.5, 0.0, n)
        expected = np.array([black_scholes_call(100.0, 100.0, 0.03, 0.2, float(t)) for t in times])
        entry_price = black_scholes_call(100.0, 100.0, 0.03, 0.2, 0.5)
        expected_pnl = 2 * 100.0 * (expected - entry_price)
        np.testing.assert_allclose(pnl, expected_pnl, rtol=1e-10)

    def test_theta_decay_flat_path_atm_call(self) -> None:
        n = 10
        spot_path = np.full(n, 100.0)
        iv_path = np.full(n, 0.2)
        entry_price = black_scholes_call(100.0, 100.0, 0.03, 0.2, 1.0)
        pnl = option_position_pnl(
            entry_price=entry_price,
            contracts=1,
            multiplier=100.0,
            spot_path=spot_path,
            strike=100.0,
            expiry=1.0,
            iv_path=iv_path,
            r=0.03,
            option_type="call",
        )
        assert np.all(np.diff(pnl) < 0.0)
        terminal_intrinsic_pnl = option_expiry_payoff(
            spot_at_expiry=100.0,
            strike=100.0,
            premium_paid=entry_price,
            contracts=1,
            multiplier=100.0,
            option_type="call",
        )
        assert pnl[-1] == pytest.approx(terminal_intrinsic_pnl, rel=1e-6)

    def test_expiry_day_collapses_to_intrinsic(self) -> None:
        spot_path = np.array([100.0, 90.0])
        iv_path = np.array([0.2, 0.2])
        pnl = option_position_pnl(
            entry_price=0.0,
            contracts=1,
            multiplier=100.0,
            spot_path=spot_path,
            strike=100.0,
            expiry=0.01,
            iv_path=iv_path,
            r=0.03,
            option_type="put",
        )
        assert pnl[-1] == pytest.approx(100.0 * 10.0)

    def test_single_day_zero_expiry(self) -> None:
        spot_path = np.array([100.0])
        iv_path = np.array([0.2])
        pnl = option_position_pnl(
            entry_price=0.0,
            contracts=1,
            multiplier=100.0,
            spot_path=spot_path,
            strike=90.0,
            expiry=0.0,
            iv_path=iv_path,
            r=0.03,
            option_type="call",
        )
        assert pnl[0] == pytest.approx(100.0 * 10.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(contracts=-1),
            dict(multiplier=0.0),
            dict(strike=0.0),
            dict(expiry=-1.0),
            dict(option_type="straddle"),
        ],
    )
    def test_invalid_scalar_inputs_raise(self, kwargs: dict[str, float]) -> None:
        base = dict(
            entry_price=1.0,
            contracts=1,
            multiplier=100.0,
            spot_path=np.array([100.0, 100.0]),
            strike=100.0,
            expiry=0.5,
            iv_path=np.array([0.2, 0.2]),
            r=0.03,
            option_type="call",
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            option_position_pnl(**base)  # type: ignore[arg-type]

    def test_mismatched_path_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            option_position_pnl(
                entry_price=1.0,
                contracts=1,
                multiplier=100.0,
                spot_path=np.array([100.0, 100.0, 100.0]),
                strike=100.0,
                expiry=0.5,
                iv_path=np.array([0.2, 0.2]),
                r=0.03,
                option_type="call",
            )

    def test_empty_spot_path_raises(self) -> None:
        with pytest.raises(ValueError):
            option_position_pnl(
                entry_price=1.0,
                contracts=1,
                multiplier=100.0,
                spot_path=np.array([]),
                strike=100.0,
                expiry=0.5,
                iv_path=np.array([]),
                r=0.03,
                option_type="call",
            )


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
