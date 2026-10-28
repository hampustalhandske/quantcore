"""Oracle tests for C12/C13 (owner follow-up 2b): dividend yield / cost-of-carry.

`dividend_yield` (q) added additively to `black_scholes_call`/`_put`, every
Greek.
q=0.0 (every function's default) must reproduce the pre-existing q-less
formulas exactly (checked in tests/unit/pricing/); this file checks q > 0
against independent oracles.

Oracle: `py_vollib.black_scholes_merton` — price and Greeks. `py_vollib`
scales vega and rho per 1% instead of per 1.00, so those two are multiplied
by 100 before comparing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vollib")

from vollib.black_scholes_merton import black_scholes_merton as vollib_price
from vollib.black_scholes_merton.greeks.analytical import delta as vollib_delta
from vollib.black_scholes_merton.greeks.analytical import gamma as vollib_gamma
from vollib.black_scholes_merton.greeks.analytical import rho as vollib_rho
from vollib.black_scholes_merton.greeks.analytical import theta as vollib_theta
from vollib.black_scholes_merton.greeks.analytical import vega as vollib_vega

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.greeks import (
    bs_delta,
    bs_gamma,
    bs_rho,
    bs_theta,
    bs_vega,
)

pytestmark = pytest.mark.oracle


BASE = dict(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)
Q = 0.03


class TestBlackScholesMertonPriceMatchesVollib:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_price_matches(self, option_type: str) -> None:
        pricer = black_scholes_call if option_type == "call" else black_scholes_put
        quantcore_price = pricer(**BASE, dividend_yield=Q)

        vollib_flag = "c" if option_type == "call" else "p"
        vollib_result = vollib_price(
            vollib_flag,
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )

        assert quantcore_price == pytest.approx(vollib_result, rel=1e-8)


class TestGreeksMatchVollib:
    def test_delta_call(self) -> None:
        actual = bs_delta(**BASE, option_type="call", dividend_yield=Q)
        vollib_result = vollib_delta(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(vollib_result, rel=1e-6)

    def test_gamma(self) -> None:
        actual = bs_gamma(**BASE, dividend_yield=Q)
        vollib_result = vollib_gamma(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(vollib_result, rel=1e-6)

    def test_vega_per_unit_vol_matches_vollib_per_100(self) -> None:
        actual = bs_vega(**BASE, dividend_yield=Q)
        vollib_result = vollib_vega(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        # py_vollib scales vega per 1% vol move.
        assert actual == pytest.approx(vollib_result * 100.0, rel=1e-4)

    def test_theta_per_calendar_day_matches_vollib(self) -> None:
        actual = bs_theta(**BASE, option_type="call", dividend_yield=Q)
        vollib_result = vollib_theta(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(vollib_result, rel=1e-4)

    def test_rho_per_unit_rate_matches_vollib_per_100(self) -> None:
        actual = bs_rho(**BASE, option_type="call", dividend_yield=Q)
        vollib_result = vollib_rho(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        # py_vollib scales rho per 1% rate move.
        assert actual == pytest.approx(vollib_result * 100.0, rel=1e-4)
