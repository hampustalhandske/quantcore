"""Oracle tests for C12/C13 (owner follow-up 2b): dividend yield / cost-of-carry.

`dividend_yield` (q) added additively to `black_scholes_call`/`_put`, every
Greek, `implied_volatility`, and `heston_cos_call`; `heston_cos_put` added.
q=0.0 (every function's default) must reproduce the pre-existing q-less
formulas exactly (checked in tests/unit/pricing/); this file checks q > 0
against independent oracles.

Oracles:
- `QuantLib.AnalyticEuropeanEngine` with a `BlackScholesMertonProcess`
  (dividend term structure) — price and all five Greeks, in the same raw
  units quantcore uses (QuantLib's `vega()`/`rho()` are per 1.00 of
  vol/rate, `thetaPerDay()` is per calendar day — matching quantcore's
  documented units, confirmed against `py_vollib` below).
- `py_vollib.black_scholes_merton` — price and Greeks, cross-checked
  against QuantLib's units (`py_vollib` scales vega/rho per 1% instead of
  per 1.00, so those two are divided by 100 before comparing).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("QuantLib")
pytest.importorskip("vollib")

import QuantLib as ql
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
    implied_volatility,
)
from quantcore.pricing.heston import heston_cos_call, heston_cos_put

pytestmark = pytest.mark.oracle


def _ql_price_and_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float,
    option_type: str,
) -> dict[str, float]:
    today = ql.Date(1, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = today + round(time_to_maturity * 365)
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()

    ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, volatility, day_count)
    )
    process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "theta_per_day": option.thetaPerDay(),
        "rho": option.rho(),
    }


BASE = dict(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)
Q = 0.03


class TestBlackScholesMertonPriceMatchesQuantLibAndVollib:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_price_matches(self, option_type: str) -> None:
        pricer = black_scholes_call if option_type == "call" else black_scholes_put
        quantcore_price = pricer(**BASE, dividend_yield=Q)

        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type=option_type)
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

        assert quantcore_price == pytest.approx(ql_result["price"], rel=1e-8)
        assert quantcore_price == pytest.approx(vollib_result, rel=1e-8)


class TestGreeksMatchQuantLibAndVollib:
    def test_delta_call(self) -> None:
        actual = bs_delta(**BASE, option_type="call", dividend_yield=Q)
        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type="call")
        vollib_result = vollib_delta(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(ql_result["delta"], rel=1e-6)
        assert actual == pytest.approx(vollib_result, rel=1e-6)

    def test_gamma(self) -> None:
        actual = bs_gamma(**BASE, dividend_yield=Q)
        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type="call")
        vollib_result = vollib_gamma(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(ql_result["gamma"], rel=1e-6)
        assert actual == pytest.approx(vollib_result, rel=1e-6)

    def test_vega_per_unit_vol_matches_quantlib_and_vollib_per_100(self) -> None:
        actual = bs_vega(**BASE, dividend_yield=Q)
        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type="call")
        vollib_result = vollib_vega(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(ql_result["vega"], rel=1e-6)
        # py_vollib scales vega per 1% vol move.
        assert actual == pytest.approx(vollib_result * 100.0, rel=1e-4)

    def test_theta_per_calendar_day_matches_quantlib_and_vollib(self) -> None:
        actual = bs_theta(**BASE, option_type="call", dividend_yield=Q)
        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type="call")
        vollib_result = vollib_theta(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(ql_result["theta_per_day"], rel=1e-4)
        assert actual == pytest.approx(vollib_result, rel=1e-4)

    def test_rho_per_unit_rate_matches_quantlib_and_vollib_per_100(self) -> None:
        actual = bs_rho(**BASE, option_type="call", dividend_yield=Q)
        ql_result = _ql_price_and_greeks(**BASE, dividend_yield=Q, option_type="call")
        vollib_result = vollib_rho(
            "c",
            BASE["spot"],
            BASE["strike"],
            BASE["time_to_maturity"],
            BASE["rate"],
            BASE["volatility"],
            Q,
        )
        assert actual == pytest.approx(ql_result["rho"], rel=1e-6)
        # py_vollib scales rho per 1% rate move.
        assert actual == pytest.approx(vollib_result * 100.0, rel=1e-4)


class TestImpliedVolatilityRoundTripsWithDividendYield:
    def test_round_trip_matches_true_sigma(self) -> None:
        true_sigma = 0.27
        market_price = black_scholes_call(**{**BASE, "volatility": true_sigma}, dividend_yield=Q)
        recovered = implied_volatility(
            market_price=market_price,
            spot=BASE["spot"],
            strike=BASE["strike"],
            rate=BASE["rate"],
            time_to_maturity=BASE["time_to_maturity"],
            option_type="call",
            dividend_yield=Q,
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-6)

    def test_grid_of_moneyness_and_maturities(self) -> None:
        rng = np.random.default_rng(20240921)
        for _ in range(20):
            spot = 100.0
            strike = float(rng.uniform(70.0, 130.0))
            time_to_maturity = float(rng.uniform(0.05, 2.0))
            sigma = float(rng.uniform(0.05, 0.6))
            q = float(rng.uniform(0.0, 0.08))
            price = black_scholes_call(spot, strike, BASE["rate"], sigma, time_to_maturity, q)
            recovered = implied_volatility(
                market_price=price,
                spot=spot,
                strike=strike,
                rate=BASE["rate"],
                time_to_maturity=time_to_maturity,
                option_type="call",
                dividend_yield=q,
            )
            assert recovered == pytest.approx(sigma, abs=1e-5)


def _ql_heston_price(
    spot: float,
    strike: float,
    rate: float,
    time_to_maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    today = ql.Date(1, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = today + round(time_to_maturity * 365)
    day_count = ql.Actual365Fixed()

    ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, day_count))
    process = ql.HestonProcess(rate_ts, div_ts, spot_handle, v0, kappa, theta, xi, rho)
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
    return float(option.NPV())


HESTON_BASE = dict(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    time_to_maturity=1.0,
    v0=0.04,
    kappa=1.5,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
)


class TestHestonDividendYieldMatchesQuantLib:
    def test_call_price_matches(self) -> None:
        actual = heston_cos_call(**HESTON_BASE, dividend_yield=Q)
        expected = _ql_heston_price(**HESTON_BASE, dividend_yield=Q, option_type="call")
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_put_price_matches(self) -> None:
        actual = heston_cos_put(**HESTON_BASE, dividend_yield=Q)
        expected = _ql_heston_price(**HESTON_BASE, dividend_yield=Q, option_type="put")
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_zero_dividend_yield_matches_quantlib(self) -> None:
        actual = heston_cos_call(**HESTON_BASE)
        expected = _ql_heston_price(**HESTON_BASE, dividend_yield=0.0, option_type="call")
        assert actual == pytest.approx(expected, rel=1e-5)
