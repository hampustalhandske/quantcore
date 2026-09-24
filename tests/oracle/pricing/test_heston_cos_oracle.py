"""Oracle test: `heston_cos_call` / `heston_cos_put` vs `QuantLib.AnalyticHestonEngine`.

Covers the moneyness / maturity / vol-of-vol grid on which the COS
truncation range matters: K/S in [0.5, 3], T in [7 days, 2 years], xi in
[0.1, 1.0] (the xi=1.0 column violates the Feller condition 2*kappa*theta
> xi^2 by a factor of 12, giving fat-tailed log-returns), at rho = 0 and
rho = -0.7. The QuantLib engine is the semi-analytic Heston (1993)
integral, evaluated with a tight adaptive tolerance so its own error is far
below the tolerance applied here.
"""

from __future__ import annotations

import itertools

import pytest

pytest.importorskip("QuantLib")

import QuantLib as ql

from quantcore.pricing.heston import heston_cos_call, heston_cos_put

pytestmark = pytest.mark.oracle

SPOT = 100.0
V0 = 0.04
KAPPA = 1.0
THETA = 0.04
RATE = 0.0
DIVIDEND_YIELD = 0.0

MONEYNESS = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
MATURITY_DAYS = [7, 37, 183, 365, 730]
VOL_OF_VOL = [0.1, 0.3, 0.6, 1.0]
RHO = [0.0, -0.7]

REL_TOL = 1e-6
ABS_TOL = 1e-8  # applied to prices below 1e-4, where relative error is meaningless


def _ql_heston_price(strike: float, days: int, xi: float, rho: float, option_type: str) -> float:
    today = ql.Date(1, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    maturity = today + ql.Period(days, ql.Days)
    day_count = ql.Actual365Fixed()

    ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type, strike), ql.EuropeanExercise(maturity))

    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, RATE, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, DIVIDEND_YIELD, day_count))
    process = ql.HestonProcess(
        rate_ts, div_ts, ql.QuoteHandle(ql.SimpleQuote(SPOT)), V0, KAPPA, THETA, xi, rho
    )
    model = ql.HestonModel(process)
    # Adaptive Gauss-Lobatto at 1e-12 relative; Gauss-Laguerre as a backup
    # for the rare parameter set where the adaptive rule does not settle.
    for engine in (
        ql.AnalyticHestonEngine(model, 1e-12, 1_000_000),
        ql.AnalyticHestonEngine(
            model,
            ql.AnalyticHestonEngine.Gatheral,
            ql.AnalyticHestonEngine_Integration.gaussLaguerre(192),
        ),
    ):
        option.setPricingEngine(engine)
        try:
            return float(option.NPV())
        except RuntimeError:
            continue
    raise RuntimeError("QuantLib could not price the reference option")


def _assert_close(actual: float, expected: float) -> None:
    if abs(expected) > 1e-4:
        assert actual == pytest.approx(expected, rel=REL_TOL)
    else:
        assert actual == pytest.approx(expected, abs=ABS_TOL)


class TestHestonCosMatchesQuantLibAcrossGrid:
    @pytest.mark.parametrize("rho", RHO)
    @pytest.mark.parametrize("xi", VOL_OF_VOL)
    def test_call_prices(self, xi: float, rho: float) -> None:
        for moneyness, days in itertools.product(MONEYNESS, MATURITY_DAYS):
            strike = SPOT * moneyness
            expected = _ql_heston_price(strike, days, xi, rho, "call")
            actual = heston_cos_call(SPOT, strike, RATE, days / 365.0, V0, KAPPA, THETA, xi, rho)
            _assert_close(actual, expected)

    @pytest.mark.parametrize("rho", RHO)
    @pytest.mark.parametrize("xi", VOL_OF_VOL)
    def test_put_prices(self, xi: float, rho: float) -> None:
        for moneyness, days in itertools.product(MONEYNESS, MATURITY_DAYS):
            strike = SPOT * moneyness
            expected = _ql_heston_price(strike, days, xi, rho, "put")
            actual = heston_cos_put(SPOT, strike, RATE, days / 365.0, V0, KAPPA, THETA, xi, rho)
            _assert_close(actual, expected)


class TestHandoverReproduction:
    """The exact cases from the edgelab finding: one-week options at
    K=60 and K=300 on S=100, xi=0.3, rho=0."""

    def test_deep_itm_call_is_intrinsic(self) -> None:
        assert heston_cos_call(100.0, 60.0, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0) == pytest.approx(
            40.0, abs=1e-8
        )

    def test_deep_otm_call_is_worthless(self) -> None:
        assert heston_cos_call(100.0, 300.0, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0) == pytest.approx(
            0.0, abs=1e-8
        )

    def test_puts_by_parity(self) -> None:
        assert heston_cos_put(100.0, 60.0, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0) == pytest.approx(
            0.0, abs=1e-8
        )
        assert heston_cos_put(100.0, 300.0, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0) == pytest.approx(
            200.0, abs=1e-8
        )
