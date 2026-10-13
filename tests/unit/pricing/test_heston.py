"""Tests for the Heston COS-method European option pricer."""

from __future__ import annotations

from functools import partial
from types import MappingProxyType

import numpy as np
import pytest
import QuantLib as ql

from quantcore import pricing
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.heston import _heston_char_func, _heston_cumulants

# `TestHestonCosCall`, `TestHestonCumulants` and `TestHestonCosPut` test the
# COS expansion, which is the "quantcore" backend. `TestQuantLibBackend` and
# `TestBackendDomains` call the public functions with their default backend.
heston_cos_call = partial(pricing.heston_cos_call, backend="quantcore")
heston_cos_put = partial(pricing.heston_cos_put, backend="quantcore")

BASE_KWARGS = dict(
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


class TestHestonCosCall:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"spot": 0.0},
            {"spot": -1.0},
            {"strike": 0.0},
            {"strike": -1.0},
            {"time_to_maturity": -0.1},
            {"v0": 0.0},
            {"v0": -0.01},
            {"theta": 0.0},
            {"theta": -0.01},
            {"kappa": 0.0},
            {"kappa": -0.5},
            {"xi": -0.1},
            {"rho": 1.5},
            {"rho": -1.5},
        ],
    )
    def test_invalid_inputs(self, overrides: dict) -> None:
        kwargs = {**BASE_KWARGS, **overrides}
        with pytest.raises(ValueError):
            heston_cos_call(**kwargs)

    def test_reduces_to_black_scholes_when_xi_zero(self) -> None:
        # xi=0 and rho=0 collapse Heston to constant-variance GBM, i.e. BS
        # with volatility = sqrt(v0) = sqrt(theta).
        kwargs = {**BASE_KWARGS, "v0": 0.04, "theta": 0.04, "xi": 0.0, "rho": 0.0}
        heston_price = heston_cos_call(**kwargs)
        bs_price = black_scholes_call(
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            rate=kwargs["rate"],
            volatility=kwargs["v0"] ** 0.5,
            time_to_maturity=kwargs["time_to_maturity"],
        )
        assert heston_price == pytest.approx(bs_price, abs=1e-2)

    def test_convergence_stabilizes_with_more_terms(self) -> None:
        price_64 = heston_cos_call(**BASE_KWARGS, n_terms=64)
        price_256 = heston_cos_call(**BASE_KWARGS, n_terms=256)
        assert price_64 == pytest.approx(price_256, abs=1e-2)

    def test_price_within_no_arbitrage_bounds(self) -> None:
        price = heston_cos_call(**BASE_KWARGS)
        assert 0.0 < price < BASE_KWARGS["spot"]

    def test_price_decreases_with_strike(self) -> None:
        low_strike = heston_cos_call(**{**BASE_KWARGS, "strike": 90.0})
        high_strike = heston_cos_call(**{**BASE_KWARGS, "strike": 110.0})
        assert low_strike >= high_strike

    def test_zero_dividend_yield_matches_plain_call(self) -> None:
        plain = heston_cos_call(**BASE_KWARGS)
        with_zero_q = heston_cos_call(**BASE_KWARGS, dividend_yield=0.0)
        assert with_zero_q == plain

    def test_dividend_yield_lowers_call_price(self) -> None:
        no_div = heston_cos_call(**BASE_KWARGS)
        with_div = heston_cos_call(**BASE_KWARGS, dividend_yield=0.03)
        assert with_div < no_div

    def test_reduces_to_black_scholes_with_integrated_variance_when_xi_zero(self) -> None:
        # With xi=0 the variance path is the deterministic mean-reverting
        # curve, so the price is Black-Scholes at the root-mean integrated
        # variance -- not at sqrt(v0).
        kwargs = {**BASE_KWARGS, "v0": 0.09, "theta": 0.04, "kappa": 2.0, "xi": 0.0, "rho": 0.0}
        t = kwargs["time_to_maturity"]
        integrated_var = 0.04 * t + (0.09 - 0.04) * (1.0 - np.exp(-2.0 * t)) / 2.0
        bs_price = black_scholes_call(
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            rate=kwargs["rate"],
            volatility=float(np.sqrt(integrated_var / t)),
            time_to_maturity=t,
        )
        assert heston_cos_call(**kwargs) == pytest.approx(bs_price, rel=1e-8)

    @pytest.mark.parametrize(
        ("strike", "expected"),
        [
            (60.0, 40.0),  # one-week deep ITM call: intrinsic value
            (300.0, 0.0),  # one-week deep OTM call: worthless
        ],
    )
    def test_far_moneyness_short_expiry(self, strike: float, expected: float) -> None:
        # The truncation range is centred on the log-moneyness, so a strike
        # far from spot no longer pushes the payoff outside the range.
        price = heston_cos_call(
            spot=100.0,
            strike=strike,
            rate=0.0,
            time_to_maturity=0.02,
            v0=0.04,
            kappa=1.0,
            theta=0.04,
            xi=0.3,
            rho=0.0,
        )
        assert price == pytest.approx(expected, abs=1e-8)

    def test_deep_in_the_money_call_above_intrinsic_bound(self) -> None:
        kwargs = {**BASE_KWARGS, "strike": 50.0}
        price = heston_cos_call(**kwargs)
        lower_bound = kwargs["spot"] - kwargs["strike"] * np.exp(
            -kwargs["rate"] * kwargs["time_to_maturity"]
        )
        assert lower_bound <= price < kwargs["spot"]


class TestHestonCumulants:
    @pytest.mark.parametrize("xi", [0.1, 0.3, 0.6, 1.0])
    @pytest.mark.parametrize("rho", [0.0, -0.7, 0.5])
    @pytest.mark.parametrize("time_to_maturity", [0.02, 0.5, 2.0])
    def test_c1_c2_match_characteristic_function_derivatives(
        self, xi: float, rho: float, time_to_maturity: float
    ) -> None:
        # The closed-form mean and variance must agree with numerical
        # derivatives of ln(phi) at u=0.
        drift, v0, kappa, theta = 0.02, 0.06, 1.0, 0.04
        c1, c2, c4 = _heston_cumulants(drift, time_to_maturity, v0, kappa, theta, xi, rho)
        h = 1e-3
        u = np.array([-h, 0.0, h], dtype=np.complex128)
        log_phi = np.log(_heston_char_func(u, drift, time_to_maturity, v0, kappa, theta, xi, rho))
        numeric_c1 = float(((log_phi[2] - log_phi[0]) / (2.0 * h) / 1j).real)
        numeric_c2 = float((-(log_phi[2] - 2.0 * log_phi[1] + log_phi[0]) / h**2).real)
        # Central differences carry O(h^2) truncation error of order 1e-8.
        assert c1 == pytest.approx(numeric_c1, abs=1e-7)
        assert c2 == pytest.approx(numeric_c2, rel=1e-4)
        assert c4 >= 0.0

    def test_c4_vanishes_when_xi_zero(self) -> None:
        _, _, c4 = _heston_cumulants(0.0, 1.0, 0.04, 1.0, 0.04, 0.0, 0.0)
        assert c4 == 0.0


class TestHestonCosPut:
    def test_reduces_to_black_scholes_when_xi_zero(self) -> None:
        kwargs = {**BASE_KWARGS, "v0": 0.04, "theta": 0.04, "xi": 0.0, "rho": 0.0}
        heston_price = heston_cos_put(**kwargs)
        bs_price = black_scholes_put(
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            rate=kwargs["rate"],
            volatility=kwargs["v0"] ** 0.5,
            time_to_maturity=kwargs["time_to_maturity"],
        )
        assert heston_price == pytest.approx(bs_price, abs=1e-2)

    def test_put_call_parity(self) -> None:
        call = heston_cos_call(**BASE_KWARGS, dividend_yield=0.02)
        put = heston_cos_put(**BASE_KWARGS, dividend_yield=0.02)
        expected_diff = BASE_KWARGS["spot"] * np.exp(
            -0.02 * BASE_KWARGS["time_to_maturity"]
        ) - BASE_KWARGS["strike"] * np.exp(-BASE_KWARGS["rate"] * BASE_KWARGS["time_to_maturity"])
        assert (call - put) == pytest.approx(expected_diff, abs=1e-8)

    def test_price_within_no_arbitrage_bounds(self) -> None:
        price = heston_cos_put(**BASE_KWARGS)
        assert 0.0 < price < BASE_KWARGS["strike"]

    @pytest.mark.parametrize(
        ("strike", "expected"),
        [
            (60.0, 0.0),  # one-week deep OTM put: worthless
            (300.0, 200.0),  # one-week deep ITM put: intrinsic value
        ],
    )
    def test_far_moneyness_short_expiry(self, strike: float, expected: float) -> None:
        price = heston_cos_put(
            spot=100.0,
            strike=strike,
            rate=0.0,
            time_to_maturity=0.02,
            v0=0.04,
            kappa=1.0,
            theta=0.04,
            xi=0.3,
            rho=0.0,
        )
        assert price == pytest.approx(expected, abs=1e-8)


def _quantlib_heston_price(
    option_type: int,
    spot: float,
    strike: float,
    rate: float,
    days: int,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dividend_yield: float,
) -> float:
    """`AnalyticHestonEngine` price for a maturity of a whole number of days (Actual/365)."""
    today = ql.Settings.instance().evaluationDate
    day_count = ql.Actual365Fixed()
    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, day_count)),
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        v0,
        kappa,
        theta,
        xi,
        rho,
    )
    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(option_type, strike), ql.EuropeanExercise(today + days)
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
    return float(option.NPV())


class TestQuantLibBackend:
    """The default backend: QuantLib's `AnalyticHestonEngine`."""

    PARAMS = MappingProxyType({"v0": 0.04, "kappa": 1.5, "theta": 0.05, "xi": 0.4, "rho": -0.6})

    def test_is_the_default(self) -> None:
        args = (100.0, 110.0, 0.03, 0.5)
        assert pricing.heston_cos_call(*args, **self.PARAMS) == pricing.heston_cos_call(
            *args, **self.PARAMS, backend="quantlib"
        )
        assert pricing.heston_cos_put(*args, **self.PARAMS) == pricing.heston_cos_put(
            *args, **self.PARAMS, backend="quantlib"
        )

    @pytest.mark.parametrize("days", [7, 30, 183, 365, 730])
    @pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
    def test_whole_day_maturity_is_the_engine_price(self, days: int, strike: float) -> None:
        maturity = days / 365.0
        call = pricing.heston_cos_call(
            100.0, strike, 0.03, maturity, **self.PARAMS, dividend_yield=0.01
        )
        put = pricing.heston_cos_put(
            100.0, strike, 0.03, maturity, **self.PARAMS, dividend_yield=0.01
        )
        expected_call = _quantlib_heston_price(
            ql.Option.Call, 100.0, strike, 0.03, days, 0.04, 1.5, 0.05, 0.4, -0.6, 0.01
        )
        expected_put = _quantlib_heston_price(
            ql.Option.Put, 100.0, strike, 0.03, days, 0.04, 1.5, 0.05, 0.4, -0.6, 0.01
        )
        assert call == pytest.approx(expected_call, rel=1e-12, abs=1e-14)
        assert put == pytest.approx(expected_put, rel=1e-12, abs=1e-14)

    @pytest.mark.parametrize("maturity", [0.0137, 0.1, 0.37, 1.6181657, 2.5])
    @pytest.mark.parametrize("strike", [60.0, 100.0, 180.0])
    def test_any_maturity_agrees_with_the_cos_backend(self, maturity: float, strike: float) -> None:
        for price in (pricing.heston_cos_call, pricing.heston_cos_put):
            from_quantlib = price(100.0, strike, 0.02, maturity, **self.PARAMS, dividend_yield=0.01)
            from_cos = price(
                100.0,
                strike,
                0.02,
                maturity,
                **self.PARAMS,
                dividend_yield=0.01,
                backend="quantcore",
            )
            assert from_quantlib == pytest.approx(from_cos, rel=1e-6, abs=1e-8)

    def test_put_call_parity(self) -> None:
        spot, strike, rate, maturity, dividend_yield = 100.0, 110.0, 0.03, 0.37, 0.01
        call = pricing.heston_cos_call(
            spot, strike, rate, maturity, **self.PARAMS, dividend_yield=dividend_yield
        )
        put = pricing.heston_cos_put(
            spot, strike, rate, maturity, **self.PARAMS, dividend_yield=dividend_yield
        )
        forward_value = spot * np.exp(-dividend_yield * maturity) - strike * np.exp(
            -rate * maturity
        )
        assert call - put == pytest.approx(forward_value, abs=1e-10)

    def test_leaves_the_quantlib_evaluation_date_alone_and_does_not_depend_on_it(self) -> None:
        settings = ql.Settings.instance()
        original = settings.evaluationDate
        args = (100.0, 95.0, 0.03, 0.41)
        try:
            baseline = pricing.heston_cos_call(*args, **self.PARAMS)
            settings.evaluationDate = ql.Date(15, 6, 2031)
            shifted = pricing.heston_cos_call(*args, **self.PARAMS)
            assert settings.evaluationDate == ql.Date(15, 6, 2031)
        finally:
            settings.evaluationDate = original
        assert shifted == pytest.approx(baseline, rel=1e-12)

    def test_returns_a_python_float(self) -> None:
        assert type(pricing.heston_cos_call(100.0, 100.0, 0.03, 1.0, **self.PARAMS)) is float

    @pytest.mark.parametrize("price", [pricing.heston_cos_call, pricing.heston_cos_put])
    def test_unknown_backend_raises(self, price) -> None:
        with pytest.raises(ValueError, match="backend"):
            price(100.0, 100.0, 0.03, 1.0, **self.PARAMS, backend="cos")

    def test_invalid_inputs_raise_before_quantlib_is_called(self) -> None:
        with pytest.raises(ValueError):
            pricing.heston_cos_call(-100.0, 100.0, 0.03, 1.0, **self.PARAMS)


class TestBackendDomains:
    """Inputs one backend prices and the other cannot."""

    PARAMS = MappingProxyType({"v0": 0.04, "kappa": 1.5, "theta": 0.05, "rho": -0.6})

    @pytest.mark.parametrize("price", [pricing.heston_cos_call, pricing.heston_cos_put])
    def test_quantlib_backend_rejects_zero_vol_of_vol(self, price) -> None:
        with pytest.raises(ValueError, match='backend="quantcore"'):
            price(100.0, 105.0, 0.03, 0.5, xi=0.0, **self.PARAMS)

    def test_quantlib_backend_prices_a_tiny_vol_of_vol_at_its_deterministic_limit(self) -> None:
        tiny = pricing.heston_cos_call(100.0, 105.0, 0.03, 0.5, xi=1e-9, **self.PARAMS)
        limit = pricing.heston_cos_call(
            100.0, 105.0, 0.03, 0.5, xi=0.0, **self.PARAMS, backend="quantcore"
        )
        assert tiny == pytest.approx(limit, rel=1e-7)

    @pytest.mark.parametrize("xi", [1e-12, 1e-8, 1e-6, 9.9e-5])
    @pytest.mark.parametrize("price", [pricing.heston_cos_call, pricing.heston_cos_put])
    def test_quantcore_backend_rejects_a_vol_of_vol_too_small_for_its_arithmetic(
        self, price, xi: float
    ) -> None:
        with pytest.raises(ValueError, match='backend="quantlib"'):
            price(100.0, 105.0, 0.03, 0.5, xi=xi, **self.PARAMS, backend="quantcore")

    @pytest.mark.parametrize("backend", ["quantlib", "quantcore"])
    def test_zero_maturity_is_the_intrinsic_value(self, backend: str) -> None:
        params = {**self.PARAMS, "xi": 0.4, "backend": backend}
        assert pricing.heston_cos_call(100.0, 90.0, 0.03, 0.0, **params) == 10.0
        assert pricing.heston_cos_put(100.0, 90.0, 0.03, 0.0, **params) == 0.0
        assert pricing.heston_cos_call(100.0, 120.0, 0.03, 0.0, **params) == 0.0
        assert pricing.heston_cos_put(100.0, 120.0, 0.03, 0.0, **params) == 20.0
