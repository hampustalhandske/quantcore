"""Tests for the Heston COS-method European option pricer."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.heston import (
    _heston_char_func,
    _heston_cumulants,
    heston_cos_call,
    heston_cos_put,
)

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
