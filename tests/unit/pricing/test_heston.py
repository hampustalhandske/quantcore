"""Tests for the Heston COS-method European call pricer."""

from __future__ import annotations

import pytest

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.heston import heston_cos_call, heston_cos_put

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
        import numpy as np

        call = heston_cos_call(**BASE_KWARGS, dividend_yield=0.02)
        put = heston_cos_put(**BASE_KWARGS, dividend_yield=0.02)
        expected_diff = BASE_KWARGS["spot"] * np.exp(
            -0.02 * BASE_KWARGS["time_to_maturity"]
        ) - BASE_KWARGS["strike"] * np.exp(-BASE_KWARGS["rate"] * BASE_KWARGS["time_to_maturity"])
        assert (call - put) == pytest.approx(expected_diff, abs=1e-8)

    def test_price_within_no_arbitrage_bounds(self) -> None:
        price = heston_cos_put(**BASE_KWARGS)
        assert 0.0 < price < BASE_KWARGS["strike"]
