"""Tests for Black-Scholes Greeks and implied volatility inversion."""

from __future__ import annotations

import pytest

from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.greeks import (
    bs_delta,
    bs_gamma,
    bs_rho,
    bs_theta,
    bs_vega,
    implied_volatility,
)

_ATM = dict(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)


class TestBsDelta:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            bs_delta(
                spot=0.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            bs_delta(
                spot=100.0,
                strike=-1.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_volatility(self) -> None:
        with pytest.raises(ValueError):
            bs_delta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=-0.1,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            bs_delta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=-1.0,
                option_type="call",
            )

    def test_invalid_option_type(self) -> None:
        with pytest.raises(ValueError):
            bs_delta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="straddle",
            )

    def test_atm_call_known_value(self) -> None:
        # d1 = (0 + (0.05+0.5*0.2^2)*1) / (0.2*sqrt(1)) = 0.35, N(0.35) ~= 0.6368.
        delta = bs_delta(**_ATM, option_type="call")
        assert delta == pytest.approx(0.6368, abs=1e-3)

    def test_call_put_symmetry(self) -> None:
        delta_call = bs_delta(**_ATM, option_type="call")
        delta_put = bs_delta(**_ATM, option_type="put")
        assert (delta_call - delta_put) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("spot", "strike", "expected"),
        [(110.0, 100.0, 1.0), (90.0, 100.0, 0.0)],
    )
    def test_zero_time_to_maturity_call(self, spot: float, strike: float, expected: float) -> None:
        delta = bs_delta(
            spot=spot,
            strike=strike,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=0.0,
            option_type="call",
        )
        assert delta == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize(
        ("spot", "strike", "expected"),
        [(90.0, 100.0, -1.0), (110.0, 100.0, 0.0)],
    )
    def test_zero_time_to_maturity_put(self, spot: float, strike: float, expected: float) -> None:
        delta = bs_delta(
            spot=spot,
            strike=strike,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=0.0,
            option_type="put",
        )
        assert delta == pytest.approx(expected, abs=1e-9)


class TestBsGamma:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            bs_gamma(spot=0.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            bs_gamma(spot=100.0, strike=0.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)

    def test_invalid_volatility(self) -> None:
        with pytest.raises(ValueError):
            bs_gamma(spot=100.0, strike=100.0, rate=0.05, volatility=-0.2, time_to_maturity=1.0)

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            bs_gamma(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=-1.0)

    def test_gamma_positive(self) -> None:
        assert bs_gamma(**_ATM) > 0.0

    def test_zero_time_to_maturity_is_zero(self) -> None:
        gamma = bs_gamma(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=0.0)
        assert gamma == 0.0


class TestBsVega:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            bs_vega(spot=-1.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            bs_vega(spot=100.0, strike=-1.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)

    def test_invalid_volatility(self) -> None:
        with pytest.raises(ValueError):
            bs_vega(spot=100.0, strike=100.0, rate=0.05, volatility=-0.2, time_to_maturity=1.0)

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            bs_vega(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=-1.0)

    def test_vega_positive(self) -> None:
        assert bs_vega(**_ATM) > 0.0

    def test_zero_time_to_maturity_is_zero(self) -> None:
        vega = bs_vega(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=0.0)
        assert vega == 0.0


class TestBsTheta:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            bs_theta(
                spot=-1.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            bs_theta(
                spot=100.0,
                strike=-1.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_volatility(self) -> None:
        with pytest.raises(ValueError):
            bs_theta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=-0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            bs_theta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=-1.0,
                option_type="call",
            )

    def test_invalid_option_type(self) -> None:
        with pytest.raises(ValueError):
            bs_theta(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="foo",
            )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_theta_typically_negative_for_long_option(self, option_type: str) -> None:
        theta = bs_theta(**_ATM, option_type=option_type)
        assert theta < 0.0

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_zero_time_to_maturity_is_zero(self, option_type: str) -> None:
        theta = bs_theta(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=0.0,
            option_type=option_type,
        )
        assert theta == 0.0


class TestBsRho:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            bs_rho(
                spot=-1.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            bs_rho(
                spot=100.0,
                strike=-1.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_volatility(self) -> None:
        with pytest.raises(ValueError):
            bs_rho(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=-0.2,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            bs_rho(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=-1.0,
                option_type="call",
            )

    def test_invalid_option_type(self) -> None:
        with pytest.raises(ValueError):
            bs_rho(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="foo",
            )

    def test_call_rho_positive(self) -> None:
        assert bs_rho(**_ATM, option_type="call") > 0.0

    def test_put_rho_negative(self) -> None:
        assert bs_rho(**_ATM, option_type="put") < 0.0

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_zero_time_to_maturity_is_zero(self, option_type: str) -> None:
        rho = bs_rho(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=0.0,
            option_type=option_type,
        )
        assert rho == 0.0


class TestImpliedVolatility:
    def test_invalid_spot(self) -> None:
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=10.0,
                spot=-1.0,
                strike=100.0,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_strike(self) -> None:
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=10.0,
                spot=100.0,
                strike=-1.0,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_invalid_time_to_maturity(self) -> None:
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=10.0,
                spot=100.0,
                strike=100.0,
                rate=0.05,
                time_to_maturity=-1.0,
                option_type="call",
            )

    def test_invalid_option_type(self) -> None:
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=10.0,
                spot=100.0,
                strike=100.0,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="foo",
            )

    def test_round_trip_recovers_volatility(self) -> None:
        true_sigma = 0.25
        price = black_scholes_call(
            spot=100.0, strike=100.0, rate=0.05, volatility=true_sigma, time_to_maturity=1.0
        )
        recovered = implied_volatility(
            market_price=price,
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            option_type="call",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-6)

    def test_round_trip_recovers_volatility_put(self) -> None:
        true_sigma = 0.25
        price = black_scholes_put(
            spot=100.0, strike=100.0, rate=0.05, volatility=true_sigma, time_to_maturity=1.0
        )
        recovered = implied_volatility(
            market_price=price,
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=1.0,
            option_type="put",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-6)

    def test_price_below_intrinsic_value_raises(self) -> None:
        # Intrinsic value of an ITM call with spot=110, strike=100 is 10.0;
        # a quoted price below that admits no arbitrage-free volatility.
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=5.0,
                spot=110.0,
                strike=100.0,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="call",
            )
