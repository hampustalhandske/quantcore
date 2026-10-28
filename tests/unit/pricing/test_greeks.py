"""Tests for Black-Scholes Greeks."""

from __future__ import annotations

import pytest

from quantcore.pricing.black_scholes import black_scholes_call
from quantcore.pricing.greeks import (
    bs_delta,
    bs_gamma,
    bs_rho,
    bs_theta,
    bs_vega,
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


_DIV_KWARGS = dict(spot=100.0, strike=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0)
_DIV_Q = 0.03
_DIV_EPS = 1e-4


class TestDividendYield:
    """All Greeks with q=0.0 (the default) must reproduce the plain
    Black-Scholes value; with q>0, each Greek must satisfy the analytic
    finite-difference relationship to the priced call/put with the same q.
    """

    @pytest.mark.parametrize(
        ("greek", "kwargs"),
        [
            (bs_delta, {"option_type": "call"}),
            (bs_delta, {"option_type": "put"}),
            (bs_gamma, {}),
            (bs_vega, {}),
            (bs_theta, {"option_type": "call"}),
            (bs_theta, {"option_type": "put"}),
            (bs_rho, {"option_type": "call"}),
            (bs_rho, {"option_type": "put"}),
        ],
    )
    def test_zero_dividend_yield_matches_plain_value(self, greek, kwargs: dict) -> None:
        plain = greek(**_DIV_KWARGS, **kwargs)
        with_zero_q = greek(**_DIV_KWARGS, **kwargs, dividend_yield=0.0)
        assert with_zero_q == plain

    def test_delta_matches_finite_difference_of_call_price_with_dividend(self) -> None:
        spot = _DIV_KWARGS["spot"]
        bumped = {**_DIV_KWARGS, "spot": spot + _DIV_EPS}
        dropped = {**_DIV_KWARGS, "spot": spot - _DIV_EPS}
        up = black_scholes_call(**bumped, dividend_yield=_DIV_Q)
        down = black_scholes_call(**dropped, dividend_yield=_DIV_Q)
        expected = (up - down) / (2.0 * _DIV_EPS)
        actual = bs_delta(**_DIV_KWARGS, option_type="call", dividend_yield=_DIV_Q)
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_vega_matches_finite_difference_with_dividend(self) -> None:
        vol = _DIV_KWARGS["volatility"]
        kwargs = {k: v for k, v in _DIV_KWARGS.items() if k != "volatility"}
        up = black_scholes_call(**kwargs, volatility=vol + _DIV_EPS, dividend_yield=_DIV_Q)
        down = black_scholes_call(**kwargs, volatility=vol - _DIV_EPS, dividend_yield=_DIV_Q)
        expected = (up - down) / (2.0 * _DIV_EPS)
        actual = bs_vega(**_DIV_KWARGS, dividend_yield=_DIV_Q)
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_rho_matches_finite_difference_of_call_price_with_dividend(self) -> None:
        # Rho's closed form doesn't mention q explicitly, but its *value*
        # still shifts with q through d2 -- so this checks the value
        # against a finite difference, not that it equals the q=0 value.
        rate = _DIV_KWARGS["rate"]
        kwargs = {k: v for k, v in _DIV_KWARGS.items() if k != "rate"}
        up = black_scholes_call(**kwargs, rate=rate + _DIV_EPS, dividend_yield=_DIV_Q)
        down = black_scholes_call(**kwargs, rate=rate - _DIV_EPS, dividend_yield=_DIV_Q)
        expected = (up - down) / (2.0 * _DIV_EPS)
        actual = bs_rho(**_DIV_KWARGS, option_type="call", dividend_yield=_DIV_Q)
        assert actual == pytest.approx(expected, abs=1e-4)
