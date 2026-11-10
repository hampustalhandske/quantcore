"""Tests for Black-Scholes Greeks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quantcore import UndefinedResultError
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put
from quantcore.pricing.greeks import (
    bs_delta,
    bs_gamma,
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


# Zero volatility with the forward exactly at the strike: ln(S/K) + (r - q)T = 0, so d1 = 0/0.
_FORWARD_AT_STRIKE = dict(spot=100.0, strike=100.0, rate=0.0, volatility=0.0, time_to_maturity=1.0)


class TestUndefinedResults:
    @pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
    @pytest.mark.parametrize(
        "greek",
        [
            lambda **kw: bs_delta(option_type="call", **kw),
            lambda **kw: bs_delta(option_type="put", **kw),
            bs_vega,
            lambda **kw: bs_theta(option_type="call", **kw),
            lambda **kw: bs_theta(option_type="put", **kw),
        ],
        ids=["delta_call", "delta_put", "vega", "theta_call", "theta_put"],
    )
    def test_zero_volatility_with_the_forward_at_the_strike_raises(self, greek) -> None:
        with pytest.raises(UndefinedResultError, match="not finite") as raised:
            greek(**_FORWARD_AT_STRIKE)
        assert isinstance(raised.value, ValueError)

    def test_gamma_with_zero_volatility_and_the_forward_at_the_strike_raises(self) -> None:
        # The sigma -> 0+ limit of Gamma there is +inf.
        carry = {**_FORWARD_AT_STRIKE, "rate": 0.03, "dividend_yield": 0.03}
        for kwargs in (_FORWARD_AT_STRIKE, carry):
            with pytest.raises(UndefinedResultError, match="Gamma") as raised:
                bs_gamma(**kwargs)
            assert isinstance(raised.value, ValueError)

    def test_gamma_is_zero_at_zero_volatility_off_the_forward_and_at_expiry(self) -> None:
        assert bs_gamma(**{**_FORWARD_AT_STRIKE, "spot": 110.0}) == 0.0
        assert bs_gamma(**{**_FORWARD_AT_STRIKE, "rate": 0.05}) == 0.0
        assert bs_gamma(**{**_FORWARD_AT_STRIKE, "time_to_maturity": 0.0}) == 0.0

    @pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
    def test_prices_at_zero_volatility_and_the_forward_at_the_strike_are_zero(
        self, dividend_yield: float
    ) -> None:
        kwargs = {**_FORWARD_AT_STRIKE, "rate": dividend_yield, "dividend_yield": dividend_yield}
        assert black_scholes_call(**kwargs) == 0.0
        assert black_scholes_put(**kwargs) == 0.0

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize(
        ("spot", "strike", "rate", "dividend_yield"),
        [(110.0, 100.0, 0.05, 0.0), (90.0, 100.0, 0.05, 0.0), (100.0, 95.0, -0.01, 0.04)],
    )
    def test_zero_volatility_off_the_forward_matches_the_infinite_d1_path(
        self, spot: float, strike: float, rate: float, dividend_yield: float
    ) -> None:
        t = 1.0
        kwargs = dict(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=0.0,
            time_to_maturity=t,
            dividend_yield=dividend_yield,
        )
        # The closed form evaluated with d1 = d2 = +-inf, and the put through parity.
        d1 = (np.log(spot / strike) + (rate - dividend_yield) * t) / np.float64(0.0)
        spot_leg = spot * np.exp(-dividend_yield * t)
        strike_leg = strike * np.exp(-rate * t)
        call_via_d1 = float(spot_leg * norm.cdf(d1) - strike_leg * norm.cdf(d1))
        put_via_parity = float(call_via_d1 - spot_leg + strike_leg)
        assert black_scholes_call(**kwargs) == call_via_d1
        assert black_scholes_put(**kwargs) == pytest.approx(put_via_parity, abs=1e-12)
        assert black_scholes_call(**kwargs) == max(spot_leg - strike_leg, 0.0)
        assert black_scholes_put(**kwargs) == max(strike_leg - spot_leg, 0.0)

    @pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
    def test_overflowing_price_raises_undefined(self) -> None:
        kwargs = dict(spot=100.0, strike=100.0, rate=-800.0, time_to_maturity=1.0)
        for volatility in (0.0, 0.2):
            with pytest.raises(UndefinedResultError, match="not finite"):
                black_scholes_put(volatility=volatility, **kwargs)

    @pytest.mark.parametrize("name", ["spot", "strike", "rate", "volatility", "time_to_maturity"])
    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_inputs_are_plain_value_errors(self, name: str, bad: float) -> None:
        kwargs = {**_ATM, name: bad}
        calls = (
            lambda: black_scholes_call(**kwargs),
            lambda: black_scholes_put(**kwargs),
            lambda: bs_delta(option_type="call", **kwargs),
            lambda: bs_gamma(**kwargs),
            lambda: bs_vega(**kwargs),
            lambda: bs_theta(option_type="put", **kwargs),
        )
        for call in calls:
            with pytest.raises(ValueError, match=f"{name} must be finite") as raised:
                call()
            assert not isinstance(raised.value, UndefinedResultError)

    def test_non_finite_dividend_yield_is_a_plain_value_error(self) -> None:
        with pytest.raises(ValueError, match="dividend_yield must be finite"):
            black_scholes_call(dividend_yield=math.nan, **_ATM)
        with pytest.raises(ValueError, match="dividend_yield must be finite"):
            bs_vega(dividend_yield=math.inf, **_ATM)
