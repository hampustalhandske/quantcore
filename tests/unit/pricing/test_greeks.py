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

    def test_price_at_intrinsic_value_raises_not_returns_boundary(self) -> None:
        # C14 contract: a price exactly at intrinsic value implies sigma=0
        # (which the search domain excludes -- _IV_LOWER_BOUND is > 0), so
        # this must raise rather than silently return the lower boundary
        # value as if it were a real volatility.
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=10.0,  # exactly spot - strike
                spot=110.0,
                strike=100.0,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="call",
            )

    @pytest.mark.parametrize(("spot", "strike"), [(100.0, 60.0), (100.0, 160.0)])
    def test_moderately_deep_moneyness_round_trips(self, spot: float, strike: float) -> None:
        # Moneyness deep enough to matter (0.6x / 1.6x) but not so extreme
        # that vega is degenerate (see the "negligible vega" test below for
        # that boundary case) -- a real round trip must still work here.
        true_sigma = 0.3
        price = black_scholes_call(
            spot=spot, strike=strike, rate=0.05, volatility=true_sigma, time_to_maturity=1.0
        )
        recovered = implied_volatility(
            market_price=price,
            spot=spot,
            strike=strike,
            rate=0.05,
            time_to_maturity=1.0,
            option_type="call",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-4)

    @pytest.mark.parametrize(("spot", "strike"), [(100.0, 1.0), (100.0, 1000.0)])
    def test_extreme_moneyness_with_negligible_vega_raises(
        self, spot: float, strike: float
    ) -> None:
        # spot/strike = 100 or 1/100: vega is ~0 in float64 at these
        # parameters, so the price equation has no numerically identifiable
        # root -- must raise, not silently return a boundary value.
        true_sigma = 0.3
        price = black_scholes_call(
            spot=spot, strike=strike, rate=0.05, volatility=true_sigma, time_to_maturity=1.0
        )
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=price,
                spot=spot,
                strike=strike,
                rate=0.05,
                time_to_maturity=1.0,
                option_type="call",
            )

    def test_very_short_expiry_round_trips_or_raises_explicitly(self) -> None:
        true_sigma = 0.3
        time_to_maturity = 1.0 / 365.0  # one calendar day
        price = black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=true_sigma,
            time_to_maturity=time_to_maturity,
        )
        recovered = implied_volatility(
            market_price=price,
            spot=100.0,
            strike=100.0,
            rate=0.05,
            time_to_maturity=time_to_maturity,
            option_type="call",
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-4)


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

    def test_implied_volatility_round_trips_with_dividend_yield(self) -> None:
        true_sigma = 0.22
        price = black_scholes_call(
            **{**_DIV_KWARGS, "volatility": true_sigma}, dividend_yield=_DIV_Q
        )
        recovered = implied_volatility(
            market_price=price,
            spot=_DIV_KWARGS["spot"],
            strike=_DIV_KWARGS["strike"],
            rate=_DIV_KWARGS["rate"],
            time_to_maturity=_DIV_KWARGS["time_to_maturity"],
            option_type="call",
            dividend_yield=_DIV_Q,
        )
        assert recovered == pytest.approx(true_sigma, abs=1e-6)
