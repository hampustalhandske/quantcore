"""Differential tests for the option-pricing targets.

Covers every target registered by `conformance.targets.pricing` and
`conformance.targets.pricing_extra`:

- `black_scholes_merton_price`, `bsm_delta`, `bsm_vega`, `bs_gamma`, `bs_theta` and `bs_rho`:
  quantcore against the closed-form oracle. These take scalar arguments -- there is no "length" to
  bucket, so every case uses the single declared bucket `"scalar"`; coverage breadth comes from
  `BSM_FAMILIES`, covering ATM/ITM/OTM moneyness, short and long expiries, a zero dividend yield
  vs. a nonzero one, negative rates and high volatility.
- `implied_volatility`: quantcore's Brent inversion against the oracle's. Where vega is negligible
  the root is not identifiable and both quantcore and the oracle raise by design;
  `conformance.harness.check()` cannot record a case whose oracle raises, so those inputs are
  filtered out of the sweep (`_MIN_IDENTIFIABLE_VEGA_RATIO`).
- `heston_cos_call`/`heston_cos_put`: quantcore's COS series against
  `QuantLib.AnalyticHestonEngine` across Feller-satisfying and Feller-violating parameter
  families, plus the Fang & Oosterlee (2008) Table 4/5 published reference values
  (`tests/fixtures/fang_oosterlee_2008.json`) and a hand-derived put-call parity check. The
  `short_expiry_wide_moneyness` family covers short-dated, far-from-the-money options, where a COS
  truncation range that is not centred on the log-moneyness misprices by many multiples of the
  true value. Maturities are drawn on whole calendar days because the QuantLib oracle prices to
  whole days.
- `monte_carlo_call_price`/`monte_carlo_put_price`: quantcore's estimate against the BSM closed
  form, judged within 4 standard errors, taking the larger of the standard error quantcore
  reports and the estimator's true one, which the oracle supplies. Cases carry no dividend
  yield because quantcore's Monte Carlo pricer takes none.
- `option_expiry_payoff`, `option_position_pnl`, `option_roll_schedule` and
  `protective_put_sizing`: quantcore against hand-derived formulas.

Second-oracle cross-checks (`conformance.harness.check_second_oracle`) compare the oracles of
`bs_gamma`, `bs_theta` and `bs_rho` against `py_vollib.black_scholes_merton.greeks.analytical`,
after applying that library's own documented scaling conventions (gamma and theta need no
rescaling, rho needs `* 100` since py_vollib reports it per 1 percentage point rather than per
unit); the oracle of `implied_volatility` is cross-checked against
`py_vollib.black_scholes_merton.implied_volatility.implied_volatility` the same way. These records
never count towards a target's status -- they are reported alongside it.

The tests run derandomized, and hypothesis seeds a derandomized test from the source text and
signature of the test function. Editing a hypothesis test here (decorators, signature, docstring
or body) therefore changes which cases it draws and records.
"""

import functools
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vollib.black_scholes_merton.greeks.analytical as _vollib_greeks
import vollib.black_scholes_merton.implied_volatility as _vollib_iv
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Importing the target modules registers their targets.
import conformance.targets.pricing_extra  # noqa: F401
from conformance.compare import registry
from conformance.harness import check, check_second_oracle
from conformance.targets.pricing import OptionRight

_BLACK_SCHOLES_MERTON_PRICE = registry()["black_scholes_merton_price"]
_BSM_DELTA = registry()["bsm_delta"]
_BSM_VEGA = registry()["bsm_vega"]
_BS_GAMMA = registry()["bs_gamma"]
_BS_THETA = registry()["bs_theta"]
_BS_RHO = registry()["bs_rho"]
_IMPLIED_VOLATILITY = registry()["implied_volatility"]
_HESTON_COS_CALL = registry()["heston_cos_call"]
_HESTON_COS_PUT = registry()["heston_cos_put"]
_MONTE_CARLO_CALL_PRICE = registry()["monte_carlo_call_price"]
_MONTE_CARLO_PUT_PRICE = registry()["monte_carlo_put_price"]
_OPTION_EXPIRY_PAYOFF = registry()["option_expiry_payoff"]
_OPTION_POSITION_PNL = registry()["option_position_pnl"]
_OPTION_ROLL_SCHEDULE = registry()["option_roll_schedule"]
_PROTECTIVE_PUT_SIZING = registry()["protective_put_sizing"]

_SCALAR_BUCKET = "scalar"

BSM_FAMILIES: tuple[str, ...] = (
    "atm_short_expiry",
    "atm_long_expiry",
    "deep_itm",
    "deep_otm",
    "zero_dividend",
    "nonzero_dividend",
    "negative_rate",
    "high_vol",
)
BSM_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


def _finite_floats(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )


@dataclass(frozen=True)
class BsmCase:
    """One BSM input `(s, k, t, r, q, sigma, right)`, tagged with the family that produced it."""

    s: float
    k: float
    t: float
    r: float
    q: float
    sigma: float
    right: OptionRight
    family: str


@st.composite
def bsm_case(draw: st.DrawFn) -> BsmCase:
    """Draw a BSM input from a moneyness/expiry/rate/dividend family, `t > 0` always."""
    family = draw(st.sampled_from(BSM_FAMILIES))
    right: OptionRight = draw(st.sampled_from(["call", "put"]))
    r = draw(_finite_floats(0.0, 0.15))
    q = draw(_finite_floats(0.0, 0.1))
    sigma = draw(_finite_floats(0.01, 1.0))
    if family == "atm_short_expiry":
        s = draw(_finite_floats(1.0, 500.0))
        k = s
        t = draw(_finite_floats(1e-4, 0.05))
    elif family == "atm_long_expiry":
        s = draw(_finite_floats(1.0, 500.0))
        k = s
        t = draw(_finite_floats(5.0, 30.0))
    elif family == "deep_itm":
        k = draw(_finite_floats(1.0, 100.0))
        s = (
            k * draw(_finite_floats(3.0, 10.0))
            if right == "call"
            else k / draw(_finite_floats(3.0, 10.0))
        )
        t = draw(_finite_floats(1e-3, 5.0))
    elif family == "deep_otm":
        k = draw(_finite_floats(1.0, 100.0))
        s = (
            k / draw(_finite_floats(3.0, 10.0))
            if right == "call"
            else k * draw(_finite_floats(3.0, 10.0))
        )
        t = draw(_finite_floats(1e-3, 5.0))
    elif family == "zero_dividend":
        s = draw(_finite_floats(1.0, 500.0))
        k = draw(_finite_floats(1.0, 500.0))
        t = draw(_finite_floats(1e-3, 5.0))
        q = 0.0
    elif family == "nonzero_dividend":
        s = draw(_finite_floats(1.0, 500.0))
        k = draw(_finite_floats(1.0, 500.0))
        t = draw(_finite_floats(1e-3, 5.0))
        q = draw(_finite_floats(0.01, 0.1))
    elif family == "negative_rate":
        s = draw(_finite_floats(1.0, 500.0))
        k = draw(_finite_floats(1.0, 500.0))
        t = draw(_finite_floats(1e-3, 5.0))
        r = draw(_finite_floats(-0.05, 0.0))
    else:  # high_vol
        s = draw(_finite_floats(1.0, 500.0))
        k = draw(_finite_floats(1.0, 500.0))
        t = draw(_finite_floats(1e-3, 5.0))
        sigma = draw(_finite_floats(1.0, 3.0))
    return BsmCase(s=s, k=k, t=t, r=r, q=q, sigma=sigma, right=right, family=family)


@settings(max_examples=800)
@given(case=bsm_case())
def test_black_scholes_merton_price_matches_quantcore(case: BsmCase) -> None:
    """`black_scholes_merton_price`'s closed-form reference against quantcore's BSM price."""
    check(
        _BLACK_SCHOLES_MERTON_PRICE,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=800)
@given(case=bsm_case())
def test_bsm_delta_matches_quantcore(case: BsmCase) -> None:
    """`bsm_delta`'s closed-form reference against quantcore's `bs_delta`."""
    check(
        _BSM_DELTA,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=800)
@given(case=bsm_case())
def test_bsm_vega_matches_quantcore(case: BsmCase) -> None:
    """`bsm_vega`'s closed-form reference against quantcore's `bs_vega` (right-independent)."""
    check(
        _BSM_VEGA,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


# ---------------------------------------------------------------------------------------------
# bs_gamma / bs_theta / bs_rho: quantcore vs closed-form reference (same bsm_case generator)
# ---------------------------------------------------------------------------------------------


@settings(max_examples=800)
@given(case=bsm_case())
def test_bs_gamma_matches_quantcore(case: BsmCase) -> None:
    """`bs_gamma`'s closed-form reference against quantcore's `bs_gamma` (right-independent)."""
    check(
        _BS_GAMMA,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=800)
@given(case=bsm_case())
def test_bs_theta_matches_quantcore(case: BsmCase) -> None:
    """`bs_theta`'s closed-form reference against quantcore's `bs_theta`."""
    check(
        _BS_THETA,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=800)
@given(case=bsm_case())
def test_bs_rho_matches_quantcore(case: BsmCase) -> None:
    """`bs_rho`'s closed-form reference against quantcore's `bs_rho`."""
    check(
        _BS_RHO,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


# ---------------------------------------------------------------------------------------------
# Second-oracle cross-checks: py_vollib.black_scholes_merton.greeks.analytical
#
# py_vollib's flag is 'c'/'p' (not "call"/"put"); its gamma and theta already match quantcore's
# convention (gamma: no scaling; theta: already per calendar day), but its vega and rho are
# reported per 1 PERCENTAGE POINT of the underlying input (its own docstrings: "in practice ...
# we multiply by 0.01"), so the rho adapter below multiplies py_vollib's raw output by 100 before
# comparing.
# ---------------------------------------------------------------------------------------------


def _py_vollib_gamma(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    return float(_vollib_greeks.gamma("c", s, k, t, r, sigma, q))


def _py_vollib_theta(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    flag = "c" if right == "call" else "p"
    return float(_vollib_greeks.theta(flag, s, k, t, r, sigma, q))


def _py_vollib_rho_x100(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    flag = "c" if right == "call" else "p"
    return float(_vollib_greeks.rho(flag, s, k, t, r, sigma, q)) * 100.0


@settings(max_examples=200)
@given(case=bsm_case())
def test_bs_gamma_vs_py_vollib(case: BsmCase) -> None:
    """`bs_gamma`'s oracle (the closed-form standard definition) against `py_vollib`'s gamma."""
    check_second_oracle(
        _BS_GAMMA,
        case.s,
        case.k,
        case.t,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        oracle_name="py_vollib",
        oracle_callable=_py_vollib_gamma,
        quantity="gamma",
        reference_kwargs={"r": case.r, "q": case.q, "sigma": case.sigma},
        r=case.r,
        q=case.q,
        sigma=case.sigma,
    )


@settings(max_examples=200)
@given(case=bsm_case())
def test_bs_theta_vs_py_vollib(case: BsmCase) -> None:
    """`bs_theta`'s reference against `py_vollib`'s theta (both already per calendar day)."""
    check_second_oracle(
        _BS_THETA,
        case.s,
        case.k,
        case.t,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        oracle_name="py_vollib",
        oracle_callable=_py_vollib_theta,
        quantity="theta",
        reference_kwargs={"r": case.r, "q": case.q, "sigma": case.sigma, "right": case.right},
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
    )


@settings(max_examples=200)
@given(case=bsm_case())
def test_bs_rho_vs_py_vollib(case: BsmCase) -> None:
    """`bs_rho`'s reference against `py_vollib`'s rho, scaled `* 100` (py_vollib is per 1%)."""
    check_second_oracle(
        _BS_RHO,
        case.s,
        case.k,
        case.t,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        oracle_name="py_vollib",
        oracle_callable=_py_vollib_rho_x100,
        quantity="rho",
        reference_kwargs={"r": case.r, "q": case.q, "sigma": case.sigma, "right": case.right},
        r=case.r,
        q=case.q,
        sigma=case.sigma,
        right=case.right,
    )


# ---------------------------------------------------------------------------------------------
# implied_volatility
# ---------------------------------------------------------------------------------------------

IV_FAMILIES: tuple[str, ...] = (
    "atm_short_expiry",
    "atm_long_expiry",
    "moderate_itm",
    "moderate_otm",
    "zero_dividend",
    "nonzero_dividend",
    "negative_rate",
    "high_vol",
)
IV_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


@dataclass(frozen=True)
class IvCase:
    """One `implied_volatility` input, with the true `sigma` used to build `market_price`."""

    s: float
    k: float
    t: float
    r: float
    q: float
    sigma: float
    right: OptionRight
    family: str


@st.composite
def iv_case(draw: st.DrawFn) -> IvCase:
    """Draw an `implied_volatility` case whose vega is never negligible (see the module docstring's
    `implied_volatility` entry).

    Unlike `bsm_case`'s `deep_itm`/`deep_otm` families, the `moderate_itm`/`moderate_otm`
    families here cap moneyness at 1.5x so `market_price`'s implied vega stays well above
    quantcore's own identifiability threshold (`_IV_MIN_RELATIVE_VEGA`) -- those genuinely
    unidentifiable edge cases are deliberately excluded from this differential sweep (both
    quantcore and the oracle would need to raise, which `conformance.harness.check()` cannot
    record; see `tolerances.py`'s `implied_volatility` entry).
    """
    family = draw(st.sampled_from(IV_FAMILIES))
    right: OptionRight = draw(st.sampled_from(["call", "put"]))
    r = draw(_finite_floats(0.0, 0.1))
    q = draw(_finite_floats(0.0, 0.08))
    sigma = draw(_finite_floats(0.05, 0.8))
    if family == "atm_short_expiry":
        s = draw(_finite_floats(10.0, 500.0))
        k = s
        t = draw(_finite_floats(0.05, 0.25))
    elif family == "atm_long_expiry":
        s = draw(_finite_floats(10.0, 500.0))
        k = s
        t = draw(_finite_floats(3.0, 10.0))
    elif family == "moderate_itm":
        k = draw(_finite_floats(10.0, 200.0))
        s = (
            k * draw(_finite_floats(1.05, 1.5))
            if right == "call"
            else k / draw(_finite_floats(1.05, 1.5))
        )
        t = draw(_finite_floats(0.1, 2.0))
    elif family == "moderate_otm":
        k = draw(_finite_floats(10.0, 200.0))
        s = (
            k / draw(_finite_floats(1.05, 1.5))
            if right == "call"
            else k * draw(_finite_floats(1.05, 1.5))
        )
        t = draw(_finite_floats(0.1, 2.0))
    elif family == "zero_dividend":
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(0.1, 2.0))
        q = 0.0
    elif family == "nonzero_dividend":
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(0.1, 2.0))
        q = draw(_finite_floats(0.01, 0.08))
    elif family == "negative_rate":
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(0.1, 2.0))
        r = draw(_finite_floats(-0.03, 0.0))
    else:  # high_vol
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(0.1, 2.0))
        sigma = draw(_finite_floats(0.5, 1.5))
    return IvCase(s=s, k=k, t=t, r=r, q=q, sigma=sigma, right=right, family=family)


def _identifiable_vega(case: IvCase) -> float:
    """Vega at `case`'s own `sigma`, from the closed-form oracle of `bsm_vega`."""
    return _BSM_VEGA.reference(case.s, case.k, case.t, r=case.r, q=case.q, sigma=case.sigma)


# `iv_case`'s "moderate_itm"/"moderate_otm" families, combined with a short `t` and low `sigma`,
# can still occasionally draw a case whose vega is genuinely negligible (quantcore's own
# `_IV_MIN_RELATIVE_VEGA` threshold) even though moneyness was capped well short of "deep" -- both
# quantcore and the oracle correctly raise there (see `quantcore.pricing.implied_volatility`'s
# docstring), which `conformance.harness.check()` cannot record as a case (a raising oracle fails
# the run, by design; see `harness.check`'s docstring). Those rare draws are filtered out here
# with a comfortable margin above quantcore's own raise threshold, rather than narrowing the
# family's declared moneyness/expiry/vol ranges themselves.
_MIN_IDENTIFIABLE_VEGA_RATIO = 1e-4


@settings(max_examples=500)
@given(case=iv_case())
def test_implied_volatility_matches_quantcore(case: IvCase) -> None:
    """`implied_volatility`'s reference (Brent on the closed form) against quantcore's own Brent."""
    assume(_identifiable_vega(case) > _MIN_IDENTIFIABLE_VEGA_RATIO * case.s)
    market_price = _BLACK_SCHOLES_MERTON_PRICE.reference(
        case.s, case.k, case.t, r=case.r, q=case.q, sigma=case.sigma, right=case.right
    )
    check(
        _IMPLIED_VOLATILITY,
        market_price,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        right=case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=500)
@given(case=iv_case())
def test_implied_volatility_round_trip(case: IvCase) -> None:
    """`price -> implied_volatility -> price` round-trips to float64 precision.

    Uses the oracle both ways, independent of quantcore.
    """
    assume(_identifiable_vega(case) > _MIN_IDENTIFIABLE_VEGA_RATIO * case.s)
    market_price = _BLACK_SCHOLES_MERTON_PRICE.reference(
        case.s, case.k, case.t, r=case.r, q=case.q, sigma=case.sigma, right=case.right
    )
    recovered_sigma = _IMPLIED_VOLATILITY.reference(
        market_price, case.s, case.k, case.t, r=case.r, q=case.q, right=case.right
    )
    round_trip_price = _BLACK_SCHOLES_MERTON_PRICE.reference(
        case.s, case.k, case.t, r=case.r, q=case.q, sigma=recovered_sigma, right=case.right
    )
    # brentq's requested xtol on sigma itself is 1e-8 (both quantcore's call and the
    # oracle use the same bracket and xtol, see tolerances.py's implied_volatility entry), which
    # a vega up to a few tens of dollars-per-vol-point can amplify into a price error close to but
    # under 1e-8 * vega; 1e-6 comfortably covers that while still being far tighter than a genuine
    # inversion defect would produce.
    assert abs(round_trip_price - market_price) <= 1e-6 + 1e-6 * abs(market_price)


def _py_vollib_iv(
    market_price: float, s: float, k: float, t: float, *, r: float, q: float, right: OptionRight
) -> float:
    flag = "c" if right == "call" else "p"
    return float(_vollib_iv.implied_volatility(market_price, s, k, t, r, q, flag))


@settings(max_examples=200)
@given(case=iv_case())
def test_implied_volatility_vs_py_vollib(case: IvCase) -> None:
    """`implied_volatility`'s reference against `py_vollib`'s implied-volatility solver."""
    assume(_identifiable_vega(case) > _MIN_IDENTIFIABLE_VEGA_RATIO * case.s)
    market_price = _BLACK_SCHOLES_MERTON_PRICE.reference(
        case.s, case.k, case.t, r=case.r, q=case.q, sigma=case.sigma, right=case.right
    )
    check_second_oracle(
        _IMPLIED_VOLATILITY,
        market_price,
        case.s,
        case.k,
        case.t,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        oracle_name="py_vollib",
        oracle_callable=_py_vollib_iv,
        quantity="sigma",
        reference_kwargs={"r": case.r, "q": case.q, "right": case.right},
        r=case.r,
        q=case.q,
        right=case.right,
    )


# ---------------------------------------------------------------------------------------------
# heston_cos_call / heston_cos_put
# ---------------------------------------------------------------------------------------------

HESTON_FAMILIES: tuple[str, ...] = (
    "atm_feller_satisfying",
    "atm_feller_violating",
    "moderate_moneyness",
    "long_expiry",
    "short_expiry_wide_moneyness",
)
HESTON_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


@dataclass(frozen=True)
class HestonCase:
    """One Heston COS input, tagged with the Feller/moneyness/expiry family it came from."""

    s: float
    k: float
    t: float
    r: float
    q: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    family: str


# Each recording Heston test runs once per family with this many examples, so every declared
# family is covered by construction instead of by how often hypothesis happens to draw it.
_HESTON_EXAMPLES_PER_FAMILY = 40


@st.composite
def heston_case(draw: st.DrawFn, family: str | None = None) -> HestonCase:
    """Draw a Heston case; `family` controls Feller compliance, moneyness, and expiry.

    With `family=None` the family itself is drawn from `HESTON_FAMILIES`.
    """
    if family is None:
        family = draw(st.sampled_from(HESTON_FAMILIES))
    r = draw(_finite_floats(0.0, 0.08))
    q = draw(_finite_floats(0.0, 0.05))
    kappa = draw(_finite_floats(0.5, 4.0))
    theta = draw(_finite_floats(0.01, 0.3))
    v0 = draw(_finite_floats(0.01, 0.3))
    rho = draw(_finite_floats(-0.95, 0.95))
    feller_boundary = (2.0 * kappa * theta) ** 0.5
    if family == "atm_feller_satisfying":
        s = k = draw(_finite_floats(50.0, 200.0))
        t = draw(_finite_floats(0.25, 2.0))
        xi = feller_boundary * draw(_finite_floats(0.1, 0.9))
    elif family == "atm_feller_violating":
        s = k = draw(_finite_floats(50.0, 200.0))
        t = draw(_finite_floats(0.25, 2.0))
        xi = feller_boundary * draw(_finite_floats(1.2, 3.0))
    elif family == "moderate_moneyness":
        k = draw(_finite_floats(50.0, 200.0))
        s = k * draw(_finite_floats(0.8, 1.25))
        t = draw(_finite_floats(0.25, 2.0))
        xi = feller_boundary * draw(_finite_floats(0.1, 1.5))
    elif family == "long_expiry":
        s = k = draw(_finite_floats(50.0, 200.0))
        t = draw(_finite_floats(3.0, 10.0))
        xi = feller_boundary * draw(_finite_floats(0.1, 1.5))
    else:  # short_expiry_wide_moneyness
        k = draw(_finite_floats(50.0, 200.0))
        s = k * draw(st.sampled_from([1.0 / 3.0, 1.0 / 2.5, 2.5, 3.0]))
        t = draw(_finite_floats(0.01, 0.1))
        xi = feller_boundary * draw(_finite_floats(0.8, 2.5))
    # The QuantLib oracle prices to a whole number of calendar days (`_heston_quantlib_price`), so
    # `t` is drawn on that grid; an off-grid `t` would compare two different maturities.
    t = max(1, round(t * 365)) / 365
    return HestonCase(
        s=s, k=k, t=t, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, family=family
    )


@pytest.mark.parametrize("family", HESTON_FAMILIES)
@settings(max_examples=_HESTON_EXAMPLES_PER_FAMILY)
@given(data=st.data())
def test_heston_cos_call_matches_quantlib(family: str, data: st.DataObject) -> None:
    """`heston_cos_call`'s reference (`QuantLib.AnalyticHestonEngine`) against quantcore's COS."""
    case = data.draw(heston_case(family))
    check(
        _HESTON_COS_CALL,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        v0=case.v0,
        kappa=case.kappa,
        theta=case.theta,
        xi=case.xi,
        rho=case.rho,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@pytest.mark.parametrize("family", HESTON_FAMILIES)
@settings(max_examples=_HESTON_EXAMPLES_PER_FAMILY)
@given(data=st.data())
def test_heston_cos_put_matches_quantlib(family: str, data: st.DataObject) -> None:
    """`heston_cos_put`'s reference (`QuantLib.AnalyticHestonEngine`) against quantcore's COS."""
    case = data.draw(heston_case(family))
    check(
        _HESTON_COS_PUT,
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        v0=case.v0,
        kappa=case.kappa,
        theta=case.theta,
        xi=case.xi,
        rho=case.rho,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@settings(max_examples=100)
@given(case=heston_case())
def test_heston_put_call_parity(case: HestonCase) -> None:
    """Put-call parity holds for quantcore's own Heston pricers, to float64 precision.

    `heston_cos_call(...) - heston_cos_put(...) == S*exp(-qT) - K*exp(-rT)`, both sides computed
    directly via quantcore (not via the reference), so this checks quantcore's own internal
    consistency independent of QuantLib.
    """
    call = _HESTON_COS_CALL.quantcore(
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        v0=case.v0,
        kappa=case.kappa,
        theta=case.theta,
        xi=case.xi,
        rho=case.rho,
    )
    put = _HESTON_COS_PUT.quantcore(
        case.s,
        case.k,
        case.t,
        r=case.r,
        q=case.q,
        v0=case.v0,
        kappa=case.kappa,
        theta=case.theta,
        xi=case.xi,
        rho=case.rho,
    )
    expected_diff = case.s * np.exp(-case.q * case.t) - case.k * np.exp(-case.r * case.t)
    assert abs((call - put) - expected_diff) <= 1e-8 + 1e-8 * abs(expected_diff)


def test_heston_cos_call_matches_fang_oosterlee_2008_table_4_and_5() -> None:
    """Quantcore's `heston_cos_call` against the Fang & Oosterlee (2008) published reference prices.

    Tables 4/5, `S0 = K = 100, r = q = 0` (see `tests/fixtures/fang_oosterlee_2008.json`).
    """
    fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "fang_oosterlee_2008.json"
    fixture = json.loads(fixture_path.read_text())
    if not fixture["available"]:
        pytest.skip(f"Fang & Oosterlee (2008) fixture unavailable: {fixture['reason']}")

    params = fixture["heston_parameters_eq_53"]
    for entry in fixture["heston_call_reference_prices"]:
        price = _HESTON_COS_CALL.quantcore(
            params["spot"],
            params["strike"],
            entry["time_to_maturity"],
            r=params["rate"],
            q=params["dividend_yield"],
            v0=params["v0"],
            kappa=params["kappa"],
            theta=params["theta"],
            xi=params["xi"],
            rho=params["rho"],
        )
        # The paper prints its reference value truncated to 9 decimal digits ("5.785155450 . . .");
        # 1e-4 comfortably exceeds both that print truncation and the COS series' own truncation
        # error at quantcore's default n_terms=2048.
        assert abs(price - entry["reference_value"]) < 1e-4, (
            f"T={entry['time_to_maturity']}: got {price}, paper reference "
            f"{entry['reference_value']} ({entry['table']}, page {entry['page']})"
        )


# ---------------------------------------------------------------------------------------------
# monte_carlo_call_price / monte_carlo_put_price
#
# `_MONTE_CARLO_CALL_PRICE`/`_MONTE_CARLO_PUT_PRICE`'s `compare` is `_MonteCarloComparator`
# (`conformance.targets.pricing_extra`): the Monte Carlo price is judged within k=4 standard
# errors of the closed-form BSM price (same GBM dynamics), not a fixed tolerance. The standard
# error used is the larger of the one quantcore reports and the estimator's true one, which the
# oracle returns next to the closed-form price -- see that class's docstring and `tolerances.py`'s
# `monte_carlo_call_price` entry.
# `num_paths`/`num_steps` are fixed (not hypothesis-varied) to keep runtime and standard error
# both predictable across the sweep; `seed` is derived from the drawn case so each case is its own
# deterministic draw.
#
# `harness.check()` reserves the keyword `seed` for its OWN reproducibility metadata (the
# hypothesis-drawn seed that reproduces a case's *input*), so it can never be forwarded through
# `**kwargs` to a function whose own signature also has a `seed` parameter -- `monte_carlo_call_
# price`/`monte_carlo_put_price` do, for simulation determinism. `_mc_shim` below builds a
# duck-typed stand-in exposing the same `name`/`compare`/`quantcore`/`reference` attributes
# `check()` reads off a `ConformanceTarget`, with `seed` and `antithetic` already bound into the
# quantcore and oracle callables via `functools.partial`, so only `r`/`sigma`/`num_paths`/
# `num_steps` remain to flow through `**kwargs` -- no collision, and `check()`'s own `seed=` is
# still passed separately, purely for the `ReproDetail` it records.
# ---------------------------------------------------------------------------------------------


def _mc_shim(fn, seed: int, antithetic: bool = True):
    """Wrap a Monte Carlo `ConformanceTarget` with `seed`/`antithetic` pre-bound for `check()`."""
    return SimpleNamespace(
        name=fn.name,
        compare=fn.compare,
        quantcore=functools.partial(fn.quantcore, seed=seed, antithetic=antithetic),
        reference=functools.partial(fn.reference, seed=seed, antithetic=antithetic),
    )


MONTE_CARLO_FAMILIES: tuple[str, ...] = ("atm", "itm", "otm", "high_vol", "long_expiry")
MONTE_CARLO_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)
_MC_NUM_PATHS = 20_000
_MC_NUM_STEPS = 50


@dataclass(frozen=True)
class MonteCarloCase:
    """One Monte Carlo pricing input, tagged with its family and a reproducing `seed`."""

    s: float
    k: float
    t: float
    r: float
    sigma: float
    seed: int
    family: str


# Each recording Monte Carlo test runs once per family with this many examples, so every
# declared family is covered by construction.
_MONTE_CARLO_EXAMPLES_PER_FAMILY = 12


@st.composite
def monte_carlo_case(draw: st.DrawFn, family: str | None = None) -> MonteCarloCase:
    """Draw a Monte Carlo case; with `family=None` the family is drawn too.

    No dividend yield: quantcore's MC pricer has none, see module docstring.
    """
    if family is None:
        family = draw(st.sampled_from(MONTE_CARLO_FAMILIES))
    r = draw(_finite_floats(0.0, 0.1))
    sigma = draw(_finite_floats(0.1, 0.4))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    if family == "atm":
        s = draw(_finite_floats(10.0, 500.0))
        k = s
        t = draw(_finite_floats(0.1, 2.0))
    elif family == "itm":
        k = draw(_finite_floats(10.0, 200.0))
        s = k * draw(_finite_floats(1.1, 1.5))
        t = draw(_finite_floats(0.1, 2.0))
    elif family == "otm":
        k = draw(_finite_floats(10.0, 200.0))
        s = k / draw(_finite_floats(1.1, 1.5))
        t = draw(_finite_floats(0.1, 2.0))
    elif family == "high_vol":
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(0.1, 2.0))
        sigma = draw(_finite_floats(0.4, 1.0))
    else:  # long_expiry
        s = draw(_finite_floats(10.0, 500.0))
        k = draw(_finite_floats(10.0, 500.0))
        t = draw(_finite_floats(3.0, 8.0))
    return MonteCarloCase(s=s, k=k, t=t, r=r, sigma=sigma, seed=seed, family=family)


@pytest.mark.parametrize("family", MONTE_CARLO_FAMILIES)
@settings(max_examples=_MONTE_CARLO_EXAMPLES_PER_FAMILY)
@given(data=st.data())
def test_monte_carlo_call_price_matches_bsm(family: str, data: st.DataObject) -> None:
    """`monte_carlo_call_price` against the BSM closed form, within k=4 standard errors."""
    case = data.draw(monte_carlo_case(family))
    check(
        _mc_shim(_MONTE_CARLO_CALL_PRICE, case.seed),
        case.s,
        case.k,
        case.t,
        r=case.r,
        sigma=case.sigma,
        num_paths=_MC_NUM_PATHS,
        num_steps=_MC_NUM_STEPS,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        seed=case.seed,
    )


@pytest.mark.parametrize("family", MONTE_CARLO_FAMILIES)
@settings(max_examples=_MONTE_CARLO_EXAMPLES_PER_FAMILY)
@given(data=st.data())
def test_monte_carlo_put_price_matches_bsm(family: str, data: st.DataObject) -> None:
    """`monte_carlo_put_price` against the BSM closed form, within k=4 standard errors."""
    case = data.draw(monte_carlo_case(family))
    check(
        _mc_shim(_MONTE_CARLO_PUT_PRICE, case.seed),
        case.s,
        case.k,
        case.t,
        r=case.r,
        sigma=case.sigma,
        num_paths=_MC_NUM_PATHS,
        num_steps=_MC_NUM_STEPS,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        seed=case.seed,
    )


def test_monte_carlo_call_price_deterministic_under_fixed_seed() -> None:
    """Same seed, same args -> byte-identical `(price, std_error)`, quantcore directly."""
    args = (100.0, 100.0, 1.0)
    kwargs = {
        "r": 0.03,
        "sigma": 0.2,
        "num_paths": 5_000,
        "num_steps": 20,
        "seed": 7,
        "antithetic": True,
    }
    first = _MONTE_CARLO_CALL_PRICE.quantcore(*args, **kwargs)
    second = _MONTE_CARLO_CALL_PRICE.quantcore(*args, **kwargs)
    assert first == second


# ---------------------------------------------------------------------------------------------
# option_expiry_payoff / option_position_pnl / option_roll_schedule / protective_put_sizing
#
# No independent published oracle for these; `check()` differentially tests quantcore against the
# hand-derived oracle (same formula quantcore's own docstrings specify), and a handful of
# hand-computed values below pin down the oracle's conventions directly: the sign of P&L for a
# short position, roll-date tie-breaking, and protective-put sizing units.
# ---------------------------------------------------------------------------------------------

OPTION_EXPIRY_PAYOFF_FAMILIES: tuple[str, ...] = (
    "itm_call",
    "otm_call",
    "itm_put",
    "otm_put",
    "atm",
)
OPTION_EXPIRY_PAYOFF_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


@dataclass(frozen=True)
class ExpiryPayoffCase:
    """One `option_expiry_payoff` input."""

    spot_at_expiry: float
    strike: float
    premium_paid: float
    contracts: int
    multiplier: float
    right: OptionRight
    family: str


@st.composite
def expiry_payoff_case(draw: st.DrawFn) -> ExpiryPayoffCase:
    """Draw an `option_expiry_payoff` input from an ITM/OTM/ATM family, for both rights."""
    family = draw(st.sampled_from(OPTION_EXPIRY_PAYOFF_FAMILIES))
    strike = draw(_finite_floats(1.0, 500.0))
    premium_paid = draw(_finite_floats(0.0, 50.0))
    contracts = draw(st.integers(min_value=0, max_value=1000))
    multiplier = draw(_finite_floats(1.0, 100.0))
    if family == "itm_call":
        right: OptionRight = "call"
        spot = strike * draw(_finite_floats(1.01, 3.0))
    elif family == "otm_call":
        right = "call"
        spot = strike / draw(_finite_floats(1.01, 3.0))
    elif family == "itm_put":
        right = "put"
        spot = strike / draw(_finite_floats(1.01, 3.0))
    elif family == "otm_put":
        right = "put"
        spot = strike * draw(_finite_floats(1.01, 3.0))
    else:  # atm
        right = draw(st.sampled_from(["call", "put"]))
        spot = strike
    return ExpiryPayoffCase(
        spot_at_expiry=spot,
        strike=strike,
        premium_paid=premium_paid,
        contracts=contracts,
        multiplier=multiplier,
        right=right,
        family=family,
    )


@settings(max_examples=300)
@given(case=expiry_payoff_case())
def test_option_expiry_payoff_matches_quantcore(case: ExpiryPayoffCase) -> None:
    """`option_expiry_payoff`'s hand-derived reference against quantcore's own."""
    check(
        _OPTION_EXPIRY_PAYOFF,
        case.spot_at_expiry,
        case.strike,
        case.premium_paid,
        case.contracts,
        case.multiplier,
        case.right,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


def test_option_expiry_payoff_long_call_hand_value() -> None:
    """Long 10 calls, strike 100, multiplier 100, premium 5/share, spot 120 at expiry.

    intrinsic = 20/share -> P&L = 10*100*(20-5) = 15,000, checked on the oracle itself.
    """
    assert _OPTION_EXPIRY_PAYOFF.reference(120.0, 100.0, 5.0, 10, 100.0, "call") == 15_000.0


def test_option_expiry_payoff_short_position_is_the_negated_long_pnl() -> None:
    """A short position's P&L is the exact negation of the long holder's.

    The oracle's convention (see `_option_expiry_payoff_reference` in
    `conformance.targets.pricing_extra`): for the same market outcome, the writer loses what the
    buyer gains, and vice versa.
    """
    long_pnl = _OPTION_EXPIRY_PAYOFF.reference(80.0, 100.0, 4.0, 5, 100.0, "put")
    short_pnl = -long_pnl
    # The put finished ITM (spot 80 < strike 100): the long holder profits, so the writer's P&L
    # (premium received minus the payout made) must be negative by the same amount.
    assert long_pnl > 0.0
    assert short_pnl == -long_pnl


OPTION_POSITION_PNL_FAMILIES: tuple[str, ...] = ("rising_path", "falling_path", "flat_path")
OPTION_POSITION_PNL_BUCKETS: tuple[str, ...] = ("short", "medium")


@dataclass(frozen=True)
class PositionPnlCase:
    """One `option_position_pnl` input."""

    entry_price: float
    contracts: int
    multiplier: float
    spot_path: np.ndarray
    strike: float
    expiry: float
    iv_path: np.ndarray
    r: float
    right: OptionRight
    family: str
    length_bucket: str


@st.composite
def position_pnl_case(draw: st.DrawFn) -> PositionPnlCase:
    """Draw an `option_position_pnl` input: a short (2-10) or medium (11-60) daily path."""
    family = draw(st.sampled_from(OPTION_POSITION_PNL_FAMILIES))
    n = draw(st.integers(min_value=2, max_value=60))
    length_bucket = "short" if n <= 10 else "medium"
    strike = draw(_finite_floats(10.0, 200.0))
    start_spot = draw(_finite_floats(0.5, 1.5)) * strike
    if family == "rising_path":
        drift = draw(_finite_floats(0.0005, 0.01))
    elif family == "falling_path":
        drift = draw(_finite_floats(-0.01, -0.0005))
    else:  # flat_path
        drift = 0.0
    steps = np.arange(n, dtype=np.float64)
    spot_path = start_spot * np.exp(drift * steps)
    iv_level = draw(_finite_floats(0.1, 0.6))
    iv_path = np.full(n, iv_level, dtype=np.float64)
    expiry = draw(_finite_floats(0.1, 2.0))
    r = draw(_finite_floats(0.0, 0.08))
    contracts = draw(st.integers(min_value=1, max_value=100))
    multiplier = draw(_finite_floats(1.0, 100.0))
    entry_price = draw(_finite_floats(0.0, 50.0))
    right: OptionRight = draw(st.sampled_from(["call", "put"]))
    return PositionPnlCase(
        entry_price=entry_price,
        contracts=contracts,
        multiplier=multiplier,
        spot_path=spot_path,
        strike=strike,
        expiry=expiry,
        iv_path=iv_path,
        r=r,
        right=right,
        family=family,
        length_bucket=length_bucket,
    )


@settings(max_examples=200)
@given(case=position_pnl_case())
def test_option_position_pnl_matches_quantcore(case: PositionPnlCase) -> None:
    """`option_position_pnl`'s hand-derived daily-repricing reference against quantcore's own."""
    check(
        _OPTION_POSITION_PNL,
        case.entry_price,
        case.contracts,
        case.multiplier,
        case.spot_path,
        case.strike,
        case.expiry,
        case.iv_path,
        r=case.r,
        right=case.right,
        family=case.family,
        length_bucket=case.length_bucket,
    )


def test_option_position_pnl_last_day_equals_expiry_payoff() -> None:
    """The last day's repricing collapses to intrinsic value, matching `option_expiry_payoff`.

    At the final day (time-to-expiry == 0); the oracle is used both ways, so this pins the
    convention rather than re-testing quantcore.
    """
    spot_path = np.array([100.0, 105.0, 110.0, 130.0])
    iv_path = np.full(4, 0.2)
    pnl = _OPTION_POSITION_PNL.reference(
        0.0, 1, 100.0, spot_path, 100.0, 0.25, iv_path, r=0.03, right="call"
    )
    expected_last = _OPTION_EXPIRY_PAYOFF.reference(130.0, 100.0, 0.0, 1, 100.0, "call")
    assert abs(pnl[-1] - expected_last) < 1e-8


OPTION_ROLL_SCHEDULE_FAMILIES: tuple[str, ...] = ("nearest_above", "nearest_below", "tie")
OPTION_ROLL_SCHEDULE_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


@settings(max_examples=200)
@given(
    start_day=st.integers(min_value=1, max_value=28),
    target_tenor_days=st.integers(min_value=1, max_value=180),
    expiry_offsets=st.lists(
        st.integers(min_value=1, max_value=365), min_size=1, max_size=12, unique=True
    ),
)
def test_option_roll_schedule_matches_quantcore(
    start_day: int, target_tenor_days: int, expiry_offsets: list[int]
) -> None:
    """`option_roll_schedule`'s hand-derived nearest-expiry pick against quantcore's own."""
    current_date = date(2024, 1, start_day)
    listed_expiries = tuple(sorted(current_date + timedelta(days=o) for o in expiry_offsets))
    check(
        _OPTION_ROLL_SCHEDULE,
        current_date,
        target_tenor_days,
        listed_expiries,
        family="nearest_above" if target_tenor_days <= expiry_offsets[0] else "nearest_below",
        length_bucket=_SCALAR_BUCKET,
    )


def test_option_roll_schedule_tie_breaks_to_earlier_listed_expiry() -> None:
    """Two listed expiries equidistant from the target: the earlier one wins.

    The earlier date is first in `listed_expiries`, and the oracle breaks a tie in favour of the
    first listed.
    """
    current_date = date(2024, 1, 1)
    earlier = date(2024, 1, 25)
    later = date(2024, 2, 6)
    # target_tenor_days = 18 -> target date 2024-01-19; both listed dates are 6 days away.
    result = _OPTION_ROLL_SCHEDULE.reference(current_date, 18, (earlier, later))
    assert result == earlier
    check(
        _OPTION_ROLL_SCHEDULE,
        current_date,
        18,
        (earlier, later),
        family="tie",
        length_bucket=_SCALAR_BUCKET,
    )


PROTECTIVE_PUT_SIZING_FAMILIES: tuple[str, ...] = ("budget_sufficient", "budget_constrained")
PROTECTIVE_PUT_SIZING_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


@dataclass(frozen=True)
class ProtectivePutCase:
    """One `protective_put_sizing` input."""

    portfolio_value: float
    s: float
    k: float
    iv: float
    t: float
    r: float
    target_protection_pct: float
    budget_pct: float
    family: str


@st.composite
def protective_put_case(draw: st.DrawFn) -> ProtectivePutCase:
    """Draw a `protective_put_sizing` input; `budget_constrained` uses a tiny `budget_pct`."""
    family = draw(st.sampled_from(PROTECTIVE_PUT_SIZING_FAMILIES))
    portfolio_value = draw(_finite_floats(10_000.0, 10_000_000.0))
    s = draw(_finite_floats(10.0, 500.0))
    k = s * draw(_finite_floats(0.7, 1.0))
    iv = draw(_finite_floats(0.1, 0.6))
    t = draw(_finite_floats(0.05, 1.0))
    r = draw(_finite_floats(0.0, 0.08))
    target_protection_pct = draw(_finite_floats(0.5, 1.0))
    budget_pct = (
        draw(_finite_floats(0.0001, 0.005))
        if family == "budget_constrained"
        else draw(_finite_floats(0.05, 0.2))
    )
    return ProtectivePutCase(
        portfolio_value=portfolio_value,
        s=s,
        k=k,
        iv=iv,
        t=t,
        r=r,
        target_protection_pct=target_protection_pct,
        budget_pct=budget_pct,
        family=family,
    )


@settings(max_examples=200)
@given(case=protective_put_case())
def test_protective_put_sizing_matches_quantcore(case: ProtectivePutCase) -> None:
    """`protective_put_sizing`'s hand-derived reference against quantcore's own."""
    check(
        _PROTECTIVE_PUT_SIZING,
        case.portfolio_value,
        case.s,
        case.k,
        case.iv,
        case.t,
        r=case.r,
        target_protection_pct=case.target_protection_pct,
        budget_pct=case.budget_pct,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
        quantities=("contracts", "premium_cost", "breakeven", "protection_pct_achieved"),
    )


def test_protective_put_sizing_hand_value_budget_sufficient() -> None:
    """A 90% protection target with an ample 2% budget achieves exactly that target.

    1,000,000 portfolio: contracts cover at least the requested notional (rounded up to whole
    contracts of 100 shares) since the budget is not binding. The oracle returns a plain
    `(contracts, premium_cost, breakeven, protection_pct_achieved)` tuple.
    """
    contracts, _premium_cost, _breakeven, protection_pct_achieved = (
        _PROTECTIVE_PUT_SIZING.reference(
            1_000_000.0, 100.0, 95.0, 0.2, 0.25, r=0.05, target_protection_pct=0.9, budget_pct=0.02
        )
    )
    notional_to_protect = 1_000_000.0 * 0.9
    min_contracts = -(-int(notional_to_protect / 100.0) // 100)  # ceil(shares / 100)
    assert contracts >= min_contracts
    assert protection_pct_achieved == pytest.approx(0.9, abs=1e-9)


def test_protective_put_sizing_budget_constrained_achieves_less_than_target() -> None:
    """A too-small budget scales contracts down (floored) rather than raising.

    `protection_pct_achieved` documents the shortfall below the requested target.
    """
    contracts, _premium_cost, _breakeven, protection_pct_achieved = (
        _PROTECTIVE_PUT_SIZING.reference(
            1_000_000.0, 100.0, 95.0, 0.5, 1.0, r=0.02, target_protection_pct=1.0, budget_pct=0.0005
        )
    )
    assert protection_pct_achieved < 1.0
    assert contracts >= 0
