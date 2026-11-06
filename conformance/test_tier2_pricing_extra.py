"""Differential tests for the option-pricing targets.

Covers every target registered by `conformance.targets.pricing` and
`conformance.targets.pricing_extra`:

- `black_scholes_merton_price`, `bsm_delta`, `bsm_vega`, `bs_gamma` and `bs_theta`:
  quantcore against the closed-form oracle. These take scalar arguments -- there is no "length" to
  bucket, so every case uses the single declared bucket `"scalar"`; coverage breadth comes from
  `BSM_FAMILIES`, covering ATM/ITM/OTM moneyness, short and long expiries, a zero dividend yield
  vs. a nonzero one, negative rates and high volatility.
- `option_roll_schedule` and `protective_put_sizing`: quantcore against hand-derived formulas.

Second-oracle cross-checks (`conformance.harness.check_second_oracle`) compare the oracles of
`bs_gamma` and `bs_theta` against `py_vollib.black_scholes_merton.greeks.analytical`
(gamma and theta need no rescaling). These records never count towards a target's status --
they are reported alongside it.

The tests run derandomized, and hypothesis seeds a derandomized test from the source text and
signature of the test function. Editing a hypothesis test here (decorators, signature, docstring
or body) therefore changes which cases it draws and records.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pytest
import vollib.black_scholes_merton.greeks.analytical as _vollib_greeks
from hypothesis import given, settings
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
# bs_gamma / bs_theta: quantcore vs closed-form reference (same bsm_case generator)
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


# ---------------------------------------------------------------------------------------------
# Second-oracle cross-checks: py_vollib.black_scholes_merton.greeks.analytical
#
# py_vollib's flag is 'c'/'p' (not "call"/"put"); its gamma and theta already match quantcore's
# convention (gamma: no scaling; theta: already per calendar day).
# ---------------------------------------------------------------------------------------------


def _py_vollib_gamma(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    return float(_vollib_greeks.gamma("c", s, k, t, r, sigma, q))


def _py_vollib_theta(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    flag = "c" if right == "call" else "p"
    return float(_vollib_greeks.theta(flag, s, k, t, r, sigma, q))


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


# ---------------------------------------------------------------------------------------------
# option_roll_schedule / protective_put_sizing
#
# No independent published oracle for these; `check()` differentially tests quantcore against the
# hand-derived oracle (same formula quantcore's own docstrings specify), and a handful of
# hand-computed values below pin down the oracle's conventions directly: roll-date tie-breaking
# and protective-put sizing units.
# ---------------------------------------------------------------------------------------------


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
