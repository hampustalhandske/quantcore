"""Tier-1 differential tests: execution costs against a hand-derived closed-form reference.

Covers `square_root_market_impact`, `short_borrow_cost`, `margin_financing_cost` and
`futures_roll_yield`. Unlike the performance and statistics targets, these take scalar arguments,
not a series -- there
is no "length" to bucket. `conformance.generators.length_bucket_for` and the `short`/`medium`/
`long` coverage buckets it defines describe series length specifically and do not apply here, so
each function declares a single length bucket, `"scalar"`, and every recorded case uses it, which
satisfies the coverage gate (`build_report.DECLARED_COVERAGE`). Coverage breadth for these
functions instead comes from the declared *families* below, which cover the magnitude and
participation-rate ranges the contract calls for.
"""

import math
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

import quantcore.execution as qc_execution
from conformance.compare import registry
from conformance.edge_cases import observe
from conformance.harness import check
from conformance.targets.execution import (
    futures_roll_yield,
    margin_financing_cost,
    short_borrow_cost,
    square_root_market_impact,
)

_SQUARE_ROOT_MARKET_IMPACT = registry()["square_root_market_impact"]
_SHORT_BORROW_COST = registry()["short_borrow_cost"]
_MARGIN_FINANCING_COST = registry()["margin_financing_cost"]
_FUTURES_ROLL_YIELD = registry()["futures_roll_yield"]

_SCALAR_BUCKET = "scalar"

# Declared coverage: `build_report.py` fails the build if any declared family or length bucket
# ends up with zero recorded cases.
SQUARE_ROOT_MARKET_IMPACT_FAMILIES: tuple[str, ...] = (
    "typical",
    "tiny_magnitude",
    "huge_magnitude",
    "zero_vol",
    "tiny_participation",
    "huge_participation",
)
SQUARE_ROOT_MARKET_IMPACT_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)

SHORT_BORROW_COST_FAMILIES: tuple[str, ...] = (
    "typical",
    "tiny_magnitude",
    "huge_magnitude",
    "zero_days",
    "long_horizon",
)
SHORT_BORROW_COST_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)

MARGIN_FINANCING_COST_FAMILIES: tuple[str, ...] = (
    "typical",
    "tiny_magnitude",
    "huge_magnitude",
    "zero_days",
    "long_horizon",
)
MARGIN_FINANCING_COST_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)

FUTURES_ROLL_YIELD_FAMILIES: tuple[str, ...] = (
    "backwardation",
    "contango",
    "flat",
    "tiny_price",
    "huge_price",
    "short_days_to_roll",
    "long_days_to_roll",
)
FUTURES_ROLL_YIELD_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)

_DAYS_PER_10_YEARS = 3650


def _finite_floats(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )


@dataclass(frozen=True)
class MarketImpactCase:
    """One `square_root_market_impact` input, tagged with the family that produced it."""

    trade_size: float
    adv: float
    daily_vol: float
    coefficient: float
    family: str


@st.composite
def market_impact_case(draw: st.DrawFn) -> MarketImpactCase:
    """Draw a `square_root_market_impact` input from a magnitude/participation-rate family."""
    family = draw(
        st.sampled_from(
            [
                "typical",
                "tiny_magnitude",
                "huge_magnitude",
                "zero_vol",
                "tiny_participation",
                "huge_participation",
            ]
        )
    )
    coefficient = draw(_finite_floats(0.0, 5.0))
    if family == "typical":
        adv = draw(_finite_floats(1.0, 1e8))
        participation = draw(_finite_floats(1e-4, 2.0))
        daily_vol = draw(_finite_floats(0.0, 0.5))
    elif family == "tiny_magnitude":
        adv = draw(_finite_floats(1e-6, 1e-3))
        participation = draw(_finite_floats(1e-6, 2.0))
        daily_vol = draw(_finite_floats(0.0, 1e-6))
    elif family == "huge_magnitude":
        adv = draw(_finite_floats(1e12, 1e15))
        participation = draw(_finite_floats(1e-4, 2.0))
        daily_vol = draw(_finite_floats(0.0, 1e3))
    elif family == "zero_vol":
        adv = draw(_finite_floats(1.0, 1e8))
        participation = draw(_finite_floats(1e-4, 2.0))
        daily_vol = 0.0
    elif family == "tiny_participation":
        adv = draw(_finite_floats(1.0, 1e8))
        participation = draw(_finite_floats(1e-6, 1e-6))
        daily_vol = draw(_finite_floats(0.0, 0.5))
    else:  # huge_participation
        adv = draw(_finite_floats(1.0, 1e8))
        participation = draw(_finite_floats(1.0, 10.0))
        daily_vol = draw(_finite_floats(0.0, 0.5))
    sign = draw(st.sampled_from([1.0, -1.0]))
    trade_size = sign * participation * adv
    return MarketImpactCase(
        trade_size=trade_size, adv=adv, daily_vol=daily_vol, coefficient=coefficient, family=family
    )


@given(case=market_impact_case())
def test_square_root_market_impact_matches_closed_form(case: MarketImpactCase) -> None:
    """Quantcore's `square_root_market_impact` against the hand-derived square-root law."""
    check(
        _SQUARE_ROOT_MARKET_IMPACT,
        case.trade_size,
        case.adv,
        case.daily_vol,
        case.coefficient,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@dataclass(frozen=True)
class CarryCase:
    """One `short_borrow_cost`/`margin_financing_cost` input, tagged with its family."""

    notional: float
    rate_annual: float
    days_held: float
    family: str


@st.composite
def carry_case(draw: st.DrawFn) -> CarryCase:
    """Draw a carry-cost input from a magnitude/horizon family, ACT/365 accrual."""
    family = draw(
        st.sampled_from(
            ["typical", "tiny_magnitude", "huge_magnitude", "zero_days", "long_horizon"]
        )
    )
    if family == "typical":
        notional = draw(_finite_floats(0.0, 1e9))
        rate_annual = draw(_finite_floats(0.0, 1.0))
        days_held = draw(_finite_floats(0.0, 400.0))
    elif family == "tiny_magnitude":
        notional = draw(_finite_floats(0.0, 1e-6))
        rate_annual = draw(_finite_floats(0.0, 1e-8))
        days_held = draw(_finite_floats(0.0, 5.0))
    elif family == "huge_magnitude":
        notional = draw(_finite_floats(1e12, 1e15))
        rate_annual = draw(_finite_floats(0.0, 5.0))
        days_held = draw(_finite_floats(0.0, 400.0))
    elif family == "zero_days":
        notional = draw(_finite_floats(0.0, 1e9))
        rate_annual = draw(_finite_floats(0.0, 1.0))
        days_held = 0.0
    else:  # long_horizon
        notional = draw(_finite_floats(0.0, 1e9))
        rate_annual = draw(_finite_floats(0.0, 1.0))
        days_held = draw(_finite_floats(365.0, float(_DAYS_PER_10_YEARS)))
    return CarryCase(notional=notional, rate_annual=rate_annual, days_held=days_held, family=family)


@given(case=carry_case())
def test_short_borrow_cost_matches_closed_form(case: CarryCase) -> None:
    """Quantcore's `short_borrow_cost` against the hand-derived ACT/365 accrual."""
    check(
        _SHORT_BORROW_COST,
        case.notional,
        case.rate_annual,
        case.days_held,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@given(case=carry_case())
def test_margin_financing_cost_matches_closed_form(case: CarryCase) -> None:
    """Quantcore's `margin_financing_cost` against the hand-derived ACT/365 accrual."""
    check(
        _MARGIN_FINANCING_COST,
        case.notional,
        case.rate_annual,
        case.days_held,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


@dataclass(frozen=True)
class RollYieldCase:
    """One `futures_roll_yield` input, tagged with the term-structure family that produced it."""

    near_price: float
    far_price: float
    days_to_roll: int
    family: str


@st.composite
def roll_yield_case(draw: st.DrawFn) -> RollYieldCase:
    """Draw a `futures_roll_yield` input from a term-structure/magnitude/horizon family."""
    family = draw(st.sampled_from(list(FUTURES_ROLL_YIELD_FAMILIES)))
    days_to_roll = draw(st.integers(min_value=1, max_value=365))
    if family == "backwardation":
        far_price = draw(_finite_floats(1.0, 1e6))
        near_price = far_price * draw(_finite_floats(1.0001, 3.0))
    elif family == "contango":
        far_price = draw(_finite_floats(1.0, 1e6))
        near_price = far_price * draw(_finite_floats(1e-3, 0.9999))
    elif family == "flat":
        near_price = draw(_finite_floats(1.0, 1e6))
        far_price = near_price
    elif family == "tiny_price":
        near_price = draw(_finite_floats(1e-6, 1e-3))
        far_price = draw(_finite_floats(1e-6, 1e-3))
    elif family == "huge_price":
        near_price = draw(_finite_floats(1e9, 1e12))
        far_price = draw(_finite_floats(1e9, 1e12))
    elif family == "short_days_to_roll":
        near_price = draw(_finite_floats(1.0, 1e6))
        far_price = draw(_finite_floats(1.0, 1e6))
        days_to_roll = 1
    else:  # long_days_to_roll
        near_price = draw(_finite_floats(1.0, 1e6))
        far_price = draw(_finite_floats(1.0, 1e6))
        days_to_roll = draw(st.integers(min_value=180, max_value=365))
    return RollYieldCase(
        near_price=near_price, far_price=far_price, days_to_roll=days_to_roll, family=family
    )


@given(case=roll_yield_case())
def test_futures_roll_yield_matches_closed_form(case: RollYieldCase) -> None:
    """Quantcore's `futures_roll_yield` against the hand-derived near/far term-structure formula."""
    check(
        _FUTURES_ROLL_YIELD,
        case.near_price,
        case.far_price,
        case.days_to_roll,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


# --- Contract tests: NaN/inf/negative through the validating functions of `targets.execution` ---
#
# What raw quantcore does on these same inputs is recorded via `conformance.edge_cases.observe`
# (feeds findings) but never asserted on -- a raw quantcore misbehaviour on an invalid input is
# not this suite's job to fail the build over.


def test_square_root_market_impact_rejects_nan_inf_and_bad_sign() -> None:
    """`square_root_market_impact` raises `ValueError` on NaN/inf and non-positive `adv`."""
    for bad_value, argument, kwargs in [
        (math.nan, "trade_size", {}),
        (math.inf, "trade_size", {}),
        (math.nan, "adv", {}),
        (0.0, "adv", {}),
        (-1.0, "adv", {}),
        (math.inf, "adv", {}),
        (math.nan, "daily_vol", {}),
        (-0.01, "daily_vol", {}),
        (math.inf, "daily_vol", {}),
        (math.nan, "coefficient", {}),
        (-1.0, "coefficient", {}),
        (math.inf, "coefficient", {}),
    ]:
        base = {"trade_size": 1000.0, "adv": 1_000_000.0, "daily_vol": 0.02, "coefficient": 1.0}
        base[argument] = bad_value
        base.update(kwargs)
        try:
            square_root_market_impact(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {argument}={bad_value!r}")
    observe(
        "square_root_market_impact",
        "quantcore",
        "nan trade_size",
        qc_execution.square_root_market_impact,
        math.nan,
        1_000_000.0,
        0.02,
        1.0,
    )
    observe(
        "square_root_market_impact",
        "quantcore",
        "zero adv",
        qc_execution.square_root_market_impact,
        1000.0,
        0.0,
        0.02,
        1.0,
    )


def test_short_borrow_cost_rejects_nan_inf_and_negative() -> None:
    """`short_borrow_cost` raises `ValueError` on NaN/inf/negative `notional`/`rate`/`days_held`."""
    for argument, bad_value in [
        ("notional", math.nan),
        ("notional", math.inf),
        ("notional", -1.0),
        ("borrow_rate_annual", math.nan),
        ("borrow_rate_annual", math.inf),
        ("borrow_rate_annual", -0.01),
        ("days_held", math.nan),
        ("days_held", math.inf),
        ("days_held", -1.0),
    ]:
        base = {"notional": 1_000_000.0, "borrow_rate_annual": 0.02, "days_held": 30.0}
        base[argument] = bad_value
        try:
            short_borrow_cost(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {argument}={bad_value!r}")
    observe(
        "short_borrow_cost",
        "quantcore",
        "negative notional",
        qc_execution.short_borrow_cost,
        -1.0,
        0.02,
        30,
    )


def test_margin_financing_cost_rejects_nan_inf_and_negative() -> None:
    """`margin_financing_cost` raises `ValueError` on NaN/inf/negative inputs."""
    for argument, bad_value in [
        ("notional", math.nan),
        ("notional", math.inf),
        ("notional", -1.0),
        ("financing_rate_annual", math.nan),
        ("financing_rate_annual", math.inf),
        ("financing_rate_annual", -0.01),
        ("days_held", math.nan),
        ("days_held", math.inf),
        ("days_held", -1.0),
    ]:
        base = {"notional": 1_000_000.0, "financing_rate_annual": 0.05, "days_held": 30.0}
        base[argument] = bad_value
        try:
            margin_financing_cost(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {argument}={bad_value!r}")
    observe(
        "margin_financing_cost",
        "quantcore",
        "negative notional",
        qc_execution.margin_financing_cost,
        -1.0,
        0.05,
        30,
    )


def test_futures_roll_yield_rejects_nan_inf_and_non_positive() -> None:
    """`futures_roll_yield` raises `ValueError` on NaN/inf/non-positive prices or `days_to_roll`."""
    for argument, bad_value in [
        ("near_price", math.nan),
        ("near_price", math.inf),
        ("near_price", 0.0),
        ("near_price", -1.0),
        ("far_price", math.nan),
        ("far_price", math.inf),
        ("far_price", 0.0),
        ("far_price", -1.0),
    ]:
        base = {"near_price": 105.0, "far_price": 100.0, "days_to_roll": 30}
        base[argument] = bad_value
        try:
            futures_roll_yield(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {argument}={bad_value!r}")
    for bad_days in (0, -1):
        try:
            futures_roll_yield(near_price=105.0, far_price=100.0, days_to_roll=bad_days)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for days_to_roll={bad_days!r}")
    observe(
        "futures_roll_yield",
        "quantcore",
        "zero far_price",
        qc_execution.futures_roll_yield,
        105.0,
        0.0,
        30,
    )
