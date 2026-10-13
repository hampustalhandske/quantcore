"""Tier-1 differential tests: `sharpe_ratio`, `maximum_drawdown` against `empyrical-reloaded`."""

import numpy as np
import pandas as pd
import quantstats.stats
from hypothesis import assume, given
from hypothesis import strategies as st

import conformance.targets.performance  # noqa: F401
from conformance.compare import registry
from conformance.convert import annual_rate_to_per_period
from conformance.generators import RETURN_FAMILIES, TaggedReturns, length_bucket_for, returns_series
from conformance.harness import check, check_second_oracle

_SHARPE_RATIO = registry()["sharpe_ratio"]
_MAXIMUM_DRAWDOWN = registry()["maximum_drawdown"]

# Declared coverage: the shortest series each target's generator draws, and the families and
# length buckets each target must record at least one case in. `build_report.py` fails the build
# if any declared family or length bucket ends up with zero recorded cases.
SHARPE_RATIO_MIN_LEN = 2
"""Smallest `returns` length `sharpe_ratio` accepts: `numpy.std(..., ddof=1)` is undefined below 2
observations, and quantcore itself only guards against a wholly empty array."""
SHARPE_RATIO_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
SHARPE_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

MAXIMUM_DRAWDOWN_MIN_LEN = 1
"""Smallest `returns` length `maximum_drawdown` accepts: quantcore only guards against a wholly
empty array, and a single observation's drawdown is well-defined."""
MAXIMUM_DRAWDOWN_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
MAXIMUM_DRAWDOWN_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    tagged=returns_series(min_len=SHARPE_RATIO_MIN_LEN),
    risk_free_annual=st.floats(
        min_value=-0.05, max_value=0.10, allow_nan=False, allow_infinity=False
    ),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_sharpe_ratio_matches_empyrical(
    tagged: TaggedReturns, risk_free_annual: float, periods_per_year: int
) -> None:
    """Quantcore's `sharpe_ratio`, given a per-period risk-free rate, against `empyrical`."""
    assume(float(np.std(tagged.values, ddof=1)) != 0.0)
    risk_free_per_period = annual_rate_to_per_period(risk_free_annual, periods_per_year)
    check(
        _SHARPE_RATIO,
        tagged.values,
        risk_free_per_period,
        periods_per_year,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(tagged=returns_series(min_len=MAXIMUM_DRAWDOWN_MIN_LEN))
def test_maximum_drawdown_matches_empyrical(tagged: TaggedReturns) -> None:
    """Quantcore's `maximum_drawdown` (sign-flipped) against `empyrical.max_drawdown`."""
    check(
        _MAXIMUM_DRAWDOWN,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


# `sharpe_ratio` is checked against `quantstats.stats.sharpe` as a second oracle. This is the
# non-zero annual risk-free rate passed to it: `quantstats.stats.sharpe(rf=..., periods=...)`
# de-annualises the rate itself (geometrically, the same convention as
# `conformance.convert.annual_rate_to_per_period`), so it is passed as an *annual* rate, never
# pre-converted to a per-period one.
SHARPE_RATIO_QUANTSTATS_NONZERO_RF_ANNUAL = 0.03

# Hand-written explanation `build_report.py` attaches to any recorded `sharpe_ratio`/`quantstats`
# disagreement in CONFORMANCE.md's "Oracle disagreements" section. Every number that section shows
# alongside this text (inputs, both values) is generated from the run; only the prose is fixed.
SHARPE_RATIO_QUANTSTATS_DISAGREEMENT_EXPLANATION = """\
`quantstats.stats.sharpe` calls `quantstats.utils._prepare_returns(data, rf, periods)` on its
input before computing anything. That function guesses whether it was handed *prices* or
*returns* from the data alone:

    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change(fill_method=None)

A returns series that happens to be all non-negative with at least one value above 1 (e.g. a
short, high-volatility draw from this suite's `huge_scale` family) is silently reinterpreted as a
price level and replaced by its period-over-period percentage change -- a different series
entirely -- before the Sharpe ratio is computed. The primary oracle (`empyrical.sharpe_ratio`)
applies no such heuristic: it treats every input as returns, always. `quantcore.performance.
sharpe_ratio` follows `empyrical`'s convention, not `quantstats`'s, for exactly this reason -- the
standard formula (`mean(r - rf_p) / std(r - rf_p, ddof=1) * sqrt(p)`) is defined on returns, and
silently reinterpreting a valid returns series as prices is not part of that definition.

Where `quantstats` and `empyrical` agree, it is because `to_excess_returns` de-annualises `rf`
the same way `conformance.convert.annual_rate_to_per_period` does:

    if nperiods is not None:
        rf = _np.power(1 + rf, 1.0 / nperiods) - 1.0

so a disagreement recorded here is specifically the price-vs-returns misdetection above, not a
different risk-free convention.
"""


@given(
    tagged=returns_series(min_len=SHARPE_RATIO_MIN_LEN),
    periods_per_year=st.sampled_from([12, 52, 252]),
)
def test_sharpe_ratio_quantstats_cross_check(tagged: TaggedReturns, periods_per_year: int) -> None:
    """`quantstats.stats.sharpe` against the `empyrical` oracle of `sharpe_ratio`.

    Never asserts: a disagreement, a raise, or a NaN/inf from `quantstats` is recorded as its own
    case, not filtered out, and never influences `sharpe_ratio`'s status -- that decision is
    `test_sharpe_ratio_matches_empyrical`'s alone.
    """
    assume(float(np.std(tagged.values, ddof=1)) != 0.0)
    # quantstats.utils.to_excess_returns tz_localizes its input, which raises on a plain
    # RangeIndex whenever rf != 0; a DatetimeIndex is "the way quantstats expects" a Series.
    index = pd.date_range("2000-01-01", periods=tagged.n, freq="D")
    series = pd.Series(tagged.values, index=index)
    length_bucket = length_bucket_for(tagged.n)

    for quantity, risk_free_annual in (
        ("rf_zero", 0.0),
        ("rf_nonzero", SHARPE_RATIO_QUANTSTATS_NONZERO_RF_ANNUAL),
    ):
        risk_free_per_period = annual_rate_to_per_period(risk_free_annual, periods_per_year)
        check_second_oracle(
            _SHARPE_RATIO,
            series,
            family=tagged.family,
            length_bucket=length_bucket,
            oracle_name="quantstats",
            oracle_callable=quantstats.stats.sharpe,
            quantity=quantity,
            seed=tagged.seed,
            reference_kwargs={
                "risk_free_per_period": risk_free_per_period,
                "periods_per_year": periods_per_year,
            },
            rf=risk_free_annual,
            periods=periods_per_year,
            annualize=True,
        )
