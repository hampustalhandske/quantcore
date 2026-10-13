"""Tier-1 differential tests for Gaussian and Cornish-Fisher VaR/CVaR.

Against the closed form (`scipy.stats.norm`).
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

import conformance.targets.risk  # noqa: F401
from conformance.compare import registry
from conformance.generators import RETURN_FAMILIES, TaggedReturns, length_bucket_for, returns_series
from conformance.harness import check

_VALUE_AT_RISK = registry()["value_at_risk"]
_CONDITIONAL_VALUE_AT_RISK = registry()["conditional_value_at_risk"]
_VALUE_AT_RISK_CORNISH_FISHER = registry()["value_at_risk_cornish_fisher"]

VALUE_AT_RISK_MIN_LEN = 2
VALUE_AT_RISK_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
VALUE_AT_RISK_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

CONDITIONAL_VALUE_AT_RISK_MIN_LEN = 2
CONDITIONAL_VALUE_AT_RISK_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
CONDITIONAL_VALUE_AT_RISK_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

VALUE_AT_RISK_CORNISH_FISHER_MIN_LEN = 2
VALUE_AT_RISK_CORNISH_FISHER_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
VALUE_AT_RISK_CORNISH_FISHER_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


@given(
    tagged=returns_series(min_len=VALUE_AT_RISK_MIN_LEN),
    confidence_level=st.sampled_from([0.90, 0.95, 0.99]),
)
def test_value_at_risk_matches_closed_form(tagged: TaggedReturns, confidence_level: float) -> None:
    """Quantcore's Gaussian `value_at_risk` against the closed-form RiskMetrics formula."""
    check(
        _VALUE_AT_RISK,
        tagged.values,
        confidence_level,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=CONDITIONAL_VALUE_AT_RISK_MIN_LEN),
    confidence_level=st.sampled_from([0.90, 0.95, 0.99]),
)
def test_conditional_value_at_risk_matches_closed_form(
    tagged: TaggedReturns, confidence_level: float
) -> None:
    """Quantcore's Gaussian CVaR vs the closed-form truncated-normal tail mean."""
    check(
        _CONDITIONAL_VALUE_AT_RISK,
        tagged.values,
        confidence_level,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@given(
    tagged=returns_series(min_len=VALUE_AT_RISK_CORNISH_FISHER_MIN_LEN),
    confidence_level=st.sampled_from([0.90, 0.95, 0.99]),
)
def test_value_at_risk_cornish_fisher_matches_hand_formula(
    tagged: TaggedReturns, confidence_level: float
) -> None:
    """Quantcore's Cornish-Fisher `value_at_risk` against the hand-computed Zangari expansion."""
    assume(float(np.std(tagged.values, ddof=0)) != 0.0)
    check(
        _VALUE_AT_RISK_CORNISH_FISHER,
        tagged.values,
        confidence_level,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )
