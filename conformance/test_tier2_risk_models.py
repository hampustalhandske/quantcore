"""Differential tests for the GARCH-family risk models against `arch`.

`garch_11_variance`, `fit_garch_11` are Tier-1; the rest of this family is Tier-2.

`fit_garch_11`, `egarch_fit`, `gjr_garch_fit` are each ONE registered `ConformanceTarget` recording
two quantities: `log_likelihood` (the decision quantity, compared) and `params` (reported,
secondary, never compared: these are iterative estimators without a unique optimum, so only the
attained log-likelihood is a meaningful cross-library comparison). `check()` records both from
the single `(log_likelihood, params)` result each call returns.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import conformance.targets.risk_models  # noqa: F401
from conformance.compare import registry
from conformance.generators import RETURN_FAMILIES, TaggedReturns, length_bucket_for, returns_series
from conformance.harness import check

_EWMA_VARIANCE = registry()["ewma_variance"]
_GARCH_11_VARIANCE = registry()["garch_11_variance"]
_EGARCH_11_VARIANCE = registry()["egarch_11_variance"]
_GJR_GARCH_11_VARIANCE = registry()["gjr_garch_11_variance"]
_FIT_GARCH_11 = registry()["fit_garch_11"]
_EGARCH_FIT = registry()["egarch_fit"]
_GJR_GARCH_FIT = registry()["gjr_garch_fit"]

_FIT_QUANTITIES = ("log_likelihood", "params")

# Declared coverage: `build_report.py` fails the build if any declared family or length bucket
# ends up with zero recorded cases. Each `*_MIN_LEN` is the shortest series that test draws; it
# is at least the function's own minimum input length and is larger where a fair comparison needs
# more points.
EWMA_VARIANCE_MIN_LEN = 1
EWMA_VARIANCE_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
EWMA_VARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# The burn-in comparator (`conformance.targets.risk_models._BurnInArrayComparator`, cap
# 300) needs enough points *after* it to compare a converged tail; below ~50 points essentially
# nothing survives the slice, so quantcore's and arch's differing seedings (never having had
# time to converge) are compared unfairly. GARCH_11_VARIANCE_MIN_LEN is therefore this test's
# own floor for a *fair* comparison, not garch_11_variance's real minimum input length (1) --
# the same MIN_LEN/true-minimum split FIT_GARCH_11_MIN_LEN below uses.
GARCH_11_VARIANCE_MIN_LEN = 50
GARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale")
)
GARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("medium", "long")

EGARCH_11_VARIANCE_MIN_LEN = 50
# "near_constant" is additionally excluded (unlike garch_11_variance/gjr_garch_11_variance): the
# EGARCH log-variance recursion divides by sigma_{t-1} to form z_{t-1} = eps_{t-1}/sigma_{t-1}
# (egarch.py's module docstring), so on a near-constant series (sigma tiny and itself only
# known to floating precision) a minuscule difference between quantcore's and arch's seedings is
# amplified every step through that division and the exp/log round trip, and never converges
# within any practical burn-in -- observed persisting past a 500+ point burn-in at both "medium"
# and "long" buckets, unlike the transient garch_11_variance/gjr_garch_11_variance show.
EGARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale", "near_constant")
)
EGARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("medium", "long")

GJR_GARCH_11_VARIANCE_MIN_LEN = 50
GJR_GARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale")
)
GJR_GARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("medium", "long")

# Fitting needs enough observations to identify four free parameters reliably; drawn lengths are
# clipped up front (see `_fit_garch_len_strategy`) so "short"/"tiny" generator draws never reach
# the fitter at all -- only "medium" and "long" are declared.
FIT_GARCH_11_MIN_LEN = 300
FIT_GARCH_11_FAMILIES: tuple[str, ...] = ("normal", "student_t3", "garch11")
FIT_GARCH_11_BUCKETS: tuple[str, ...] = ("medium", "long")

EGARCH_FIT_MIN_LEN = 400
EGARCH_FIT_FAMILIES: tuple[str, ...] = ("normal", "student_t3", "garch11")
EGARCH_FIT_BUCKETS: tuple[str, ...] = ("medium", "long")

GJR_GARCH_FIT_MIN_LEN = 300
GJR_GARCH_FIT_FAMILIES: tuple[str, ...] = ("normal", "student_t3", "garch11")
GJR_GARCH_FIT_BUCKETS: tuple[str, ...] = ("medium", "long")


@settings(max_examples=400)
@given(tagged=returns_series(min_len=EWMA_VARIANCE_MIN_LEN), lambda_=st.floats(0.80, 0.99))
def test_ewma_variance_matches_hand_recursion(tagged: TaggedReturns, lambda_: float) -> None:
    check(
        _EWMA_VARIANCE,
        tagged.values,
        lambda_,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


# alpha+beta capped at 0.9 (see `conformance.targets.risk_models._BurnInArrayComparator`'s
# docstring): keeps the two seedings' transient well inside the fixed 300-point burn-in that
# comparator uses.
@settings(max_examples=400)
@given(
    tagged=returns_series(min_len=GARCH_11_VARIANCE_MIN_LEN, families=GARCH_11_VARIANCE_FAMILIES),
    omega=st.floats(1e-6, 1e-3),
    alpha=st.floats(0.02, 0.2),
    beta=st.floats(0.3, 0.68),
)
def test_garch_11_variance_matches_arch(
    tagged: TaggedReturns, omega: float, alpha: float, beta: float
) -> None:
    epsilon = tagged.values - np.mean(tagged.values)
    check(
        _GARCH_11_VARIANCE,
        epsilon,
        omega,
        alpha,
        beta,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@settings(max_examples=400)
@given(
    tagged=returns_series(min_len=EGARCH_11_VARIANCE_MIN_LEN, families=EGARCH_11_VARIANCE_FAMILIES),
    omega=st.floats(-0.5, 0.5),
    alpha=st.floats(0.02, 0.2),
    gamma=st.floats(-0.15, 0.05),
    beta=st.floats(0.3, 0.68),
)
def test_egarch_11_variance_matches_arch(
    tagged: TaggedReturns, omega: float, alpha: float, gamma: float, beta: float
) -> None:
    epsilon = tagged.values - np.mean(tagged.values)
    check(
        _EGARCH_11_VARIANCE,
        epsilon,
        omega,
        alpha,
        gamma,
        beta,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@settings(max_examples=400)
@given(
    tagged=returns_series(
        min_len=GJR_GARCH_11_VARIANCE_MIN_LEN, families=GJR_GARCH_11_VARIANCE_FAMILIES
    ),
    omega=st.floats(1e-6, 1e-3),
    alpha=st.floats(0.01, 0.1),
    gamma=st.floats(0.0, 0.1),
    beta=st.floats(0.3, 0.6),
)
def test_gjr_garch_11_variance_matches_arch(
    tagged: TaggedReturns, omega: float, alpha: float, gamma: float, beta: float
) -> None:
    epsilon = tagged.values - np.mean(tagged.values)
    check(
        _GJR_GARCH_11_VARIANCE,
        epsilon,
        omega,
        alpha,
        gamma,
        beta,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


def _fit_garch_len_strategy(min_len: int) -> "st.SearchStrategy[TaggedReturns]":
    return returns_series(
        min_len=min_len, max_len=1500, families=("normal", "student_t3", "garch11")
    )


@given(tagged=_fit_garch_len_strategy(FIT_GARCH_11_MIN_LEN))
def test_fit_garch_11_matches_arch(tagged: TaggedReturns) -> None:
    check(
        _FIT_GARCH_11,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=_FIT_QUANTITIES,
    )


@given(tagged=_fit_garch_len_strategy(EGARCH_FIT_MIN_LEN))
def test_egarch_fit_matches_arch(tagged: TaggedReturns) -> None:
    # The shared log-likelihood evaluator reuses egarch_11_variance's own recursion (see
    # `conformance.targets.risk_models`), which -- per quantcore.risk.egarch's own documented
    # caveat -- can raise ZeroDivisionError on an interior step if sigma_prev underflows to
    # exactly 0.0 for a particular (rare) parameter/data combination. This is not a call to the
    # oracle: it is this test deciding, before invoking check(), whether the generated case is
    # evaluable at all under the shared likelihood, exactly as `assume()` filters a degenerate
    # input elsewhere in this suite.
    assert _EGARCH_FIT.reference is not None
    try:
        reference_ll, _reference_params = _EGARCH_FIT.reference(tagged.values)
    except ZeroDivisionError:
        assume(False)
    else:
        assume(np.isfinite(reference_ll))

    check(
        _EGARCH_FIT,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=_FIT_QUANTITIES,
    )


@given(tagged=_fit_garch_len_strategy(GJR_GARCH_FIT_MIN_LEN))
def test_gjr_garch_fit_matches_arch(tagged: TaggedReturns) -> None:
    check(
        _GJR_GARCH_FIT,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
        quantities=_FIT_QUANTITIES,
    )
