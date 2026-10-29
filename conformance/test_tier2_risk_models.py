"""Differential tests for the GARCH-family variance recursions against `arch`.

`garch_11_variance` is Tier-1; the rest of this family is Tier-2. The fitters
(`fit_garch_11`, `egarch_fit`, `gjr_garch_fit`) are computed by `arch` itself, so their checks are
the convention tests in `tests/unit/risk/test_arch_fits.py`.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import conformance.targets.risk_models  # noqa: F401
from conformance.compare import registry
from conformance.generators import RETURN_FAMILIES, TaggedReturns, length_bucket_for, returns_series
from conformance.harness import check

_EWMA_VARIANCE = registry()["ewma_variance"]
_GARCH_11_VARIANCE = registry()["garch_11_variance"]
_EGARCH_11_VARIANCE = registry()["egarch_11_variance"]
_GJR_GARCH_11_VARIANCE = registry()["gjr_garch_11_variance"]


# Declared coverage: `build_report.py` fails the build if any declared family or length bucket
# ends up with zero recorded cases. Each `*_MIN_LEN` is the shortest series that test draws; it
# is at least the function's own minimum input length and is larger where a fair comparison needs
# more points.
EWMA_VARIANCE_MIN_LEN = 1
EWMA_VARIANCE_FAMILIES: tuple[str, ...] = RETURN_FAMILIES
EWMA_VARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

GARCH_11_VARIANCE_MIN_LEN = 1
GARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale")
)
GARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

EGARCH_11_VARIANCE_MIN_LEN = 1
# "near_constant" is additionally excluded (unlike garch_11_variance/gjr_garch_11_variance): the
# EGARCH log-variance recursion divides by sigma_{t-1} to form z_{t-1} = eps_{t-1}/sigma_{t-1}
# (egarch.py's module docstring), so on a near-constant series (sigma tiny and itself only
# known to floating precision) rounding differences are amplified through that division and the
# exp/log round trip; the two paths then differ by up to 0.6 relative.
EGARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale", "near_constant")
)
EGARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

GJR_GARCH_11_VARIANCE_MIN_LEN = 1
GJR_GARCH_11_VARIANCE_FAMILIES: tuple[str, ...] = tuple(
    f for f in RETURN_FAMILIES if f not in ("huge_scale", "tiny_scale")
)
GJR_GARCH_11_VARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# Fitting needs enough observations to identify four free parameters reliably; drawn lengths are
# clipped up front (see `_fit_garch_len_strategy`) so "short"/"tiny" generator draws never reach
# the fitter at all -- only "medium" and "long" are declared.


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
