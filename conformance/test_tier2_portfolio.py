"""Tier-2 differential tests for the PORTFOLIO family (`conformance.targets.portfolio`).

Covariance estimators (`sample_covariance`, `ewma_covariance`, `ledoit_wolf_shrinkage`) are checked
against hand/sklearn references on `returns_panel()` draws, bucketed by the panel's observation
count `n` (`length_bucket_for`, the harness's existing series-length bucketing). The mean-variance
family (`min_variance_weights`, `mean_variance_weights`, `unconstrained_mean_variance_weights`,
`l1_turnover_penalized_weights`) and `implied_equilibrium_returns`/`black_litterman` take a
covariance matrix, not a series, so there is no natural "length" in the `length_bucket_for` sense;
each instead declares a `_dim_bucket` (`"dim_small"`/`"dim_medium"`/`"dim_large"`) from the matrix's
own dimension `k`, the same approach `test_tier1_execution.py` takes for its scalar-only
functions. `risk_parity_weights` uses the same
dimension bucketing. `kelly_fraction` takes two scalars with no covariance matrix at all, so it
declares its own small family set and the `"scalar"` bucket, exactly as
`test_tier1_execution.py`'s functions do.
"""

from dataclasses import dataclass

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import conformance.targets.portfolio  # noqa: F401
from conformance.compare import registry
from conformance.generators import (
    COVARIANCE_FAMILIES,
    TaggedCovariance,
    TaggedPanel,
    covariance_matrix,
    length_bucket_for,
    returns_panel,
)
from conformance.harness import check

_SAMPLE_COVARIANCE = registry()["sample_covariance"]
_EWMA_COVARIANCE = registry()["ewma_covariance"]
_LEDOIT_WOLF_SHRINKAGE = registry()["ledoit_wolf_shrinkage"]
_MIN_VARIANCE_WEIGHTS = registry()["min_variance_weights"]
_MEAN_VARIANCE_WEIGHTS = registry()["mean_variance_weights"]
_UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS = registry()["unconstrained_mean_variance_weights"]
_L1_TURNOVER_PENALIZED_WEIGHTS = registry()["l1_turnover_penalized_weights"]
_RISK_PARITY_WEIGHTS = registry()["risk_parity_weights"]
_KELLY_FRACTION = registry()["kelly_fraction"]
_IMPLIED_EQUILIBRIUM_RETURNS = registry()["implied_equilibrium_returns"]
_BLACK_LITTERMAN = registry()["black_litterman"]

_OPTIMIZER_QUANTITIES = ("objective", "weights")
_SCALAR_BUCKET = "scalar"

# Dimension buckets for the covariance-matrix-only functions (no natural "length" to bucket, see
# the module docstring); `covariance_matrix()` draws k in [2, 10].
_DIM_SMALL_MAX = 4
_DIM_MEDIUM_MAX = 7


def _dim_bucket(k: int) -> str:
    if k <= _DIM_SMALL_MAX:
        return "dim_small"
    if k <= _DIM_MEDIUM_MAX:
        return "dim_medium"
    return "dim_large"


_DIM_BUCKETS: tuple[str, ...] = ("dim_small", "dim_medium", "dim_large")

# Declared coverage: `build_report.py` fails the build if any declared family or length/dimension
# bucket ends up with zero recorded cases.
SAMPLE_COVARIANCE_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
SAMPLE_COVARIANCE_BUCKETS: tuple[str, ...] = ("medium", "long")

EWMA_COVARIANCE_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
EWMA_COVARIANCE_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

LEDOIT_WOLF_SHRINKAGE_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
LEDOIT_WOLF_SHRINKAGE_BUCKETS: tuple[str, ...] = ("medium", "long")

MIN_VARIANCE_WEIGHTS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
MIN_VARIANCE_WEIGHTS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

MEAN_VARIANCE_WEIGHTS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
MEAN_VARIANCE_WEIGHTS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

L1_TURNOVER_PENALIZED_WEIGHTS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
L1_TURNOVER_PENALIZED_WEIGHTS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

RISK_PARITY_WEIGHTS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
RISK_PARITY_WEIGHTS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

IMPLIED_EQUILIBRIUM_RETURNS_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
IMPLIED_EQUILIBRIUM_RETURNS_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

BLACK_LITTERMAN_FAMILIES: tuple[str, ...] = COVARIANCE_FAMILIES
BLACK_LITTERMAN_BUCKETS: tuple[str, ...] = _DIM_BUCKETS

KELLY_FRACTION_FAMILIES: tuple[str, ...] = (
    "positive_edge",
    "negative_edge",
    "clip_binding_positive",
    "clip_binding_negative",
)
KELLY_FRACTION_BUCKETS: tuple[str, ...] = (_SCALAR_BUCKET,)


def _finite_floats(min_value: float, max_value: float) -> "st.SearchStrategy[float]":
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )


# --- sample_covariance, ewma_covariance, ledoit_wolf_shrinkage -----------------------------------


@settings(max_examples=200)
@given(tagged=returns_panel(min_len=15, max_len=1200))
def test_sample_covariance_matches_hand_reference(tagged: TaggedPanel) -> None:
    check(
        _SAMPLE_COVARIANCE,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@settings(max_examples=200)
@given(
    tagged=returns_panel(min_len=1, max_len=1200),
    lambda_=st.floats(0.80, 0.99),
)
def test_ewma_covariance_matches_hand_reference(tagged: TaggedPanel, lambda_: float) -> None:
    check(
        _EWMA_COVARIANCE,
        tagged.values,
        lambda_,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


@settings(max_examples=200)
@given(tagged=returns_panel(min_len=15, max_len=1200))
def test_ledoit_wolf_shrinkage_matches_sklearn(tagged: TaggedPanel) -> None:
    check(
        _LEDOIT_WOLF_SHRINKAGE,
        tagged.values,
        family=tagged.family,
        length_bucket=length_bucket_for(tagged.n),
        seed=tagged.seed,
    )


# --- mean-variance family: objective-based sign-test comparison ----------------------------------


@settings(max_examples=200)
@given(tagged=covariance_matrix(), allow_short=st.booleans())
def test_min_variance_weights_matches_cvxpy(tagged: TaggedCovariance, allow_short: bool) -> None:
    check(
        _MIN_VARIANCE_WEIGHTS,
        tagged.matrix,
        allow_short,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
        quantities=_OPTIMIZER_QUANTITIES,
    )


@settings(max_examples=200)
@given(
    tagged=covariance_matrix(),
    risk_aversion=st.floats(0.1, 10.0),
    allow_short=st.booleans(),
)
def test_mean_variance_weights_matches_cvxpy(
    tagged: TaggedCovariance, risk_aversion: float, allow_short: bool
) -> None:
    mu = np.random.default_rng(tagged.seed ^ 0x5EED).normal(0.0005, 0.002, tagged.k)
    check(
        _MEAN_VARIANCE_WEIGHTS,
        mu,
        tagged.matrix,
        risk_aversion,
        allow_short,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
        quantities=_OPTIMIZER_QUANTITIES,
    )


@settings(max_examples=200)
@given(tagged=covariance_matrix(), risk_aversion=st.floats(0.1, 10.0))
def test_unconstrained_mean_variance_weights_matches_closed_form(
    tagged: TaggedCovariance, risk_aversion: float
) -> None:
    mu = np.random.default_rng(tagged.seed ^ 0x5EED).normal(0.0005, 0.002, tagged.k)
    check(
        _UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS,
        mu,
        tagged.matrix,
        risk_aversion,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
        quantities=_OPTIMIZER_QUANTITIES,
    )


@settings(max_examples=200)
@given(
    tagged=covariance_matrix(),
    risk_aversion=st.floats(0.1, 10.0),
    cost_bps=st.floats(0.0, 100.0),
    allow_short=st.booleans(),
)
def test_l1_turnover_penalized_weights_matches_cvxpy(
    tagged: TaggedCovariance, risk_aversion: float, cost_bps: float, allow_short: bool
) -> None:
    rng = np.random.default_rng(tagged.seed ^ 0x5EED)
    mu = rng.normal(0.0005, 0.002, tagged.k)
    previous = rng.dirichlet(np.ones(tagged.k))
    check(
        _L1_TURNOVER_PENALIZED_WEIGHTS,
        mu,
        tagged.matrix,
        previous,
        cost_bps,
        risk_aversion,
        allow_short,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
        quantities=_OPTIMIZER_QUANTITIES,
    )


# --- risk_parity_weights: plain weight comparison against the Spinu (2013) reference -------------


@settings(max_examples=200)
@given(tagged=covariance_matrix())
def test_risk_parity_weights_matches_spinu_reference(tagged: TaggedCovariance) -> None:
    check(
        _RISK_PARITY_WEIGHTS,
        tagged.matrix,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
    )


# --- kelly_fraction ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KellyCase:
    """One `kelly_fraction` input, tagged with the family that produced it."""

    expected_return: float
    variance: float
    clip: bool
    family: str


@st.composite
def kelly_case(draw: "st.DrawFn") -> KellyCase:
    """Draw a `kelly_fraction` input covering ordinary, negative-edge, and clip-binding regimes."""
    family = draw(st.sampled_from(KELLY_FRACTION_FAMILIES))
    clip = draw(st.booleans())
    if family == "positive_edge":
        expected_return = draw(_finite_floats(1e-6, 0.05))
        variance = draw(_finite_floats(1e-4, 0.1))
    elif family == "negative_edge":
        expected_return = draw(_finite_floats(-0.05, -1e-6))
        variance = draw(_finite_floats(1e-4, 0.1))
    elif family == "clip_binding_positive":
        expected_return = draw(_finite_floats(0.5, 5.0))
        variance = draw(_finite_floats(1e-4, 0.05))
    else:  # clip_binding_negative
        expected_return = draw(_finite_floats(-5.0, -0.5))
        variance = draw(_finite_floats(1e-4, 0.05))
    return KellyCase(expected_return=expected_return, variance=variance, clip=clip, family=family)


@settings(max_examples=200)
@given(case=kelly_case())
def test_kelly_fraction_matches_hand_formula(case: KellyCase) -> None:
    check(
        _KELLY_FRACTION,
        case.expected_return,
        case.variance,
        case.clip,
        family=case.family,
        length_bucket=_SCALAR_BUCKET,
    )


# --- implied_equilibrium_returns, black_litterman -------------------------------------------------


@settings(max_examples=200)
@given(tagged=covariance_matrix(), risk_aversion=st.floats(0.5, 10.0))
def test_implied_equilibrium_returns_matches_hand_formula(
    tagged: TaggedCovariance, risk_aversion: float
) -> None:
    market_weights = np.random.default_rng(tagged.seed ^ 0x5EED).dirichlet(np.ones(tagged.k))
    check(
        _IMPLIED_EQUILIBRIUM_RETURNS,
        tagged.matrix,
        market_weights,
        risk_aversion,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
    )


@settings(max_examples=200)
@given(
    tagged=covariance_matrix(),
    risk_aversion=st.floats(0.5, 10.0),
    tau=st.floats(0.01, 0.2),
)
def test_black_litterman_matches_hand_formula(
    tagged: TaggedCovariance, risk_aversion: float, tau: float
) -> None:
    rng = np.random.default_rng(tagged.seed ^ 0x5EED)
    market_weights = rng.dirichlet(np.ones(tagged.k))
    # A single "asset 0 outperforms" view, present in every draw regardless of k, keeps the view
    # matrix well-posed (full row rank) across every generated dimension.
    p = np.zeros((1, tagged.k))
    p[0, 0] = 1.0
    q = np.array([float(rng.normal(0.0, 0.02))])
    omega = np.array([[float(rng.uniform(1e-5, 1e-2))]])
    check(
        _BLACK_LITTERMAN,
        tagged.matrix,
        market_weights,
        p,
        q,
        omega,
        risk_aversion,
        tau,
        family=tagged.family,
        length_bucket=_dim_bucket(tagged.k),
        seed=tagged.seed,
    )


# Since no library implements Black-Litterman's exact He & Litterman (1999) eq. 8 form,
# `black_litterman`'s
# reference is anchored by citation plus two independent invariants a wrong transcription would
# break -- checked here against the REFERENCE implementation directly (not quantcore), because
# these are cross-checks on the hand-derived formula's own correctness, independent of quantcore.
@settings(max_examples=100)
@given(tagged=covariance_matrix(), risk_aversion=st.floats(0.5, 10.0), tau=st.floats(0.01, 0.2))
def test_black_litterman_reference_zero_confidence_view_recovers_the_prior(
    tagged: TaggedCovariance, risk_aversion: float, tau: float
) -> None:
    """An infinitely uncertain view carries no information (He & Litterman 1999 eq. 8).

    As a view's uncertainty Omega -> infinity, the posterior (mu_BL, Sigma_BL) must converge to
    the prior (Pi, tau*Sigma). A transcription bug (e.g. a sign error, or Omega on the wrong side
    of the solve) would not, in general, preserve this.
    """
    reference = _BLACK_LITTERMAN.reference
    rng = np.random.default_rng(tagged.seed ^ 0xC0FFEE)
    market_weights = rng.dirichlet(np.ones(tagged.k))
    p = np.zeros((1, tagged.k))
    p[0, 0] = 1.0
    q = np.array([float(rng.normal(0.0, 0.02))])
    tau_sigma = tau * tagged.matrix
    view_variance = float((p @ tau_sigma @ p.T)[0, 0])
    omega = np.array([[max(view_variance, 1e-300) * 1e10]])

    prior_returns = risk_aversion * tagged.matrix @ market_weights
    prior_cov = tau_sigma
    posterior = reference(tagged.matrix, market_weights, p, q, omega, risk_aversion, tau)
    posterior_returns, posterior_cov = (
        posterior[: tagged.k],
        posterior[tagged.k :].reshape(tagged.k, tagged.k),
    )

    assert np.allclose(
        posterior_returns, prior_returns, rtol=1e-4, atol=1e-4 * max(1.0, view_variance)
    )
    assert np.allclose(posterior_cov, prior_cov, rtol=1e-4, atol=1e-4)


@settings(max_examples=100)
@given(tagged=covariance_matrix(), risk_aversion=st.floats(0.5, 10.0), tau=st.floats(0.01, 0.2))
def test_black_litterman_reference_infinite_confidence_view_recovers_the_view(
    tagged: TaggedCovariance, risk_aversion: float, tau: float
) -> None:
    """A certain view fully overrides the prior (He & Litterman 1999 eq. 8).

    As a single absolute view's uncertainty Omega -> 0, the posterior expected return for the
    viewed asset must converge to the view itself. A transcription bug would not, in general,
    have this property.
    """
    reference = _BLACK_LITTERMAN.reference
    rng = np.random.default_rng(tagged.seed ^ 0xC0FFEE)
    market_weights = rng.dirichlet(np.ones(tagged.k))
    p = np.zeros((1, tagged.k))
    p[0, 0] = 1.0
    q = np.array([float(rng.normal(0.0, 0.02))])
    tau_sigma = tau * tagged.matrix
    view_variance = float((p @ tau_sigma @ p.T)[0, 0])
    omega = np.array([[max(view_variance, 1e-300) * 1e-10]])

    posterior = reference(tagged.matrix, market_weights, p, q, omega, risk_aversion, tau)
    posterior_returns = posterior[: tagged.k]

    assert np.isclose(posterior_returns[0], q[0], rtol=1e-4, atol=1e-4)
