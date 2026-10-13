"""Differential tests for Kalman filtering/smoothing (`pykalman`) and the hedge-ratio regression."""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from conformance.compare import registry
from conformance.harness import check
from conformance.targets.filtering import KalmanModel

_KALMAN_FILTER_MEANS = registry()["kalman_filter_means"]
_KALMAN_FILTER_COVS = registry()["kalman_filter_covs"]
_KALMAN_SMOOTH_MEANS = registry()["kalman_smooth_means"]
_KALMAN_SMOOTH_COVS = registry()["kalman_smooth_covs"]
_DYNAMIC_HEDGE_RATIO = registry()["dynamic_hedge_ratio"]

type _KalmanCase = tuple[
    npt.NDArray[np.float64],
    KalmanModel,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    str,
    int,
]

_KALMAN_FAMILIES: tuple[str, ...] = ("scalar_random_walk", "two_state_local_level")
_KALMAN_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

KALMAN_FILTER_MEANS_FAMILIES = _KALMAN_FAMILIES
KALMAN_FILTER_MEANS_BUCKETS = _KALMAN_BUCKETS
KALMAN_FILTER_COVS_FAMILIES = _KALMAN_FAMILIES
KALMAN_FILTER_COVS_BUCKETS = _KALMAN_BUCKETS
KALMAN_SMOOTH_MEANS_FAMILIES = _KALMAN_FAMILIES
KALMAN_SMOOTH_MEANS_BUCKETS = _KALMAN_BUCKETS
KALMAN_SMOOTH_COVS_FAMILIES = _KALMAN_FAMILIES
KALMAN_SMOOTH_COVS_BUCKETS = _KALMAN_BUCKETS

DYNAMIC_HEDGE_RATIO_FAMILIES: tuple[str, ...] = ("stable_hedge", "drifting_hedge")
DYNAMIC_HEDGE_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


def _length_bucket(n: int) -> str:
    if n <= 10:
        return "short"
    if n <= 500:
        return "medium"
    return "long"


@st.composite
def kalman_problem(draw: st.DrawFn, min_len: int = 2, max_len: int = 800) -> _KalmanCase:
    """Draw a 1-D or 2-D linear-Gaussian state-space problem.

    Returns `(observations, model, x0, p0, family, seed)`.
    """
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    family = draw(st.sampled_from(["scalar_random_walk", "two_state_local_level"]))

    if family == "scalar_random_walk":
        a = np.array([[1.0]])
        h = np.array([[1.0]])
        q = np.array([[0.01]])
        r = np.array([[0.1]])
        x0 = np.array([0.0])
        p0 = np.array([[1.0]])
        state = 0.0
        observations = np.empty((n, 1), dtype=np.float64)
        for t in range(n):
            state += rng.normal(0.0, np.sqrt(q[0, 0]))
            observations[t, 0] = state + rng.normal(0.0, np.sqrt(r[0, 0]))
    else:  # two_state_local_level: level + drift
        a = np.array([[1.0, 1.0], [0.0, 1.0]])
        h = np.array([[1.0, 0.0]])
        q = np.diag([0.001, 0.0001])
        r = np.array([[0.05]])
        x0 = np.array([0.0, 0.01])
        p0 = np.eye(2)
        state = x0.copy()
        observations = np.empty((n, 1), dtype=np.float64)
        for t in range(n):
            state = a @ state + rng.normal(0.0, np.sqrt(np.diag(q)))
            observations[t, 0] = float((h @ state).item()) + rng.normal(0.0, np.sqrt(r[0, 0]))

    model = KalmanModel(a=a, h=h, q=q, r=r)
    return observations, model, x0, p0, family, seed


@settings(max_examples=40)
@given(draw_result=kalman_problem())
def test_kalman_filter_matches_pykalman(draw_result: _KalmanCase) -> None:
    observations, model, x0, p0, family, seed = draw_result
    bucket = _length_bucket(observations.shape[0])
    check(
        _KALMAN_FILTER_MEANS,
        observations,
        model,
        x0,
        p0,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )
    check(
        _KALMAN_FILTER_COVS,
        observations,
        model,
        x0,
        p0,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )


@settings(max_examples=40)
@given(draw_result=kalman_problem())
def test_kalman_smooth_matches_pykalman(draw_result: _KalmanCase) -> None:
    observations, model, x0, p0, family, seed = draw_result
    bucket = _length_bucket(observations.shape[0])
    check(
        _KALMAN_SMOOTH_MEANS,
        observations,
        model,
        x0,
        p0,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )
    check(
        _KALMAN_SMOOTH_COVS,
        observations,
        model,
        x0,
        p0,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )


@st.composite
def hedge_ratio_problem(
    draw: st.DrawFn, min_len: int = 2, max_len: int = 800
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str, int]:
    """Draw `(y, x, family, seed)` for a hedge relationship with a stable or drifting beta."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    family = draw(st.sampled_from(["stable_hedge", "drifting_hedge"]))

    x = rng.normal(0.0, 1.0, n) + 5.0
    if family == "stable_hedge":
        beta = 1.5
        y = beta * x + rng.normal(0.0, 0.1, n)
    else:
        beta_path = 1.0 + 0.5 * np.sin(np.linspace(0.0, 3.0, n))
        y = beta_path * x + rng.normal(0.0, 0.1, n)
    return y.astype(np.float64), x.astype(np.float64), family, seed


@settings(max_examples=40)
@given(draw_result=hedge_ratio_problem())
def test_dynamic_hedge_ratio_matches_hand_kalman_regression(
    draw_result: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str, int],
) -> None:
    y, x, family, seed = draw_result
    check(
        _DYNAMIC_HEDGE_RATIO,
        y,
        x,
        1e-3,
        1e-5,
        family=family,
        length_bucket=_length_bucket(y.shape[0]),
        seed=seed,
    )


@settings(max_examples=40)
@given(
    draw_result=hedge_ratio_problem(),
    initial_beta=st.one_of(
        st.floats(min_value=-5.0, max_value=-0.5), st.floats(min_value=0.5, max_value=5.0)
    ),
)
def test_dynamic_hedge_ratio_with_prior_mean_matches_hand_kalman_regression(
    draw_result: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str, int],
    initial_beta: float,
) -> None:
    """The filter started from a drawn prior mean `initial_beta` instead of the default `0`.

    The mean is at least 0.5 from the default, which moves the first value by about
    `0.5 * obs_var / x[0]**2`: 2e-5 at `x[0] = 5`, several times the tolerance. A prior mean that
    was ignored would therefore put the cases out of tolerance.
    """
    y, x, family, seed = draw_result
    check(
        _DYNAMIC_HEDGE_RATIO,
        y,
        x,
        1e-3,
        1e-5,
        initial_beta,
        family=family,
        length_bucket=_length_bucket(y.shape[0]),
        seed=seed,
    )
