"""Differential tests for the Kalman hedge-ratio regression."""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from conformance.compare import registry
from conformance.harness import check

_DYNAMIC_HEDGE_RATIO = registry()["dynamic_hedge_ratio"]


DYNAMIC_HEDGE_RATIO_FAMILIES: tuple[str, ...] = ("stable_hedge", "drifting_hedge")
DYNAMIC_HEDGE_RATIO_BUCKETS: tuple[str, ...] = ("short", "medium", "long")


def _length_bucket(n: int) -> str:
    if n <= 10:
        return "short"
    if n <= 500:
        return "medium"
    return "long"


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
