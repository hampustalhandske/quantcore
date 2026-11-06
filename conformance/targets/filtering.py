"""A time-varying (Kalman) hedge-ratio regression.

The quantity is registered as a `ConformanceTarget` pairing a quantcore adapter (a thin call into
`quantcore.filtering.kalman`) with an independent oracle. Importing this module registers it.

`dynamic_hedge_ratio` is a one-state filter for the slope of `y` on `x` (no intercept), started
from the prior `beta ~ N(initial_beta, 1)` with `initial_beta = 0` unless the caller passes
another mean. Its oracle is the same model written out by hand from the same prior, and every
element is compared from the first.

Full citations for Kalman (1960) are in `docs/REFERENCES.md` under "Kalman filtering".
"""

import numpy as np
import numpy.typing as npt

import quantcore.filtering.kalman as qc_kalman
from conformance.compare import ArrayComparator, ConformanceTarget

__all__: list[str] = []


# --- dynamic_hedge_ratio: hand scalar Kalman regression started from the same prior -------------


def _dynamic_hedge_ratio_quantcore(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    obs_var: float,
    proc_var: float,
    initial_beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_kalman.dynamic_hedge_ratio(y, x, obs_var, proc_var, initial_beta=initial_beta),
        dtype=np.float64,
    )


def _dynamic_hedge_ratio_reference(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    obs_var: float,
    proc_var: float,
    initial_beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Hand scalar Kalman regression of `y` on `x`, started from `beta ~ N(initial_beta, 1)`.

    One state (the slope, no intercept) following a random walk with variance `proc_var`,
    observed through `y_t = beta_t * x_t + v_t` with `Var[v_t] = obs_var`. A predict/update step
    runs for every observation including the first, so the first predicted variance is
    `1 + proc_var`.
    """
    n = y.shape[0]
    beta = np.empty(n, dtype=np.float64)
    beta_prev, p_prev = initial_beta, 1.0
    for t in range(n):
        beta_pred = beta_prev
        p_pred = p_prev + proc_var
        h_t = x[t]
        innovation_cov = h_t * p_pred * h_t + obs_var
        gain = p_pred * h_t / innovation_cov
        innovation = y[t] - h_t * beta_pred
        beta_upd = beta_pred + gain * innovation
        p_upd = (1.0 - gain * h_t) * p_pred
        beta[t] = beta_upd
        beta_prev, p_prev = beta_upd, p_upd
    return beta


_DYNAMIC_HEDGE_RATIO = ConformanceTarget(
    name="dynamic_hedge_ratio",
    tier=2,
    oracle=(
        "hand scalar Kalman regression started from the same prior, beta ~ N(initial_beta, 1), "
        "compared on every element"
    ),
    citation=(
        "Kalman, R.E. (1960) for the recursion; Chan, E.P. (2013), Algorithmic Trading: Winning "
        "Strategies and Their Rationale, ch. 3, for the hedge ratio as a Kalman-filter state."
    ),
    compare=ArrayComparator(),
    quantcore=_dynamic_hedge_ratio_quantcore,
    reference=_dynamic_hedge_ratio_reference,
)
