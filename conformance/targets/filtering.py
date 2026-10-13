"""Linear Kalman filtering and smoothing, and a time-varying (Kalman) hedge-ratio regression.

Each quantity is registered as a `ConformanceTarget` pairing a quantcore adapter (a thin call into
`quantcore.filtering.kalman`) with an independent oracle. Importing this module registers all of
them.

State `x_t = A x_{t-1} + w_t` (`w ~ N(0, Q)`), observation `y_t = H x_t + v_t` (`v ~ N(0, R)`);
`KalmanModel` bundles `(A, H, Q, R)`. quantcore predicts once before the first update (`x0` is
`x_{-1|-1}`); `pykalman` takes its initial state as `x_{0|-1}`, so it is fed `A @ x0` and
`A @ P0 @ A.T + Q`.

`dynamic_hedge_ratio` is a one-state filter for the slope of `y` on `x` (no intercept), started
from the prior `beta ~ N(initial_beta, 1)` with `initial_beta = 0` unless the caller passes
another mean. Its oracle is the same model written out by hand from the same prior, and every
element is compared from the first.

Full citations for Kalman (1960) and Rauch, Tung & Striebel (1965) are in `docs/REFERENCES.md`
under "Kalman filtering".
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pykalman import KalmanFilter

import quantcore.filtering.kalman as qc_kalman
from conformance.compare import ArrayComparator, ConformanceTarget

__all__ = ["KalmanModel"]


@dataclass(frozen=True)
class KalmanModel:
    """The four fixed matrices of a linear Kalman model, bundled into one argument.

    Attributes:
        a: State transition matrix, shape `(n, n)`.
        h: Observation matrix, shape `(m, n)`.
        q: Process noise covariance, shape `(n, n)`.
        r: Observation noise covariance, shape `(m, m)`.
    """

    a: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]
    q: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]


def _pykalman_filter(
    model: KalmanModel, x0: npt.NDArray[np.float64], p0: npt.NDArray[np.float64]
) -> KalmanFilter:
    """A `pykalman.KalmanFilter` initialised so its "step 0" matches quantcore's."""
    x0_shifted = model.a @ x0
    p0_shifted = model.a @ p0 @ model.a.T + model.q
    return KalmanFilter(
        transition_matrices=model.a,
        observation_matrices=model.h,
        transition_covariance=model.q,
        observation_covariance=model.r,
        initial_state_mean=x0_shifted,
        initial_state_covariance=p0_shifted,
    )


# Registered as two ConformanceTargets apiece (means, covs), not one returning a 2-tuple:
# `TupleComparator` (conformance.compare) is typed for scalar (`Comparator[float]`)
# sub-results only, matching how `ols_coefficients`/`ols_r_squared` (statistics_extra.py) split a
# multi-quantity result rather than force a mixed comparator through it.


def _kalman_filter_means_quantcore(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    means, _covs = qc_kalman.kalman_filter(observations, model.a, model.h, model.q, model.r, x0, p0)
    return np.asarray(means, dtype=np.float64)


def _kalman_filter_means_reference(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    means, _covs = _pykalman_filter(model, x0, p0).filter(observations)
    return np.asarray(means, dtype=np.float64)


_KALMAN_FILTER_MEANS = ConformanceTarget(
    name="kalman_filter_means",
    tier=2,
    oracle="pykalman.KalmanFilter (initial state pre-advanced one predict step to match quantcore)",
    citation="Kalman, R.E. (1960). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_kalman_filter_means_quantcore,
    reference=_kalman_filter_means_reference,
)


def _kalman_filter_covs_quantcore(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    _means, covs = qc_kalman.kalman_filter(observations, model.a, model.h, model.q, model.r, x0, p0)
    return np.asarray(covs, dtype=np.float64)


def _kalman_filter_covs_reference(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    _means, covs = _pykalman_filter(model, x0, p0).filter(observations)
    return np.asarray(covs, dtype=np.float64)


_KALMAN_FILTER_COVS = ConformanceTarget(
    name="kalman_filter_covs",
    tier=2,
    oracle="pykalman.KalmanFilter (initial state pre-advanced one predict step to match quantcore)",
    citation="Kalman, R.E. (1960). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_kalman_filter_covs_quantcore,
    reference=_kalman_filter_covs_reference,
)


def _kalman_smooth_means_quantcore(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    means, _covs = qc_kalman.kalman_smooth(observations, model.a, model.h, model.q, model.r, x0, p0)
    return np.asarray(means, dtype=np.float64)


def _kalman_smooth_means_reference(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    means, _covs = _pykalman_filter(model, x0, p0).smooth(observations)
    return np.asarray(means, dtype=np.float64)


_KALMAN_SMOOTH_MEANS = ConformanceTarget(
    name="kalman_smooth_means",
    tier=2,
    oracle="pykalman.KalmanFilter.smooth (RTS), same initial-state adjustment as the filter",
    citation="Rauch, Tung & Striebel (1965). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_kalman_smooth_means_quantcore,
    reference=_kalman_smooth_means_reference,
)


def _kalman_smooth_covs_quantcore(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    _means, covs = qc_kalman.kalman_smooth(observations, model.a, model.h, model.q, model.r, x0, p0)
    return np.asarray(covs, dtype=np.float64)


def _kalman_smooth_covs_reference(
    observations: npt.NDArray[np.float64],
    model: KalmanModel,
    x0: npt.NDArray[np.float64],
    p0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    _means, covs = _pykalman_filter(model, x0, p0).smooth(observations)
    return np.asarray(covs, dtype=np.float64)


_KALMAN_SMOOTH_COVS = ConformanceTarget(
    name="kalman_smooth_covs",
    tier=2,
    oracle="pykalman.KalmanFilter.smooth (RTS), same initial-state adjustment as the filter",
    citation="Rauch, Tung & Striebel (1965). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_kalman_smooth_covs_quantcore,
    reference=_kalman_smooth_covs_reference,
)


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
