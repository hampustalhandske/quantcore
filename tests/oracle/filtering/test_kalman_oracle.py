"""Oracle tests for `dynamic_hedge_ratio` vs `filterpy`.

Oracle: `filterpy.kalman.KalmanFilter` (predict/update forward pass).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("filterpy")

from filterpy.kalman import KalmanFilter

from quantcore.filtering.kalman import (
    dynamic_hedge_ratio,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


# Local linear trend model: state = (level, slope).
A = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.01
R = np.array([[0.5]])
X0 = np.array([0.0, 0.1])
P0 = np.eye(2) * 1.0


def _fit_filterpy_hedge_ratio(
    y: np.ndarray, x: np.ndarray, obs_var: float, proc_var: float, prior_mean: float
) -> np.ndarray:
    """Scalar filterpy filter with H_t = x_t, started from the prior N(prior_mean, 1)."""
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([prior_mean])
    kf.P = np.array([[1.0]])
    kf.F = np.array([[1.0]])
    kf.Q = np.array([[proc_var]])
    kf.R = np.array([[obs_var]])

    beta = np.empty(y.shape[0])
    for t in range(y.shape[0]):
        kf.H = np.array([[x[t]]])
        kf.predict()
        kf.update(np.array([y[t]]))
        beta[t] = kf.x[0]
    return beta


class TestDynamicHedgeRatioMatchesFilterpy:
    """dynamic_hedge_ratio is a scalar Kalman filter with a time-varying
    observation matrix H_t = x_t -- verified against filterpy.KalmanFilter
    with per-step H overrides, not merely a self-consistency check against
    quantcore's own formula. filterpy is started from the same prior,
    N(initial_beta, 1), so the comparison covers every element from the
    first."""

    @staticmethod
    def _drifting_pair() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(RNG_SEED + 2)
        n = 100
        x = rng.normal(size=n)
        true_beta = 1.0 + 0.01 * np.cumsum(rng.normal(scale=0.1, size=n))
        y = true_beta * x + rng.normal(scale=0.1, size=n)
        return y, x

    def test_matches_filterpy_with_time_varying_h(self) -> None:
        y, x = self._drifting_pair()
        obs_var, proc_var = 0.01, 1e-4

        quantcore_beta = dynamic_hedge_ratio(y, x, obs_var=obs_var, proc_var=proc_var)
        filterpy_beta = _fit_filterpy_hedge_ratio(y, x, obs_var, proc_var, prior_mean=0.0)

        np.testing.assert_allclose(quantcore_beta, filterpy_beta, atol=1e-10)

    @pytest.mark.parametrize("prior_mean", [-2.5, 0.7, "first_ratio"])
    def test_initial_beta_matches_filterpy_started_at_the_same_mean(
        self, prior_mean: float | str
    ) -> None:
        y, x = self._drifting_pair()
        obs_var, proc_var = 0.01, 1e-4
        mean = float(y[0] / x[0]) if prior_mean == "first_ratio" else float(prior_mean)

        quantcore_beta = dynamic_hedge_ratio(
            y, x, obs_var=obs_var, proc_var=proc_var, initial_beta=mean
        )
        filterpy_beta = _fit_filterpy_hedge_ratio(y, x, obs_var, proc_var, prior_mean=mean)

        np.testing.assert_allclose(quantcore_beta, filterpy_beta, atol=1e-10)
