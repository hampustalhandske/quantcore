"""Oracle tests for `kalman_filter`/`kalman_smooth` vs `filterpy`.

Covers Workstream D priority 3 (owner follow-up, decision 4).

Oracle: `filterpy.kalman.KalmanFilter` (predict/update forward pass,
`rts_smoother` backward pass).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("filterpy")

from filterpy.kalman import KalmanFilter

from quantcore.filtering.kalman import (
    dynamic_hedge_ratio,
    kalman_filter,
    kalman_smooth,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _simulate_local_linear_trend(
    n: int, a: np.ndarray, h: np.ndarray, q: np.ndarray, r: np.ndarray, x0: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim_z = h.shape[0]
    x = x0.copy()
    observations = np.empty((n, dim_z))
    for t in range(n):
        x = a @ x + rng.multivariate_normal(np.zeros_like(x), q)
        observations[t] = h @ x + rng.multivariate_normal(np.zeros(dim_z), r)
    return observations


def _fit_filterpy(
    observations: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dim_x = x0.shape[0]
    dim_z = h.shape[0]
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.x = x0.copy()
    kf.P = p0.copy()
    kf.F = a
    kf.H = h
    kf.Q = q
    kf.R = r

    means = np.empty((observations.shape[0], dim_x))
    covs = np.empty((observations.shape[0], dim_x, dim_x))
    for t in range(observations.shape[0]):
        kf.predict()
        kf.update(observations[t])
        means[t] = kf.x
        covs[t] = kf.P
    return means, covs


# Local linear trend model: state = (level, slope).
A = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.01
R = np.array([[0.5]])
X0 = np.array([0.0, 0.1])
P0 = np.eye(2) * 1.0


class TestKalmanFilterMatchesFilterpy:
    def test_filtered_means_and_covariances_match(self) -> None:
        observations = _simulate_local_linear_trend(50, A, H, Q, R, X0, seed=RNG_SEED)

        quantcore_means, quantcore_covs = kalman_filter(observations, A, H, Q, R, X0, P0)
        filterpy_means, filterpy_covs = _fit_filterpy(observations, A, H, Q, R, X0, P0)

        np.testing.assert_allclose(quantcore_means, filterpy_means, atol=1e-10)
        np.testing.assert_allclose(quantcore_covs, filterpy_covs, atol=1e-10)

    def test_multivariate_observation_matches(self) -> None:
        # 2D state observed through a 2x2 identity-like H (two noisy sensors
        # of the same underlying level/slope state), to also exercise the
        # multivariate observation-noise-covariance inverse.
        h = np.array([[1.0, 0.0], [1.0, 0.5]])
        r = np.array([[0.5, 0.05], [0.05, 0.3]])
        observations = _simulate_local_linear_trend(40, A, h, Q, r, X0, seed=RNG_SEED + 1)

        quantcore_means, quantcore_covs = kalman_filter(observations, A, h, Q, r, X0, P0)
        filterpy_means, filterpy_covs = _fit_filterpy(observations, A, h, Q, r, X0, P0)

        np.testing.assert_allclose(quantcore_means, filterpy_means, atol=1e-8)
        np.testing.assert_allclose(quantcore_covs, filterpy_covs, atol=1e-8)


class TestKalmanSmoothMatchesFilterpyRtsSmoother:
    def test_smoothed_means_and_covariances_match(self) -> None:
        observations = _simulate_local_linear_trend(50, A, H, Q, R, X0, seed=RNG_SEED)

        quantcore_means, quantcore_covs = kalman_smooth(observations, A, H, Q, R, X0, P0)

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = X0.copy()
        kf.P = P0.copy()
        kf.F = A
        kf.H = H
        kf.Q = Q
        kf.R = R
        filtered_means = []
        filtered_covs = []
        for t in range(observations.shape[0]):
            kf.predict()
            kf.update(observations[t])
            filtered_means.append(kf.x.copy())
            filtered_covs.append(kf.P.copy())
        filterpy_smoothed_means, filterpy_smoothed_covs, _, _ = kf.rts_smoother(
            np.array(filtered_means), np.array(filtered_covs)
        )

        np.testing.assert_allclose(quantcore_means, filterpy_smoothed_means, atol=1e-8)
        np.testing.assert_allclose(quantcore_covs, filterpy_smoothed_covs, atol=1e-8)


class TestDynamicHedgeRatioMatchesFilterpy:
    """dynamic_hedge_ratio is a scalar Kalman filter with a time-varying
    observation matrix H_t = x_t -- verified against filterpy.KalmanFilter
    with per-step H overrides, not merely a self-consistency check against
    quantcore's own formula."""

    def test_matches_filterpy_with_time_varying_h(self) -> None:
        rng = np.random.default_rng(RNG_SEED + 2)
        n = 100
        x = rng.normal(size=n)
        true_beta = 1.0 + 0.01 * np.cumsum(rng.normal(scale=0.1, size=n))
        y = true_beta * x + rng.normal(scale=0.1, size=n)

        obs_var, proc_var = 0.01, 1e-4
        quantcore_beta = dynamic_hedge_ratio(y, x, obs_var=obs_var, proc_var=proc_var)

        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([y[0] / x[0]])
        kf.P = np.array([[1.0]])
        kf.F = np.array([[1.0]])
        kf.Q = np.array([[proc_var]])
        kf.R = np.array([[obs_var]])

        filterpy_beta = np.empty(n)
        for t in range(n):
            kf.H = np.array([[x[t]]])
            kf.predict()
            kf.update(np.array([y[t]]))
            filterpy_beta[t] = kf.x[0]

        np.testing.assert_allclose(quantcore_beta, filterpy_beta, atol=1e-10)
