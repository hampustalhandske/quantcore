"""Tests for the linear Kalman filter, RTS smoother, and dynamic hedge ratio."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.filtering.kalman import dynamic_hedge_ratio, kalman_filter, kalman_smooth

RNG_SEED = 7


def _scalar_system(n_obs: int, q: float = 0.0, r: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "A": np.array([[1.0]]),
        "H": np.array([[1.0]]),
        "Q": np.array([[q]]),
        "R": np.array([[r]]),
        "x0": np.array([0.0]),
        "P0": np.array([[1.0]]),
    }


class TestKalmanFilter:
    def test_invalid_inputs_empty_observations(self) -> None:
        sys = _scalar_system(0)
        with pytest.raises(ValueError):
            kalman_filter(
                np.empty((0, 1)), sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
            )

    def test_invalid_inputs_mismatched_h_columns(self) -> None:
        # H has 2 state columns but A/x0/P0 describe a 1-dimensional state.
        observations = np.zeros((5, 1))
        A = np.array([[1.0]])
        H = np.array([[1.0, 1.0]])
        Q = np.array([[0.0]])
        R = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        with pytest.raises(ValueError):
            kalman_filter(observations, A, H, Q, R, x0, P0)

    def test_invalid_inputs_mismatched_observation_dimension(self) -> None:
        # observations has 2 columns but H maps to a 1-dimensional observation.
        observations = np.zeros((5, 2))
        sys = _scalar_system(5)
        with pytest.raises(ValueError):
            kalman_filter(
                observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
            )

    def test_output_shapes(self) -> None:
        sys = _scalar_system(4)
        observations = np.array([[1.0], [1.1], [0.9], [1.05]])
        means, covs = kalman_filter(
            observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
        )
        assert means.shape == (4, 1)
        assert covs.shape == (4, 1, 1)

    def test_constant_signal_zero_process_noise_converges_to_constant(self) -> None:
        constant = 3.0
        rng = np.random.default_rng(RNG_SEED)
        observations = (constant + rng.normal(0.0, 0.1, size=50)).reshape(-1, 1)
        sys = _scalar_system(50, q=0.0, r=0.1**2)
        means, _ = kalman_filter(
            observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
        )
        assert means[-1, 0] == pytest.approx(constant, abs=0.1)


class TestKalmanSmooth:
    def test_invalid_inputs_empty_observations(self) -> None:
        sys = _scalar_system(0)
        with pytest.raises(ValueError):
            kalman_smooth(
                np.empty((0, 1)), sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
            )

    def test_output_shapes(self) -> None:
        sys = _scalar_system(4)
        observations = np.array([[1.0], [1.1], [0.9], [1.05]])
        means, covs = kalman_smooth(
            observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
        )
        assert means.shape == (4, 1)
        assert covs.shape == (4, 1, 1)

    def test_smoothed_at_least_as_close_to_truth_as_filtered(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 60
        truth = np.linspace(0.0, 5.0, n)
        observations = (truth + rng.normal(0.0, 0.5, size=n)).reshape(-1, 1)
        sys = _scalar_system(n, q=0.01, r=0.25)

        filtered_means, _filtered_covs = kalman_filter(
            observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
        )
        smoothed_means, _ = kalman_smooth(
            observations, sys["A"], sys["H"], sys["Q"], sys["R"], sys["x0"], sys["P0"]
        )

        filtered_error = np.abs(filtered_means[:, 0] - truth).mean()
        smoothed_error = np.abs(smoothed_means[:, 0] - truth).mean()
        assert smoothed_error <= filtered_error + 1e-9


class TestDynamicHedgeRatio:
    def test_invalid_inputs_mismatched_lengths(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            dynamic_hedge_ratio(y, x)

    def test_invalid_inputs_empty_series(self) -> None:
        with pytest.raises(ValueError):
            dynamic_hedge_ratio(np.array([]), np.array([]))

    @pytest.mark.parametrize("bad_obs_var", [0.0, -1e-3])
    def test_invalid_inputs_non_positive_obs_var(self, bad_obs_var: float) -> None:
        x = np.linspace(1.0, 10.0, 20)
        y = 2.0 * x
        with pytest.raises(ValueError):
            dynamic_hedge_ratio(y, x, obs_var=bad_obs_var)

    @pytest.mark.parametrize("bad_proc_var", [0.0, -1e-3])
    def test_invalid_inputs_non_positive_proc_var(self, bad_proc_var: float) -> None:
        x = np.linspace(1.0, 10.0, 20)
        y = 2.0 * x
        with pytest.raises(ValueError):
            dynamic_hedge_ratio(y, x, proc_var=bad_proc_var)

    def test_output_shape(self) -> None:
        x = np.linspace(1.0, 10.0, 20)
        y = 2.0 * x
        beta = dynamic_hedge_ratio(y, x)
        assert beta.shape == (20,)

    def test_converges_toward_true_static_ratio(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 200
        x = np.linspace(1.0, 20.0, n)
        true_beta = 2.0
        y = true_beta * x + rng.normal(0.0, 0.05, size=n)

        beta = dynamic_hedge_ratio(y, x, obs_var=1e-3, proc_var=1e-5)
        tail = beta[-int(0.2 * n) :]
        assert tail.mean() == pytest.approx(true_beta, abs=0.05)
