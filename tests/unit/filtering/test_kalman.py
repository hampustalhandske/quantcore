"""Tests for the linear Kalman filter, RTS smoother, and dynamic hedge ratio."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.filtering.kalman import dynamic_hedge_ratio

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

    @pytest.mark.parametrize("bad_initial_beta", [np.nan, np.inf, -np.inf])
    def test_invalid_inputs_non_finite_initial_beta(self, bad_initial_beta: float) -> None:
        x = np.linspace(1.0, 10.0, 20)
        y = 2.0 * x
        with pytest.raises(ValueError):
            dynamic_hedge_ratio(y, x, initial_beta=bad_initial_beta)

    def test_initial_beta_is_keyword_only(self) -> None:
        x = np.linspace(1.0, 10.0, 20)
        y = 2.0 * x
        with pytest.raises(TypeError):
            dynamic_hedge_ratio(y, x, 1e-3, 1e-5, 2.0)  # type: ignore[misc]

    def test_first_two_values_follow_from_the_zero_mean_unit_variance_prior(self) -> None:
        # One predict/update step from beta ~ N(0, 1):
        #   P_pred = 1 + Q,  K = P_pred*x0 / (x0^2*P_pred + R),  beta_0 = K*y0,
        # then a second step from (beta_0, (1 - K*x0)*P_pred).
        y = np.array([5.14774703, 4.54901971])
        x = np.array([5.18905338, 4.47725156])
        obs_var, proc_var = 1e-3, 1e-5

        p_pred = 1.0 + proc_var
        gain = p_pred * x[0] / (x[0] ** 2 * p_pred + obs_var)
        beta_0 = gain * y[0]
        p_pred = (1.0 - gain * x[0]) * p_pred + proc_var
        gain = p_pred * x[1] / (x[1] ** 2 * p_pred + obs_var)
        beta_1 = beta_0 + gain * (y[1] - x[1] * beta_0)

        beta = dynamic_hedge_ratio(y, x, obs_var=obs_var, proc_var=proc_var)
        assert beta[0] == pytest.approx(beta_0, rel=1e-13)
        assert beta[1] == pytest.approx(beta_1, rel=1e-13)
        assert beta[0] == pytest.approx(0.9920028727, rel=1e-9)
        assert beta[1] == pytest.approx(1.0036758439, rel=1e-9)

    def test_default_prior_mean_is_zero(self) -> None:
        y = np.array([2.1, 3.9, 6.2, 7.9, 10.1, 12.2])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        default = dynamic_hedge_ratio(y, x)
        np.testing.assert_array_equal(default, dynamic_hedge_ratio(y, x, initial_beta=0.0))
        # The first value is pulled towards y[0] / x[0] = 2.1 but not onto it.
        assert default[0] == pytest.approx(2.1 * (1.0 - 1e-3 / (1.0 + 1e-5 + 1e-3)), rel=1e-13)

    def test_initial_beta_of_first_ratio_reproduces_data_seeded_values(self) -> None:
        # Starting the filter at y[0] / x[0] makes the first innovation zero, so the first
        # value is that ratio itself.
        y = np.array([2.1, 3.9, 6.2, 7.9, 10.1, 12.2])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = np.array(
            [
                2.1,
                1.979785520212058,
                2.036637415302432,
                2.001529957159298,
                2.011369014779288,
                2.0230073545386613,
            ]
        )
        beta = dynamic_hedge_ratio(y, x, obs_var=1e-3, proc_var=1e-5, initial_beta=y[0] / x[0])
        np.testing.assert_allclose(beta, expected, rtol=1e-14, atol=0.0)

    def test_prior_mean_is_forgotten_as_observations_arrive(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 60
        x = rng.normal(5.0, 1.0, size=n)
        y = 1.5 * x + rng.normal(0.0, 0.1, size=n)
        from_zero = dynamic_hedge_ratio(y, x)
        from_far_away = dynamic_hedge_ratio(y, x, initial_beta=-40.0)
        assert abs(from_far_away[0] - from_zero[0]) > 1e-4
        assert from_far_away[-1] == pytest.approx(from_zero[-1], abs=1e-12)

    @pytest.mark.parametrize("x_first", [0.0, 1e-12, 1e-200, 5e-324])
    def test_zero_or_tiny_first_regressor_does_not_distort_later_values(
        self, x_first: float
    ) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 40
        x = rng.normal(5.0, 1.0, size=n)
        y = 1.5 * x + rng.normal(0.0, 0.1, size=n)
        x[0], y[0] = x_first, 2.0

        beta = dynamic_hedge_ratio(y, x, obs_var=1e-3, proc_var=1e-5)
        assert np.all(np.isfinite(beta))
        assert abs(beta[0]) < 1e-6
        assert beta[1:] == pytest.approx(1.5, abs=0.05)

    def test_observation_with_zero_regressor_leaves_the_estimate_unchanged(self) -> None:
        y = np.array([2.1, 3.9, 5.0, 7.9, 10.1])
        x = np.array([1.0, 2.0, 0.0, 4.0, 5.0])
        beta = dynamic_hedge_ratio(y, x)
        assert beta[2] == beta[1]

        x_starting_at_zero = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
        assert dynamic_hedge_ratio(y, x_starting_at_zero)[0] == 0.0
        assert dynamic_hedge_ratio(y, x_starting_at_zero, initial_beta=0.7)[0] == 0.7

    def test_converges_toward_true_static_ratio(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 200
        x = np.linspace(1.0, 20.0, n)
        true_beta = 2.0
        y = true_beta * x + rng.normal(0.0, 0.05, size=n)

        beta = dynamic_hedge_ratio(y, x, obs_var=1e-3, proc_var=1e-5)
        tail = beta[-int(0.2 * n) :]
        assert tail.mean() == pytest.approx(true_beta, abs=0.05)


class TestDynamicHedgeRatioRejectsNonFiniteInput:
    """A NaN or infinite input raises `ValueError` instead of propagating into the estimates."""

    Y = np.array([1.0, 2.1, 2.9, 4.2, 5.0])
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("which", ["y", "x"])
    def test_non_finite_observation(self, which: str, bad: float) -> None:
        y, x = self.Y.copy(), self.X.copy()
        (y if which == "y" else x)[2] = bad
        with pytest.raises(ValueError, match=f"{which} must contain only finite values"):
            dynamic_hedge_ratio(y, x)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_observation_variance(self, bad: float) -> None:
        with pytest.raises(ValueError, match="obs_var must be strictly positive and finite"):
            dynamic_hedge_ratio(self.Y, self.X, obs_var=bad)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_process_variance(self, bad: float) -> None:
        with pytest.raises(ValueError, match="proc_var must be strictly positive and finite"):
            dynamic_hedge_ratio(self.Y, self.X, proc_var=bad)
