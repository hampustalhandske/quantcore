"""Tests for Ledoit-Wolf shrinkage."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.portfolio.covariance import _ledoit_wolf_shrinkage, ledoit_wolf_shrinkage


class TestLedoitWolfShrinkage:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ledoit_wolf_shrinkage(np.empty((0, 3)))

    def test_result_is_symmetric_positive_definite(self) -> None:
        rng = np.random.default_rng(42)
        cov = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
        returns = rng.multivariate_normal(np.zeros(3), cov, size=200)

        shrunk = ledoit_wolf_shrinkage(returns)
        assert shrunk.shape == (3, 3)
        assert np.max(np.abs(shrunk - shrunk.T)) < 1e-8
        eigenvalues = np.linalg.eigvalsh(shrunk)
        assert np.all(eigenvalues > 0.0)

    def test_shrinkage_reduces_or_maintains_condition_number(self) -> None:
        rng = np.random.default_rng(123)
        cov = np.array(
            [
                [1.0, 0.9, 0.85, 0.8, 0.75],
                [0.9, 1.0, 0.9, 0.85, 0.8],
                [0.85, 0.9, 1.0, 0.9, 0.85],
                [0.8, 0.85, 0.9, 1.0, 0.9],
                [0.75, 0.8, 0.85, 0.9, 1.0],
            ]
        )
        returns = rng.multivariate_normal(np.zeros(5), cov, size=200)

        sample = np.cov(returns, rowvar=False)
        shrunk = ledoit_wolf_shrinkage(returns)

        sample_eigs = np.linalg.eigvalsh(sample)
        shrunk_eigs = np.linalg.eigvalsh(shrunk)
        sample_cond = sample_eigs[-1] / sample_eigs[0]
        shrunk_cond = shrunk_eigs[-1] / shrunk_eigs[0]
        assert shrunk_cond <= sample_cond + 1e-9


def _looped_ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) Theorem 1 with the dispersion sum formed observation by observation."""
    num_obs, num_assets = returns.shape
    demeaned = returns - returns.mean(axis=0)
    sample = (demeaned.T @ demeaned) / num_obs
    target = float(np.trace(sample)) / num_assets * np.eye(num_assets)
    d_squared = float(np.sum((sample - target) ** 2)) / num_assets
    b_bar_squared = 0.0
    for t in range(num_obs):
        outer_t = np.outer(demeaned[t], demeaned[t])
        b_bar_squared += float(np.sum((outer_t - sample) ** 2)) / num_assets
    b_bar_squared /= num_obs**2
    alpha = 0.0 if d_squared == 0.0 else min(b_bar_squared, d_squared) / d_squared
    return alpha * target + (1.0 - alpha) * sample


class TestLedoitWolfClosedFormDispersion:
    @pytest.mark.parametrize(
        ("num_obs", "num_assets"),
        [(2, 1), (2, 5), (3, 8), (5, 20), (30, 4), (252, 10), (400, 60)],
    )
    def test_matches_looped_computation(self, num_obs: int, num_assets: int) -> None:
        rng = np.random.default_rng(num_obs * 1000 + num_assets)
        returns = rng.normal(scale=0.01, size=(num_obs, num_assets))
        np.testing.assert_allclose(
            _ledoit_wolf_shrinkage(returns),
            _looped_ledoit_wolf(returns),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_matches_looped_computation_with_constant_column(self) -> None:
        rng = np.random.default_rng(7)
        returns = rng.normal(scale=0.01, size=(50, 4))
        returns[:, 2] = 0.003
        np.testing.assert_allclose(
            _ledoit_wolf_shrinkage(returns),
            _looped_ledoit_wolf(returns),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_two_observations_give_finite_positive_semidefinite_result(self) -> None:
        returns = np.array([[0.01, -0.02, 0.005], [-0.01, 0.03, 0.0]])
        result = _ledoit_wolf_shrinkage(returns)
        assert np.all(np.isfinite(result))
        assert np.linalg.eigvalsh(result).min() >= -1e-15
