"""Tests for EWMA covariance, Ledoit-Wolf shrinkage, and sample covariance."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.portfolio.covariance import (
    ewma_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)


class TestEwmaCovariance:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ewma_covariance(np.empty((0, 3)), lambda_=0.94)

    @pytest.mark.parametrize("bad_lambda", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_lambda_raises(self, bad_lambda: float) -> None:
        returns = np.array([[0.01], [-0.02], [0.03], [0.015]])
        with pytest.raises(ValueError):
            ewma_covariance(returns, lambda_=bad_lambda)

    def test_scalar_case_matches_manual_three_step_recursion(self) -> None:
        # k=1: "covariance" reduces to a scalar variance recursion, matching
        # the same seed-then-recurse pattern as risk/volatility.py's GARCH.
        r0, r1, r2, r3 = 0.02, -0.01, 0.03, 0.015
        returns = np.array([[r0], [r1], [r2], [r3]])
        lambda_ = 0.9

        sigma0 = float(np.var(returns[:, 0], ddof=1))
        sigma1 = lambda_ * sigma0 + (1 - lambda_) * r0**2
        sigma2 = lambda_ * sigma1 + (1 - lambda_) * r1**2
        sigma3 = lambda_ * sigma2 + (1 - lambda_) * r2**2

        result = ewma_covariance(returns, lambda_=lambda_)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(sigma3, abs=1e-10)

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(7)
        returns = rng.normal(size=(50, 4)) * 0.01
        result = ewma_covariance(returns, lambda_=0.94)
        assert result.shape == (4, 4)
        assert np.max(np.abs(result - result.T)) < 1e-10


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

        sample = sample_covariance(returns)
        shrunk = ledoit_wolf_shrinkage(returns)

        sample_eigs = np.linalg.eigvalsh(sample)
        shrunk_eigs = np.linalg.eigvalsh(shrunk)
        sample_cond = sample_eigs[-1] / sample_eigs[0]
        shrunk_cond = shrunk_eigs[-1] / shrunk_eigs[0]
        assert shrunk_cond <= sample_cond + 1e-9


class TestSampleCovariance:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            sample_covariance(np.empty((0, 3)))

    def test_too_few_observations_raises(self) -> None:
        returns = np.ones((3, 5))  # T=3 <= k=5
        with pytest.raises(ValueError):
            sample_covariance(returns)

    def test_matches_numpy_cov(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(size=(30, 4))
        expected = np.cov(returns.T, ddof=1)
        result = sample_covariance(returns)
        assert np.max(np.abs(result - expected)) < 1e-10
