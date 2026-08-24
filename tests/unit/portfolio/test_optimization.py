"""Tests for mean-variance, risk-parity, and Kelly portfolio sizing."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.portfolio.optimization import (
    kelly_fraction,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
)


class TestMinVarianceWeights:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            min_variance_weights(np.array([1.0, 2.0, 3.0]))

    def test_invalid_inputs_non_square(self) -> None:
        with pytest.raises(ValueError):
            min_variance_weights(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_weights_sum_to_one_and_non_negative(self) -> None:
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        weights = min_variance_weights(cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-8)

    def test_matches_analytical_solution_for_uncorrelated_assets(self) -> None:
        # For diagonal Sigma, the unconstrained min-variance weights are
        # w_i proportional to 1 / sigma_i^2.
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        expected = np.array([1.0 / 0.04, 1.0 / 0.01])
        expected = expected / expected.sum()
        weights = min_variance_weights(cov)
        assert weights[0] == pytest.approx(expected[0], abs=0.02)
        assert weights[1] == pytest.approx(expected[1], abs=0.02)


class TestMeanVarianceWeights:
    def test_invalid_inputs_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            mean_variance_weights(
                expected_returns=np.array([0.05, 0.07, 0.03]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=2.0,
            )

    def test_invalid_inputs_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=-1.0,
            )

    def test_weights_sum_to_one_and_non_negative(self) -> None:
        weights = mean_variance_weights(
            expected_returns=np.array([0.05, 0.07]),
            cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
            risk_aversion=3.0,
        )
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-8)


class TestRiskParityWeights:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.array([1.0, 2.0, 3.0]))

    def test_risk_contributions_are_equal(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T + 4 * np.eye(4)  # random PD matrix

        weights = risk_parity_weights(cov)
        marginal = cov @ weights
        contributions = weights * marginal
        assert contributions.max() - contributions.min() == pytest.approx(0.0, abs=1e-4)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestKellyFraction:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            kelly_fraction(expected_return=0.05, variance=0.0)

    def test_invalid_inputs_negative_variance(self) -> None:
        with pytest.raises(ValueError):
            kelly_fraction(expected_return=0.05, variance=-0.01)

    @pytest.mark.parametrize(
        ("expected_return", "variance", "expected"),
        [
            (0.02, 0.04, 0.5),
            (0.01, 0.1, 0.1),
            (-0.02, 0.04, -0.5),
        ],
    )
    def test_matches_closed_form(
        self, expected_return: float, variance: float, expected: float
    ) -> None:
        assert kelly_fraction(expected_return, variance) == pytest.approx(expected, abs=1e-9)

    def test_clamps_to_unit_interval(self) -> None:
        assert kelly_fraction(expected_return=10.0, variance=0.01) == pytest.approx(1.0)
        assert kelly_fraction(expected_return=-10.0, variance=0.01) == pytest.approx(-1.0)
