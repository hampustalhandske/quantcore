"""Tests for Brinson-Fachler single-period performance attribution."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.performance.attribution import brinson_fachler_attribution


class TestBrinsonFachlerAttribution:
    def test_effects_sum_to_active_return_identity(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(2, 8)
            portfolio_weights = rng.dirichlet(np.ones(n))
            benchmark_weights = rng.dirichlet(np.ones(n))
            portfolio_returns = rng.normal(0.0, 0.05, size=n)
            benchmark_returns = rng.normal(0.0, 0.05, size=n)

            result = brinson_fachler_attribution(
                portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
            )

            per_segment_total = (
                result.allocation_effect + result.selection_effect + result.interaction_effect
            )
            expected_active_return = float(
                np.sum(portfolio_weights * portfolio_returns)
                - np.sum(benchmark_weights * benchmark_returns)
            )
            assert np.sum(per_segment_total) == pytest.approx(expected_active_return, abs=1e-12)
            assert (
                result.total_allocation_effect
                + result.total_selection_effect
                + result.total_interaction_effect
            ) == pytest.approx(expected_active_return, abs=1e-12)

    def test_matches_hand_computed_two_segment_example(self) -> None:
        # w_p = [0.6, 0.4], w_b = [0.5, 0.5], r_p = [0.10, 0.05], r_b = [0.08, 0.02]
        # r_b_total = 0.5*0.08 + 0.5*0.02 = 0.05
        # Allocation_1 = (0.6-0.5)*(0.08-0.05) = 0.1*0.03 = 0.003
        # Allocation_2 = (0.4-0.5)*(0.02-0.05) = -0.1*-0.03 = 0.003
        # Selection_1 = 0.5*(0.10-0.08) = 0.01
        # Selection_2 = 0.5*(0.05-0.02) = 0.015
        # Interaction_1 = (0.1)*(0.02) = 0.002
        # Interaction_2 = (-0.1)*(0.03) = -0.003
        # total active = (0.6*0.10+0.4*0.05) - 0.05 = 0.08 - 0.05 = 0.03
        portfolio_weights = np.array([0.6, 0.4])
        benchmark_weights = np.array([0.5, 0.5])
        portfolio_returns = np.array([0.10, 0.05])
        benchmark_returns = np.array([0.08, 0.02])

        result = brinson_fachler_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        assert result.allocation_effect == pytest.approx([0.003, 0.003])
        assert result.selection_effect == pytest.approx([0.01, 0.015])
        assert result.interaction_effect == pytest.approx([0.002, -0.003])
        assert result.total_allocation_effect == pytest.approx(0.006)
        assert result.total_selection_effect == pytest.approx(0.025)
        assert result.total_interaction_effect == pytest.approx(-0.001)

    def test_pure_allocation_case_has_zero_selection_and_interaction(self) -> None:
        portfolio_weights = np.array([0.7, 0.3])
        benchmark_weights = np.array([0.4, 0.6])
        returns = np.array([0.05, -0.02])

        result = brinson_fachler_attribution(portfolio_weights, benchmark_weights, returns, returns)

        assert result.selection_effect == pytest.approx([0.0, 0.0])
        assert result.interaction_effect == pytest.approx([0.0, 0.0])
        expected_active_return = float(
            np.sum(portfolio_weights * returns) - np.sum(benchmark_weights * returns)
        )
        assert result.total_allocation_effect == pytest.approx(expected_active_return)

    def test_pure_selection_case_has_zero_allocation_and_interaction(self) -> None:
        weights = np.array([0.5, 0.5])
        portfolio_returns = np.array([0.06, 0.01])
        benchmark_returns = np.array([0.03, 0.02])

        result = brinson_fachler_attribution(weights, weights, portfolio_returns, benchmark_returns)

        assert result.allocation_effect == pytest.approx([0.0, 0.0])
        assert result.interaction_effect == pytest.approx([0.0, 0.0])
        expected_active_return = float(
            np.sum(weights * portfolio_returns) - np.sum(weights * benchmark_returns)
        )
        assert result.total_selection_effect == pytest.approx(expected_active_return)

    def test_invalid_inputs(self) -> None:
        weights = np.array([0.5, 0.5])
        returns = np.array([0.05, 0.02])

        with pytest.raises(ValueError):
            brinson_fachler_attribution(np.array([0.6, 0.3, 0.1]), weights, returns, returns)

        with pytest.raises(ValueError):
            brinson_fachler_attribution(np.array([0.6, 0.5]), weights, returns, returns)

        with pytest.raises(ValueError):
            brinson_fachler_attribution(np.array([]), np.array([]), np.array([]), np.array([]))

        with pytest.raises(ValueError):
            brinson_fachler_attribution(weights, weights, np.array([0.05]), returns)
