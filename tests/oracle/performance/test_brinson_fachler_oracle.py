"""Oracle test for `brinson_fachler_attribution` vs an independent hand derivation.

Covers Workstream D: "Brinson-Fachler vs a worked example from Bacon."

No specific numeric worked example from Bacon's *Practical Portfolio
Performance Measurement and Attribution* could be reliably sourced and
transcribed for this pass (unlike the He & Litterman Black-Litterman
example, which was read directly from the primary paper's PDF) -- rather
than risk pinning a fabricated "Bacon" table, this test instead uses a
clean, independently hand-worked two-segment example, with every effect
computed by a from-scratch, standalone application of the Brinson-Fachler
formulas (Brinson & Fachler 1985) -- not by calling any of quantcore's own
code -- and cross-checked against the model's own closed identity (segment
effects sum to the portfolio's total active return). This is the same
"internal-consistency + independent hand arithmetic" style used for
`component_var`'s oracle coverage, not a claim of matching a specific
published table.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.performance.attribution import brinson_fachler_attribution

pytestmark = pytest.mark.oracle


class TestBrinsonFachlerMatchesHandDerivedTwoSegmentExample:
    def test_per_segment_effects_and_totals(self) -> None:
        # Two segments: Equities, Bonds.
        portfolio_weights = np.array([0.6, 0.4])
        benchmark_weights = np.array([0.5, 0.5])
        portfolio_returns = np.array([0.08, 0.02])
        benchmark_returns = np.array([0.05, 0.01])

        # Hand-worked (independent of quantcore's implementation):
        #   benchmark total return B = 0.5*0.05 + 0.5*0.01 = 0.03
        #   A_1 = (0.6-0.5)*(0.05-0.03) = 0.002   A_2 = (0.4-0.5)*(0.01-0.03) = 0.002
        #   S_1 = 0.5*(0.08-0.05) = 0.015          S_2 = 0.5*(0.02-0.01) = 0.005
        #   I_1 = (0.6-0.5)*(0.08-0.05) = 0.003     I_2 = (0.4-0.5)*(0.02-0.01) = -0.001
        expected_allocation = np.array([0.002, 0.002])
        expected_selection = np.array([0.015, 0.005])
        expected_interaction = np.array([0.003, -0.001])

        result = brinson_fachler_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        np.testing.assert_allclose(result.allocation_effect, expected_allocation, atol=1e-12)
        np.testing.assert_allclose(result.selection_effect, expected_selection, atol=1e-12)
        np.testing.assert_allclose(result.interaction_effect, expected_interaction, atol=1e-12)

    def test_total_effects_reconcile_to_active_return(self) -> None:
        portfolio_weights = np.array([0.6, 0.4])
        benchmark_weights = np.array([0.5, 0.5])
        portfolio_returns = np.array([0.08, 0.02])
        benchmark_returns = np.array([0.05, 0.01])

        result = brinson_fachler_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        portfolio_total_return = float(np.sum(portfolio_weights * portfolio_returns))
        benchmark_total_return = float(np.sum(benchmark_weights * benchmark_returns))
        active_return = portfolio_total_return - benchmark_total_return

        total_effects = (
            result.total_allocation_effect
            + result.total_selection_effect
            + result.total_interaction_effect
        )

        assert active_return == pytest.approx(0.026, abs=1e-12)
        assert total_effects == pytest.approx(active_return, abs=1e-12)

    def test_three_segment_example_reconciles(self) -> None:
        # A second, larger example to guard against a defect that only
        # shows up with more than two segments (e.g. an off-by-one in a
        # loop or reduction).
        portfolio_weights = np.array([0.5, 0.3, 0.2])
        benchmark_weights = np.array([0.4, 0.4, 0.2])
        portfolio_returns = np.array([0.10, -0.02, 0.05])
        benchmark_returns = np.array([0.06, 0.00, 0.03])

        result = brinson_fachler_attribution(
            portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
        )

        portfolio_total_return = float(np.sum(portfolio_weights * portfolio_returns))
        benchmark_total_return = float(np.sum(benchmark_weights * benchmark_returns))
        active_return = portfolio_total_return - benchmark_total_return

        total_effects = (
            result.total_allocation_effect
            + result.total_selection_effect
            + result.total_interaction_effect
        )
        assert total_effects == pytest.approx(active_return, abs=1e-12)
