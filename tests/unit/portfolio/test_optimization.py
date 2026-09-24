"""Tests for mean-variance, risk-parity, and Kelly portfolio sizing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quantcore.portfolio.optimization import (
    kelly_fraction,
    l1_turnover_penalized_weights,
    mean_variance_weights,
    min_variance_weights,
    risk_parity_weights,
    unconstrained_mean_variance_weights,
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

    def test_allow_short_can_produce_a_negative_weight(self) -> None:
        # A strongly negative view on one asset, uncorrelated with the
        # other, should be expressed as a short position when allowed to.
        weights = mean_variance_weights(
            expected_returns=np.array([0.05, -0.20]),
            cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
            risk_aversion=1.0,
            allow_short=True,
        )
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights[1] < 0.0

    def test_allow_short_false_still_constrains_to_non_negative(self) -> None:
        weights = mean_variance_weights(
            expected_returns=np.array([0.05, -0.20]),
            cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
            risk_aversion=1.0,
            allow_short=False,
        )
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-8)


class TestRiskParityWeights:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.array([1.0, 2.0, 3.0]))

    def test_non_positive_variance_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_risk_contributions_are_equal(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T + 4 * np.eye(4)  # random PD matrix

        weights = risk_parity_weights(cov)
        contributions = weights * (cov @ weights) / (weights @ cov @ weights)
        np.testing.assert_allclose(contributions, 0.25, atol=1e-9)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(weights > 0.0)

    def test_one_dominant_variance_asset(self) -> None:
        # Asset 0's variance is ~2700x asset 1's; an equal-weight start on
        # the un-normalised least-squares objective used to terminate at
        # [0.5, 0.5] with asset 0 carrying 99.99% of the risk.
        cov = np.array([[4658.98781, -1.41048619], [-1.41048619, 1.72544360]])
        weights = risk_parity_weights(cov)
        contributions = weights * (cov @ weights) / (weights @ cov @ weights)
        np.testing.assert_allclose(contributions, 0.5, atol=1e-9)
        assert weights[0] < 0.05

    def test_diagonal_covariance_gives_inverse_volatility_weights(self) -> None:
        vols = np.array([0.1, 0.2, 0.4])
        weights = risk_parity_weights(np.diag(vols**2))
        expected = (1.0 / vols) / np.sum(1.0 / vols)
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_scale_invariant(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.normal(size=(5, 5))
        cov = a @ a.T + np.eye(5)
        np.testing.assert_allclose(
            risk_parity_weights(cov), risk_parity_weights(1e6 * cov), atol=1e-10
        )

    @pytest.mark.parametrize(
        "family",
        ["well_conditioned", "near_singular", "equal_vol", "one_dominant", "high_condition"],
    )
    def test_risk_contributions_equal_across_covariance_families(self, family: str) -> None:
        rng = np.random.default_rng(20240924)
        for _ in range(40):
            k = int(rng.integers(2, 11))
            if family == "well_conditioned":
                a = rng.normal(size=(k, k))
                cov = a @ a.T + 0.1 * np.eye(k)
            elif family == "near_singular":
                a = rng.normal(size=(k, 1))
                cov = a @ a.T + 1e-8 * np.eye(k)
            elif family == "equal_vol":
                corr = rng.uniform(-0.3, 0.9)
                cov = np.full((k, k), corr) + (1.0 - corr) * np.eye(k)
            elif family == "one_dominant":
                a = rng.normal(size=(k, k))
                cov = a @ a.T + np.eye(k)
                cov[0, 0] *= 1e4
            else:
                q, _ = np.linalg.qr(rng.normal(size=(k, k)))
                cov = q @ np.diag(np.logspace(-4, 4, k)) @ q.T
            weights = risk_parity_weights(cov)
            contributions = weights * (cov @ weights) / (weights @ cov @ weights)
            np.testing.assert_allclose(contributions, 1.0 / k, atol=1e-6)


class TestL1TurnoverPenalizedWeights:
    def test_invalid_inputs_mismatched_expected_returns_shape(self) -> None:
        with pytest.raises(ValueError):
            l1_turnover_penalized_weights(
                expected_returns=np.array([0.05, 0.07, 0.03]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                previous_weights=np.array([0.5, 0.5]),
                cost_bps=10.0,
                risk_aversion=2.0,
            )

    def test_invalid_inputs_mismatched_previous_weights_shape(self) -> None:
        with pytest.raises(ValueError):
            l1_turnover_penalized_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                previous_weights=np.array([0.5, 0.3, 0.2]),
                cost_bps=10.0,
                risk_aversion=2.0,
            )

    def test_invalid_inputs_previous_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError):
            l1_turnover_penalized_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                previous_weights=np.array([0.5, 0.3]),
                cost_bps=10.0,
                risk_aversion=2.0,
            )

    def test_invalid_inputs_negative_cost_bps(self) -> None:
        with pytest.raises(ValueError):
            l1_turnover_penalized_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                previous_weights=np.array([0.5, 0.5]),
                cost_bps=-1.0,
                risk_aversion=2.0,
            )

    def test_invalid_inputs_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            l1_turnover_penalized_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                previous_weights=np.array([0.5, 0.5]),
                cost_bps=10.0,
                risk_aversion=-1.0,
            )

    def test_zero_cost_recovers_mean_variance_weights(self) -> None:
        # SLSQP is run on a different (buy/sell slack) parameterization than
        # mean_variance_weights' direct-w parameterization, so the two solvers
        # converge to slightly different points near the shared optimum;
        # a loose-ish tolerance captures the equivalence without requiring
        # bitwise-identical convergence paths.
        expected_returns = np.array([0.05, 0.07, 0.03])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]])
        previous_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        mv_weights = mean_variance_weights(expected_returns, cov, risk_aversion=3.0)
        penalized_weights = l1_turnover_penalized_weights(
            expected_returns,
            cov,
            previous_weights,
            cost_bps=0.0,
            risk_aversion=3.0,
        )
        assert np.allclose(mv_weights, penalized_weights, atol=1e-3)

    def test_increasing_cost_bps_weakly_reduces_turnover(self) -> None:
        expected_returns = np.array([0.10, -0.02, 0.05])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]])
        previous_weights = np.array([0.2, 0.5, 0.3])

        turnovers = []
        for cost_bps in [0.0, 10.0, 100.0, 1000.0]:
            weights = l1_turnover_penalized_weights(
                expected_returns,
                cov,
                previous_weights,
                cost_bps=cost_bps,
                risk_aversion=2.0,
            )
            turnovers.append(np.abs(weights - previous_weights).sum())

        assert all(turnovers[i] >= turnovers[i + 1] - 1e-6 for i in range(len(turnovers) - 1))

    def test_weights_sum_to_one_and_non_negative(self) -> None:
        expected_returns = np.array([0.05, 0.07, 0.03])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]])
        for previous_weights in [
            np.array([1 / 3, 1 / 3, 1 / 3]),
            np.array([0.6, 0.1, 0.3]),
            np.array([0.0, 1.0, 0.0]),
        ]:
            for cost_bps in [0.0, 25.0, 250.0]:
                weights = l1_turnover_penalized_weights(
                    expected_returns,
                    cov,
                    previous_weights,
                    cost_bps=cost_bps,
                    risk_aversion=1.5,
                )
                assert weights.sum() == pytest.approx(1.0, abs=1e-6)
                assert np.all(weights >= -1e-8)

    def test_allow_short_can_produce_a_negative_weight(self) -> None:
        weights = l1_turnover_penalized_weights(
            expected_returns=np.array([0.05, -0.20]),
            cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
            previous_weights=np.array([0.5, 0.5]),
            cost_bps=1.0,
            risk_aversion=1.0,
            allow_short=True,
        )
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights[1] < 0.0

    def test_high_cost_starting_at_previous_weights_barely_moves(self) -> None:
        expected_returns = np.array([0.20, -0.10, 0.05])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]])
        previous_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        weights = l1_turnover_penalized_weights(
            expected_returns,
            cov,
            previous_weights,
            cost_bps=1_000_000.0,
            risk_aversion=1.0,
        )
        turnover = np.abs(weights - previous_weights).sum()
        assert turnover == pytest.approx(0.0, abs=1e-4)


class TestUnconstrainedMeanVarianceWeights:
    def test_invalid_inputs_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07, 0.03]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=2.0,
            )

    def test_invalid_inputs_non_square_cov_matrix(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                risk_aversion=2.0,
            )

    def test_invalid_inputs_zero_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=0.0,
            )

    def test_invalid_inputs_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            unconstrained_mean_variance_weights(
                expected_returns=np.array([0.05, 0.07]),
                cov_matrix=np.array([[0.04, 0.0], [0.0, 0.01]]),
                risk_aversion=-1.0,
            )

    def test_matches_analytical_solution_for_uncorrelated_assets(self) -> None:
        # For diagonal Sigma, the closed-form solution is
        # w_i = mu_i / (risk_aversion * sigma_i^2).
        expected_returns = np.array([0.05, 0.07])
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        risk_aversion = 3.0

        expected = np.array([0.05 / (3.0 * 0.04), 0.07 / (3.0 * 0.01)])
        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)

        assert weights == pytest.approx(expected, abs=1e-10)

    def test_satisfies_first_order_condition(self) -> None:
        expected_returns = np.array([0.05, 0.07, 0.03])
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.02]])
        risk_aversion = 2.0

        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)

        assert cov @ weights * risk_aversion == pytest.approx(expected_returns, abs=1e-8)

    def test_does_not_sum_to_one_and_differs_from_mean_variance_weights(self) -> None:
        expected_returns = np.array([0.05, -0.20])
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        risk_aversion = 1.0

        unconstrained = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)
        constrained = mean_variance_weights(expected_returns, cov, risk_aversion, allow_short=True)

        assert unconstrained.sum() != pytest.approx(1.0, abs=1e-6)
        assert not np.allclose(unconstrained, constrained, atol=1e-3)

    def test_matches_explicit_inverse_on_ill_conditioned_matrix(self) -> None:
        # Near-singular but still invertible: a strong cross-correlation
        # pushes the condition number high without breaking invertibility.
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        expected_returns = np.array([0.05, 0.06])
        risk_aversion = 2.0

        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)
        reference = np.linalg.inv(cov) @ expected_returns / risk_aversion

        assert weights == pytest.approx(reference, abs=1e-6)


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

    def test_clip_false_returns_unclamped_textbook_value(self) -> None:
        assert kelly_fraction(expected_return=10.0, variance=0.01, clip=False) == pytest.approx(
            1000.0
        )
        assert kelly_fraction(expected_return=-10.0, variance=0.01, clip=False) == pytest.approx(
            -1000.0
        )

    def test_clip_false_matches_clip_true_within_unit_interval(self) -> None:
        clipped = kelly_fraction(expected_return=0.02, variance=0.04)
        unclipped = kelly_fraction(expected_return=0.02, variance=0.04, clip=False)
        assert clipped == pytest.approx(unclipped)


class TestSlsqpConvergenceIsChecked:
    """min_variance_weights, mean_variance_weights and
    l1_turnover_penalized_weights must raise when SLSQP reports
    `success=False` rather than return `result.x` -- a silent fallback to a
    possibly infeasible or non-optimal point. risk_parity_weights has the
    same contract for its own Newton iteration.
    """

    def test_min_variance_weights_raises_on_slsqp_failure(self, monkeypatch) -> None:
        import quantcore.portfolio.optimization as opt_module

        fake_result = SimpleNamespace(
            success=False, status=4, message="fake failure", x=np.zeros(2)
        )
        monkeypatch.setattr(opt_module, "minimize", lambda *a, **k: fake_result)
        with pytest.raises(RuntimeError):
            min_variance_weights(np.eye(2))

    def test_mean_variance_weights_raises_on_slsqp_failure(self, monkeypatch) -> None:
        import quantcore.portfolio.optimization as opt_module

        fake_result = SimpleNamespace(
            success=False, status=4, message="fake failure", x=np.zeros(2)
        )
        monkeypatch.setattr(opt_module, "minimize", lambda *a, **k: fake_result)
        with pytest.raises(RuntimeError):
            mean_variance_weights(np.array([0.05, 0.03]), np.eye(2), risk_aversion=1.0)

    def test_l1_turnover_penalized_weights_raises_on_slsqp_failure(self, monkeypatch) -> None:
        import quantcore.portfolio.optimization as opt_module

        fake_result = SimpleNamespace(
            success=False, status=4, message="fake failure", x=np.zeros(4)
        )
        monkeypatch.setattr(opt_module, "minimize", lambda *a, **k: fake_result)
        with pytest.raises(RuntimeError):
            l1_turnover_penalized_weights(
                np.array([0.05, 0.03]),
                np.eye(2),
                previous_weights=np.array([0.5, 0.5]),
                cost_bps=10.0,
                risk_aversion=1.0,
            )

    def test_risk_parity_weights_raises_on_non_convergence(self, monkeypatch) -> None:
        # risk_parity_weights uses its own Newton iteration rather than
        # SLSQP; forcing an iteration budget of zero with no fallback
        # tolerance must raise rather than return the starting point.
        import quantcore.portfolio.optimization as opt_module

        monkeypatch.setattr(opt_module, "_RISK_PARITY_MAX_ITER", 0)
        monkeypatch.setattr(opt_module, "_RISK_PARITY_STALL_TOL", 0.0)
        with pytest.raises(RuntimeError):
            risk_parity_weights(np.array([[1.0, 0.5], [0.5, 2.0]]))
