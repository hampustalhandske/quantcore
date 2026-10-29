"""Tests for minimum-variance, unconstrained mean-variance and risk-parity weights."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from quantcore.portfolio.optimization import (
    risk_parity_weights,
    unconstrained_mean_variance_weights,
)


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

    def test_matches_explicit_inverse_on_ill_conditioned_matrix(self) -> None:
        # Near-singular but still invertible: a strong cross-correlation
        # pushes the condition number high without breaking invertibility.
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        expected_returns = np.array([0.05, 0.06])
        risk_aversion = 2.0

        weights = unconstrained_mean_variance_weights(expected_returns, cov, risk_aversion)
        reference = np.linalg.inv(cov) @ expected_returns / risk_aversion

        assert weights == pytest.approx(reference, abs=1e-6)


FloatArray = npt.NDArray[np.float64]


_FINITE_COV = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])
_FINITE_MU = np.array([0.05, 0.03, 0.02])
_FINITE_PREVIOUS = np.array([0.2, 0.3, 0.5])


def _with_entry(array: np.ndarray, index: tuple[int, ...], value: float) -> np.ndarray:
    changed = array.copy()
    changed[index] = value
    return changed


class TestNonFiniteInputsAreRejected:
    """A NaN or infinite input raises `ValueError` instead of producing weights."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda cov: unconstrained_mean_variance_weights(_FINITE_MU, cov, 3.0),
            lambda cov: risk_parity_weights(cov),
        ],
        ids=["unconstrained", "risk_parity"],
    )
    def test_non_finite_covariance_entry(self, solve, bad):
        cov = _with_entry(_FINITE_COV, (0, 1), bad)
        cov[1, 0] = bad
        with pytest.raises(ValueError, match="cov_matrix must contain only finite values"):
            solve(cov)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda mu: unconstrained_mean_variance_weights(mu, _FINITE_COV, 3.0),
        ],
        ids=["unconstrained"],
    )
    def test_non_finite_expected_return(self, solve, bad):
        with pytest.raises(ValueError, match="expected_returns must contain only finite values"):
            solve(_with_entry(_FINITE_MU, (1,), bad))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    @pytest.mark.parametrize(
        "solve",
        [
            lambda gamma: unconstrained_mean_variance_weights(_FINITE_MU, _FINITE_COV, gamma),
        ],
        ids=["unconstrained"],
    )
    def test_non_finite_risk_aversion(self, solve, bad):
        with pytest.raises(ValueError, match="risk_aversion must be finite"):
            solve(bad)
