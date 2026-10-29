"""Tests for Black-Litterman posterior estimation."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.portfolio.black_litterman import black_litterman


class TestBlackLitterman:
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
    market_weights = np.array([0.6, 0.4])
    risk_aversion = 2.5
    tau = 0.025

    def test_invalid_inputs_non_square_cov_matrix(self) -> None:
        cov_matrix = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0]])
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02])
        omega = np.array([[1e-4]])
        with pytest.raises(ValueError):
            black_litterman(
                cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
            )

    def test_invalid_inputs_view_matrix_column_count_mismatch(self) -> None:
        P = np.array([[1.0, -1.0, 0.5]])
        Q = np.array([0.02])
        omega = np.array([[1e-4]])
        with pytest.raises(ValueError):
            black_litterman(
                self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
            )

    def test_invalid_inputs_q_length_mismatch(self) -> None:
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02, 0.03])
        omega = np.array([[1e-4]])
        with pytest.raises(ValueError):
            black_litterman(
                self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
            )

    def test_invalid_inputs_omega_shape_mismatch(self) -> None:
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02])
        omega = np.array([[1e-4, 0.0], [0.0, 1e-4]])
        with pytest.raises(ValueError):
            black_litterman(
                self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
            )

    def test_invalid_inputs_non_positive_tau(self) -> None:
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02])
        omega = np.array([[1e-4]])
        with pytest.raises(ValueError):
            black_litterman(
                self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, tau=0.0
            )

    def test_uninformative_view_matches_equilibrium_returns(self) -> None:
        pi = self.risk_aversion * self.cov_matrix @ self.market_weights
        P = np.array([[1.0, 0.0]])
        Q = np.array([pi[0]])
        omega = np.array([[1e8]])

        posterior_returns, _ = black_litterman(
            self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
        )

        assert posterior_returns == pytest.approx(pi, abs=1e-3)

    def test_strong_view_shifts_posterior_toward_view(self) -> None:
        pi = self.risk_aversion * self.cov_matrix @ self.market_weights
        strong_view_value = pi[0] + 0.10
        P = np.array([[1.0, 0.0]])
        Q = np.array([strong_view_value])
        omega = np.array([[1e-6]])

        posterior_returns, _ = black_litterman(
            self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
        )

        assert abs(posterior_returns[0] - strong_view_value) < abs(pi[0] - strong_view_value)

    def test_zero_market_weights_gives_zero_prior_and_finite_posterior(self) -> None:
        # market_weights need not sum to 1; an all-zero vector is a
        # deliberately supported "no informative prior" input (Pi = 0),
        # leaving the posterior driven entirely by the views.
        market_weights = np.zeros(2)
        pi = self.risk_aversion * self.cov_matrix @ market_weights
        assert pi == pytest.approx(np.zeros(2), abs=1e-12)

        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02])
        omega = np.array([[1e-4]])

        posterior_returns, posterior_cov = black_litterman(
            self.cov_matrix, market_weights, P, Q, omega, self.risk_aversion, self.tau
        )

        assert np.all(np.isfinite(posterior_returns))
        assert np.all(np.isfinite(posterior_cov))

    def test_posterior_cov_is_symmetric(self) -> None:
        P = np.array([[1.0, -1.0]])
        Q = np.array([0.02])
        omega = np.array([[1e-4]])

        _, posterior_cov = black_litterman(
            self.cov_matrix, self.market_weights, P, Q, omega, self.risk_aversion, self.tau
        )

        assert posterior_cov == pytest.approx(posterior_cov.T, abs=1e-10)
