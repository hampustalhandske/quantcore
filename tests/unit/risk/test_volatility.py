"""Tests for the GARCH(1,1) conditional variance recursion.

Reference: Bollerslev (1986), "Generalized Autoregressive Conditional
Heteroskedasticity."

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

Open interface question: the docstring does not specify how sigma_0 (the
seed for the recursion) is initialized. Rather than assume a convention
(e.g. sample variance of `returns`, or the long-run variance
omega / (1 - alpha - beta)), the tests below verify the *recursion
relationship* between consecutive output values -- which holds regardless of
how sigma_0 was seeded -- and treat output[0] as a given.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.volatility import garch_11_variance

# ---------------------------------------------------------------------------
# Shape / well-formedness
# ---------------------------------------------------------------------------


def test_output_length_matches_input() -> None:
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.015])
    variance = garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=0.85)
    assert len(variance) == len(returns)


def test_variance_series_is_nonnegative() -> None:
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.015])
    variance = garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=0.85)
    assert np.all(variance >= 0.0)
    assert np.all(np.isfinite(variance))


def test_deterministic_no_randomness_involved() -> None:
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    v1 = garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=0.85)
    v2 = garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=0.85)
    assert np.array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Core correctness: the recursion relationship between consecutive outputs
# must hold exactly, given whatever sigma_0 = variance[0] the implementation
# produces. This is the strongest correctness check possible without pinning
# down the (unspecified) sigma_0 seeding convention.
# ---------------------------------------------------------------------------


def test_recursion_holds_between_consecutive_outputs() -> None:
    returns = np.array([0.02, -0.01, 0.03, 0.00, -0.02, 0.01, 0.015, -0.005])
    omega, alpha, beta = 0.00002, 0.10, 0.85

    variance = garch_11_variance(returns, omega=omega, alpha=alpha, beta=beta)

    expected_next = omega + alpha * returns[:-1] ** 2 + beta * variance[:-1]
    assert np.allclose(variance[1:], expected_next)


def test_recursion_holds_with_different_parameters() -> None:
    # Same check with a different (still stationary) parameterization to
    # make sure the check isn't coincidentally passing for one omega/alpha/
    # beta combination.
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.02, size=25)
    omega, alpha, beta = 0.00005, 0.05, 0.9

    variance = garch_11_variance(returns, omega=omega, alpha=alpha, beta=beta)

    expected_next = omega + alpha * returns[:-1] ** 2 + beta * variance[:-1]
    assert np.allclose(variance[1:], expected_next)


def test_variance_depends_only_on_squared_residuals() -> None:
    # The recursion only ever uses epsilon_{t-1}^2, so negating the entire
    # residual series must leave the conditional variance path unchanged
    # (any sensible sigma_0 seeding is also a function of squared/symmetric
    # statistics of `returns`).
    returns = np.array([0.02, -0.01, 0.03, 0.00, -0.02, 0.01])
    omega, alpha, beta = 0.00002, 0.10, 0.85

    variance_pos = garch_11_variance(returns, omega=omega, alpha=alpha, beta=beta)
    variance_neg = garch_11_variance(-returns, omega=omega, alpha=alpha, beta=beta)
    assert np.allclose(variance_pos, variance_neg)


def test_single_observation_boundary() -> None:
    returns = np.array([0.01])
    variance = garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=0.85)
    assert len(variance) == 1
    assert variance[0] >= 0.0
    assert np.isfinite(variance[0])


# ---------------------------------------------------------------------------
# Parameter domain validation, per the docstring: omega > 0, alpha >= 0,
# beta >= 0, and alpha + beta < 1 required for covariance stationarity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omega", [0.0, -0.001])
def test_non_positive_omega_raises(omega: float) -> None:
    returns = np.array([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        garch_11_variance(returns, omega=omega, alpha=0.1, beta=0.85)


def test_negative_alpha_raises() -> None:
    returns = np.array([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        garch_11_variance(returns, omega=0.0001, alpha=-0.1, beta=0.85)


def test_negative_beta_raises() -> None:
    returns = np.array([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        garch_11_variance(returns, omega=0.0001, alpha=0.1, beta=-0.85)


@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5), (0.6, 0.5), (1.0, 0.0)])
def test_non_stationary_alpha_plus_beta_raises(alpha: float, beta: float) -> None:
    # Docstring: "alpha + beta < 1 required for covariance stationarity."
    returns = np.array([0.01, -0.02, 0.015, 0.005])
    with pytest.raises(ValueError):
        garch_11_variance(returns, omega=0.0001, alpha=alpha, beta=beta)


def test_empty_returns_raises() -> None:
    with pytest.raises(ValueError):
        garch_11_variance(np.array([]), omega=0.0001, alpha=0.1, beta=0.85)
