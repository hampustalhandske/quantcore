"""Oracle tests for the GARCH-family conditional-variance recursions vs `arch`.

Covers Workstream D priority 1 (owner follow-up, decision 4): GARCH(1,1),
GJR-GARCH(1,1) (via `arch`'s `GARCH(o=1)`, the same Glosten-Jagannathan-
Runkle parameterization), and EGARCH(1,1) — including, per the brief's
explicit concern, the EGARCH asymmetry term's parameterization.

Method: fix parameters directly (via `arch`'s `.fix(params)`, which
evaluates the model at a given parameter vector without optimizing) rather
than comparing two independently-optimized fits against each other.
Fitting-to-fitting comparison is confounded by each optimizer's own local
convergence and by numerical scale sensitivity (both packages' docs warn
that data scaled far from O(1) hurts optimizer conditioning); comparing
the conditional-variance recursion and Gaussian log-likelihood at
identical, fixed parameters isolates exactly what needs checking here —
whether quantcore's formula and parameter ordering/sign convention match
the standard ones — without that confound.

The two recursions' seed values differ (quantcore seeds at the model's
theoretical unconditional variance; `arch` backcasts from early squared
residuals, a common alternative convention, also legitimate and
documented as such in quantcore's own docstrings) — both conventions
converge to the same path after a short burn-in, so comparisons below use
the tail of the series, after burn-in, plus a full-series log-likelihood
check (small burn-in differences wash out over hundreds of observations).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("arch")

from arch.univariate import EGARCH, GARCH, ZeroMean

from quantcore.risk.egarch import egarch_11_variance, gjr_garch_11_variance
from quantcore.risk.volatility import garch_11_variance

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921
BURN_IN = 300


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    variance = np.empty(n)
    epsilon = np.empty(n)
    variance[0] = omega / (1.0 - alpha - beta)
    epsilon[0] = np.sqrt(variance[0]) * z[0]
    for t in range(1, n):
        variance[t] = omega + alpha * epsilon[t - 1] ** 2 + beta * variance[t - 1]
        epsilon[t] = np.sqrt(variance[t]) * z[t]
    return epsilon


def _simulate_gjr_garch(
    n: int, omega: float, alpha: float, gamma: float, beta: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    variance = np.empty(n)
    epsilon = np.empty(n)
    variance[0] = omega / (1.0 - alpha - gamma / 2.0 - beta)
    epsilon[0] = np.sqrt(variance[0]) * z[0]
    for t in range(1, n):
        bad_news = 1.0 if epsilon[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * epsilon[t - 1] ** 2
            + gamma * epsilon[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )
        epsilon[t] = np.sqrt(variance[t]) * z[t]
    return epsilon


def _simulate_egarch(
    n: int, omega: float, alpha: float, gamma: float, beta: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    e_abs_z = np.sqrt(2.0 / np.pi)
    log_variance = np.empty(n)
    epsilon = np.empty(n)
    log_variance[0] = omega / (1.0 - beta)
    epsilon[0] = np.sqrt(np.exp(log_variance[0])) * z[0]
    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_variance[t - 1]))
        z_prev = epsilon[t - 1] / sigma_prev
        log_variance[t] = (
            omega + beta * log_variance[t - 1] + alpha * (np.abs(z_prev) - e_abs_z) + gamma * z_prev
        )
        epsilon[t] = np.sqrt(np.exp(log_variance[t])) * z[t]
    return epsilon


class TestGarch11VarianceMatchesArch:
    @pytest.mark.parametrize(
        ("omega", "alpha", "beta"),
        [(1e-5, 0.10, 0.85), (5e-6, 0.05, 0.90), (2e-5, 0.20, 0.60)],
    )
    def test_conditional_variance_matches_after_burn_in(
        self, omega: float, alpha: float, beta: float
    ) -> None:
        epsilon = _simulate_garch(600, omega, alpha, beta, seed=RNG_SEED)
        model = ZeroMean(epsilon, rescale=False)
        model.volatility = GARCH(p=1, o=0, q=1)
        arch_variance = model.fix([omega, alpha, beta]).conditional_volatility ** 2

        quantcore_variance = garch_11_variance(epsilon, omega, alpha, beta)

        np.testing.assert_allclose(quantcore_variance[BURN_IN:], arch_variance[BURN_IN:], rtol=1e-6)

    def test_full_series_log_likelihood_matches(self) -> None:
        omega, alpha, beta = 1e-5, 0.1, 0.85
        epsilon = _simulate_garch(2000, omega, alpha, beta, seed=RNG_SEED)
        model = ZeroMean(epsilon, rescale=False)
        model.volatility = GARCH(p=1, o=0, q=1)
        arch_loglik = model.fix([omega, alpha, beta]).loglikelihood

        variance = garch_11_variance(epsilon, omega, alpha, beta)
        quantcore_loglik = -0.5 * np.sum(
            np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance
        )

        # Small burn-in seeding difference washes out over 2000 observations.
        assert quantcore_loglik == pytest.approx(arch_loglik, rel=1e-3)


class TestGjrGarch11VarianceMatchesArch:
    """arch's GARCH(p=1, o=1, q=1) is the same Glosten-Jagannathan-Runkle
    parameterization as quantcore's gjr_garch_11_variance -- confirms the
    parameter order (omega, alpha, gamma, beta) and the "bad news" indicator
    convention agree."""

    @pytest.mark.parametrize(
        ("omega", "alpha", "gamma", "beta"),
        [(1e-5, 0.05, 0.10, 0.80), (5e-6, 0.02, 0.15, 0.75), (2e-5, 0.10, -0.05, 0.70)],
    )
    def test_conditional_variance_matches_after_burn_in(
        self, omega: float, alpha: float, gamma: float, beta: float
    ) -> None:
        epsilon = _simulate_gjr_garch(600, omega, alpha, gamma, beta, seed=RNG_SEED)
        model = ZeroMean(epsilon, rescale=False)
        model.volatility = GARCH(p=1, o=1, q=1)
        arch_variance = model.fix([omega, alpha, gamma, beta]).conditional_volatility ** 2

        quantcore_variance = gjr_garch_11_variance(epsilon, omega, alpha, gamma, beta)

        np.testing.assert_allclose(quantcore_variance[BURN_IN:], arch_variance[BURN_IN:], rtol=1e-6)


class TestEgarch11VarianceMatchesArchAsymmetryParameterization:
    """The brief's explicit concern: EGARCH's asymmetry term has several
    incompatible parameterizations across implementations. Confirms
    quantcore's alpha*(|z|-E|z|) + gamma*z (Nelson 1991) matches `arch`'s
    EGARCH exactly, not just up to a sign or scale factor."""

    @pytest.mark.parametrize(
        ("omega", "alpha", "gamma", "beta"),
        [(0.0, 0.10, -0.05, 0.90), (-0.05, 0.05, -0.10, 0.85), (0.02, 0.15, 0.00, 0.70)],
    )
    def test_conditional_variance_matches_after_burn_in(
        self, omega: float, alpha: float, gamma: float, beta: float
    ) -> None:
        epsilon = _simulate_egarch(600, omega, alpha, gamma, beta, seed=RNG_SEED)
        model = ZeroMean(epsilon, rescale=False)
        model.volatility = EGARCH(p=1, o=1, q=1)
        arch_variance = model.fix([omega, alpha, gamma, beta]).conditional_volatility ** 2

        quantcore_variance = np.exp(egarch_11_variance(epsilon, omega, alpha, gamma, beta))

        np.testing.assert_allclose(quantcore_variance[BURN_IN:], arch_variance[BURN_IN:], rtol=1e-6)
