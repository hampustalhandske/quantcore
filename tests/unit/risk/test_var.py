"""Tests for parametric VaR and CVaR (Expected Shortfall).

References:
    J.P. Morgan/Reuters (1996), RiskMetrics Technical Document — parametric
    (delta-normal) VaR: VaR_alpha = -(mu + z_alpha * sigma),
    z_alpha = Phi^-1(1 - alpha).

    Rockafellar & Uryasev (2000) — CVaR_alpha = E[Loss | Loss > VaR_alpha].
    For R ~ N(mu, sigma^2), the closed-form Gaussian expected shortfall is
    CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha), z_alpha as above.
    (Derived from the standard truncated-normal mean formula and verified
    numerically below against the well-known textbook value
    CVaR_0.95 ~= 2.063 for a standard normal N(0, 1).)

Note on ddof: neither docstring specifies whether sample standard deviation
uses ddof=0 or ddof=1. Tests below use a large synthetic sample so the two
conventions are numerically indistinguishable at the chosen tolerance,
sidestepping the ambiguity rather than assuming a convention.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quantcore.risk.var import conditional_value_at_risk, value_at_risk

RNG_SEED = 12345
N_LARGE = 100_000


# ---------------------------------------------------------------------------
# VaR: closed-form parametric formula, checked against independently
# computed sample mean/std of the same array (ddof-convention-agnostic due
# to large N).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
def test_var_matches_parametric_formula(confidence_level: float) -> None:
    rng = np.random.default_rng(RNG_SEED)
    returns = rng.normal(loc=0.0005, scale=0.02, size=N_LARGE)

    mu = returns.mean()
    sigma = returns.std()
    z_alpha = norm.ppf(1 - confidence_level)
    expected_var = -(mu + z_alpha * sigma)

    actual = value_at_risk(returns, confidence_level)
    assert actual == pytest.approx(expected_var, abs=1e-3)


def test_var_higher_confidence_implies_larger_loss_threshold() -> None:
    # Convention-independent invariant: as confidence_level increases, the
    # loss threshold not exceeded with that probability must be at least as
    # large (z_alpha = Phi^-1(1-alpha) grows as alpha grows).
    rng = np.random.default_rng(RNG_SEED)
    returns = rng.normal(loc=0.0, scale=0.02, size=5000)

    var_90 = value_at_risk(returns, 0.90)
    var_95 = value_at_risk(returns, 0.95)
    var_99 = value_at_risk(returns, 0.99)
    assert var_90 <= var_95 <= var_99


def test_var_zero_mean_zero_vol_degenerate_returns() -> None:
    # Degenerate distribution: all returns identical -> sigma=0, so
    # VaR_alpha = -mu exactly, regardless of confidence_level.
    returns = np.full(500, 0.01)
    actual = value_at_risk(returns, 0.95)
    assert actual == pytest.approx(-0.01, abs=1e-9)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.5])
def test_var_confidence_level_outside_unit_interval_raises(confidence_level: float) -> None:
    # z_alpha = Phi^-1(1 - alpha) is undefined (+/-inf or nan) outside
    # alpha in (0, 1); a well-behaved public API should reject this rather
    # than silently returning inf/nan.
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    with pytest.raises(ValueError):
        value_at_risk(returns, confidence_level)


def test_var_empty_returns_raises() -> None:
    with pytest.raises(ValueError):
        value_at_risk(np.array([]), 0.95)


# ---------------------------------------------------------------------------
# CVaR: invariant relative to VaR, plus closed-form Gaussian check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
def test_cvar_at_least_as_large_as_var(confidence_level: float) -> None:
    # CVaR is the mean of the tail beyond VaR, so it can never be smaller
    # than VaR itself -- this holds for any returns distribution.
    rng = np.random.default_rng(RNG_SEED)
    returns = rng.standard_t(df=5, size=5000) * 0.02  # fat-tailed, not normal

    var = value_at_risk(returns, confidence_level)
    cvar = conditional_value_at_risk(returns, confidence_level)
    assert cvar >= var - 1e-9


@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
def test_cvar_matches_gaussian_closed_form(confidence_level: float) -> None:
    rng = np.random.default_rng(RNG_SEED)
    returns = rng.normal(loc=0.0005, scale=0.02, size=N_LARGE)

    mu = returns.mean()
    sigma = returns.std()
    z_alpha = norm.ppf(1 - confidence_level)
    expected_cvar = -mu + sigma * norm.pdf(z_alpha) / (1 - confidence_level)

    actual = conditional_value_at_risk(returns, confidence_level)
    assert actual == pytest.approx(expected_cvar, abs=1e-3)


def test_cvar_standard_normal_textbook_value_95() -> None:
    # Well-known textbook value: expected shortfall of a standard normal at
    # the 95% confidence level is approximately 2.063. Tolerance is looser
    # here than test_cvar_matches_gaussian_closed_form above because this
    # compares against a fixed population-level constant rather than the
    # sample's own mu/sigma, so it also absorbs finite-sample noise.
    rng = np.random.default_rng(RNG_SEED)
    returns = rng.normal(loc=0.0, scale=1.0, size=N_LARGE)
    actual = conditional_value_at_risk(returns, 0.95)
    assert actual == pytest.approx(2.063, abs=3e-2)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.5])
def test_cvar_confidence_level_outside_unit_interval_raises(confidence_level: float) -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    with pytest.raises(ValueError):
        conditional_value_at_risk(returns, confidence_level)


def test_cvar_empty_returns_raises() -> None:
    with pytest.raises(ValueError):
        conditional_value_at_risk(np.array([]), 0.95)
