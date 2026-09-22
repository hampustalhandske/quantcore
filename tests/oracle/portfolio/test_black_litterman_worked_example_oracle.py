"""Oracle test: `black_litterman` vs the He & Litterman (1999) worked example.

Covers Workstream D: "Black-Litterman vs a hand-worked example from He &
Litterman (1999)."

The exact numeric inputs and outputs are transcribed from Idzorek, T.M.
(2005 draft), "A Step-by-Step Guide to the Black-Litterman Model" (Tables
1, 2, 5, 6, and equations 7-8), which reproduces He & Litterman (1999)'s
eight-asset-class example (US Bonds, International Bonds, US Large
Growth, US Large Value, US Small Growth, US Small Value, International
Developed Equity, International Emerging Equity) with three investor
views. This is the standard worked example the finance literature cites
for validating a Black-Litterman implementation end to end.

Tolerance: the source table's published values are themselves rounded to
two decimal places (0.01 percentage points = 1e-4 in return units), so an
exact match isn't meaningful -- `abs=1e-3` (0.1 percentage points) is
generous enough to absorb that rounding while still being a real check
that the implementation reproduces the example, not just "close in some
loose sense."
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.portfolio.black_litterman import black_litterman, implied_equilibrium_returns

pytestmark = pytest.mark.oracle

# Table 5: covariance matrix of excess returns (annualized), 8 asset classes
# in the order: US Bonds, Int'l Bonds, US Large Growth, US Large Value,
# US Small Growth, US Small Value, Int'l Dev. Equity, Int'l Emerg. Equity.
COV_MATRIX = np.array(
    [
        [0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437],
        [0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535],
        [-0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039],
        [-0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854],
        [0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994],
        [0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235],
        [-0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064],
        [-0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.079958],
    ]
)
# Table 2: market capitalization weights.
MARKET_WEIGHTS = np.array([0.1934, 0.2613, 0.1209, 0.1209, 0.0134, 0.0134, 0.2418, 0.0349])
# Stated risk-aversion coefficient ("approximately 3.07").
RISK_AVERSION = 3.07
TAU = 0.025

# Table 1, "Implied Equilibrium Return Vector" column.
EXPECTED_PI = np.array([0.0008, 0.0067, 0.0641, 0.0408, 0.0743, 0.0370, 0.0480, 0.0660])

# Equation 7 (market-capitalization-weighted P) and the paper's Q, Omega.
# View 1 (absolute): Int'l Dev. Equity = 5.25%.
# View 2 (relative): Int'l Bonds outperforms US Bonds by 0.25%.
# View 3 (relative): US Large/Small Growth outperform US Large/Small Value by 2%.
P_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, -0.9, 0.1, -0.1, 0.0, 0.0],
    ]
)
Q_VIEWS = np.array([0.0525, 0.0025, 0.02])
OMEGA = np.diag([0.000709, 0.000141, 0.000866])

# Table 6, "New Combined Return Vector" column.
EXPECTED_POSTERIOR_RETURNS = np.array(
    [0.0007, 0.0050, 0.0650, 0.0432, 0.0759, 0.0394, 0.0493, 0.0684]
)


class TestImpliedEquilibriumReturnsMatchesHeLittermanExample:
    def test_matches_table_1(self) -> None:
        pi = implied_equilibrium_returns(COV_MATRIX, MARKET_WEIGHTS, RISK_AVERSION)
        np.testing.assert_allclose(pi, EXPECTED_PI, atol=1e-3)


class TestBlackLittermanMatchesHeLittermanExample:
    def test_posterior_returns_match_table_6(self) -> None:
        posterior_returns, _ = black_litterman(
            COV_MATRIX, MARKET_WEIGHTS, P_MATRIX, Q_VIEWS, OMEGA, RISK_AVERSION, TAU
        )
        np.testing.assert_allclose(posterior_returns, EXPECTED_POSTERIOR_RETURNS, atol=1e-3)

    def test_posterior_returns_are_between_equilibrium_and_views_for_view_1(self) -> None:
        # View 1 (absolute, Int'l Dev. Equity = 5.25%) is a well-known
        # sanity property of the model: since the equilibrium return
        # (4.80%) is below the view (5.25%), the posterior must land
        # strictly between them.
        posterior_returns, _ = black_litterman(
            COV_MATRIX, MARKET_WEIGHTS, P_MATRIX, Q_VIEWS, OMEGA, RISK_AVERSION, TAU
        )
        int_dev_equity_idx = 6
        pi = implied_equilibrium_returns(COV_MATRIX, MARKET_WEIGHTS, RISK_AVERSION)
        assert pi[int_dev_equity_idx] < posterior_returns[int_dev_equity_idx] < Q_VIEWS[0]
