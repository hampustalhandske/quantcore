"""Integration: sanity relationship between VaR and CVaR computed on the
same returns sample, across a variety of return distributions and
confidence levels.

CVaR (Expected Shortfall) is, by construction (Rockafellar & Uryasev, 2000),
the expectation of losses in excess of the VaR threshold. Regardless of the
underlying return distribution, this means CVaR_alpha >= VaR_alpha must hold
for every valid confidence_level. This module exercises that relationship
end to end -- using both a Gaussian sample (where value_at_risk's parametric
assumption is exactly satisfied) and a fat-tailed sample (where it is not)
-- as a cross-check that the two risk/var.py functions agree with each
other, not just with their own individual closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.var import conditional_value_at_risk, value_at_risk

CONFIDENCE_LEVELS = [0.90, 0.95, 0.975, 0.99]


@pytest.mark.parametrize("confidence_level", CONFIDENCE_LEVELS)
def test_cvar_ge_var_for_normal_returns(confidence_level: float) -> None:
    rng = np.random.default_rng(2025)
    returns = rng.normal(loc=0.0004, scale=0.018, size=10_000)

    var = value_at_risk(returns, confidence_level)
    cvar = conditional_value_at_risk(returns, confidence_level)
    assert cvar >= var


@pytest.mark.parametrize("confidence_level", CONFIDENCE_LEVELS)
def test_cvar_ge_var_for_fat_tailed_returns(confidence_level: float) -> None:
    rng = np.random.default_rng(2025)
    returns = rng.standard_t(df=3, size=10_000) * 0.015

    var = value_at_risk(returns, confidence_level)
    cvar = conditional_value_at_risk(returns, confidence_level)
    assert cvar >= var


def test_var_and_cvar_both_increase_with_confidence_level() -> None:
    rng = np.random.default_rng(2025)
    returns = rng.normal(loc=0.0, scale=0.02, size=10_000)

    vars_ = [value_at_risk(returns, cl) for cl in CONFIDENCE_LEVELS]
    cvars = [conditional_value_at_risk(returns, cl) for cl in CONFIDENCE_LEVELS]

    assert vars_ == sorted(vars_)
    assert cvars == sorted(cvars)
