"""Oracle tests for finding C5: `value_at_risk`/`conditional_value_at_risk` methods.

`method="historical"` verified against `empyrical.value_at_risk`/
`conditional_value_at_risk`. `method="cornish_fisher"` verified against a
hand derivation of the Cornish-Fisher (1938) expansion, cross-checked for
its skewness/kurtosis moment convention against `scipy.stats.skew`/
`kurtosis` (both `bias=True`, i.e. population moments) — no installed
oracle library implements Cornish-Fisher VaR directly (quantstats/
empyrical don't), matching the brief's "hand derivation" instruction for
this specific method.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import kurtosis, norm, skew

pytest.importorskip("empyrical")

import empyrical

from quantcore.risk.var import conditional_value_at_risk, value_at_risk

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _hand_derived_cornish_fisher_var(
    returns: np.ndarray, confidence_level: float, mean: float | None, volatility: float | None
) -> float:
    """Independent (not calling quantcore's `_cornish_fisher_z`) reimplementation."""
    mu = float(np.mean(returns)) if mean is None else mean
    sigma = float(np.std(returns, ddof=1)) if volatility is None else volatility
    z = float(norm.ppf(1.0 - confidence_level))
    s = float(skew(returns, bias=True))
    k = float(kurtosis(returns, bias=True))  # scipy's default is excess kurtosis
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )
    return float(-(mu + z_cf * sigma))


class TestHistoricalMethodMatchesEmpyrical:
    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_var_matches(self, confidence_level: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.standard_t(df=4, size=2000) * 0.01 + 0.001

        quantcore_var = value_at_risk(returns, confidence_level, method="historical")
        empyrical_var = empyrical.value_at_risk(returns, cutoff=1.0 - confidence_level)

        assert quantcore_var == pytest.approx(-empyrical_var, rel=1e-9)

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_cvar_matches(self, confidence_level: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.standard_t(df=4, size=2000) * 0.01 + 0.001

        quantcore_cvar = conditional_value_at_risk(returns, confidence_level, method="historical")
        empyrical_cvar = empyrical.conditional_value_at_risk(returns, cutoff=1.0 - confidence_level)

        assert quantcore_cvar == pytest.approx(-empyrical_cvar, rel=1e-9)

    def test_historical_method_rejects_mean_or_volatility_override(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        with pytest.raises(ValueError):
            value_at_risk(returns, 0.95, mean=0.0, method="historical")
        with pytest.raises(ValueError):
            value_at_risk(returns, 0.95, volatility=0.1, method="historical")


class TestCornishFisherMethodMatchesHandDerivation:
    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_var_matches_hand_derivation(self, confidence_level: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.standard_t(df=4, size=2000) * 0.01 + 0.001

        quantcore_var = value_at_risk(returns, confidence_level, method="cornish_fisher")
        expected = _hand_derived_cornish_fisher_var(returns, confidence_level, None, None)

        assert quantcore_var == pytest.approx(expected, rel=1e-9)

    def test_reduces_to_gaussian_for_a_near_normal_sample(self) -> None:
        # A large, genuinely-normal sample has skewness/excess kurtosis
        # near 0, so Cornish-Fisher's z_cf must be close to the plain
        # Gaussian z.
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(0.0005, 0.01, size=200_000)
        confidence_level = 0.95

        cf_var = value_at_risk(returns, confidence_level, method="cornish_fisher")
        gaussian_var = value_at_risk(returns, confidence_level, method="gaussian")

        assert cf_var == pytest.approx(gaussian_var, rel=1e-2)

    def test_var_monotonically_increases_with_confidence_level(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.standard_t(df=4, size=2000) * 0.01 + 0.001
        levels = [0.90, 0.95, 0.99]
        for method in ("gaussian", "historical", "cornish_fisher"):
            values = [value_at_risk(returns, cl, method=method) for cl in levels]
            assert all(a <= b for a, b in itertools.pairwise(values))

    def test_var_at_most_cvar_across_all_methods(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.standard_t(df=4, size=2000) * 0.01 + 0.001
        for method in ("gaussian", "historical", "cornish_fisher"):
            var = value_at_risk(returns, 0.95, method=method)
            cvar = conditional_value_at_risk(returns, 0.95, method=method)
            assert var <= cvar
