"""Oracle tests for C1: `calmar_ratio` / `cagr` / `annualized_return` vs empyrical.

Confirmed finding C1 (see QUANTCORE_FIX_PROMPT.md): `calmar_ratio` used
arithmetic annualized return (`mean(r) * periods_per_year`), where
empyrical, QuantStats, and Young (1991) as commonly applied use compound
annual growth rate. Owner-approved breaking change: `calmar_ratio`'s
default `return_method` is now `"cagr"`; the prior arithmetic behavior is
available via `return_method="arithmetic"`.

Oracle: `empyrical.annual_return` (CAGR) and `empyrical.calmar_ratio`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("empyrical")

import empyrical

from quantcore.performance.metrics import (
    annualized_return,
    cagr,
    calmar_ratio,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _returns(n: int, seed: int, daily_vol: float = 0.01, drift: float = 0.0003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(drift, daily_vol, size=n)


class TestCagrMatchesEmpyricalAnnualReturn:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches(self, seed: int) -> None:
        returns = _returns(500, seed)
        expected = empyrical.annual_return(returns, period="daily")
        actual = cagr(returns, periods_per_year=252)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_annualized_return_cagr_method_matches(self) -> None:
        returns = _returns(500, RNG_SEED)
        expected = empyrical.annual_return(returns, period="daily")
        actual = annualized_return(returns, periods_per_year=252, method="cagr")
        assert actual == pytest.approx(expected, rel=1e-9)


class TestCalmarRatioMatchesEmpyrical:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_with_default_cagr_method(self, seed: int) -> None:
        returns = _returns(500, seed, drift=0.0005)
        expected = empyrical.calmar_ratio(returns, period="daily")
        actual = calmar_ratio(returns, periods_per_year=252)
        if np.isnan(expected):
            # empyrical returns NaN when max_dd >= 0 (no drawdown); quantcore
            # returns +inf for that case instead (documented, pre-existing
            # convention, unrelated to this finding) -- not comparable
            # directly, so just check quantcore's own invariant here.
            assert np.isinf(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_arithmetic_method_does_not_match_empyrical(self) -> None:
        # Demonstrates the fix: the old default (arithmetic) does NOT match
        # the industry-standard (CAGR) figure empyrical reports, for a
        # volatile-enough series that the two methods diverge.
        returns = _returns(1000, RNG_SEED, daily_vol=0.03, drift=0.001)
        expected = empyrical.calmar_ratio(returns, period="daily")
        arithmetic_actual = calmar_ratio(returns, return_method="arithmetic")
        cagr_actual = calmar_ratio(returns, return_method="cagr")
        assert cagr_actual == pytest.approx(expected, rel=1e-6)
        assert arithmetic_actual != pytest.approx(expected, rel=1e-3)
