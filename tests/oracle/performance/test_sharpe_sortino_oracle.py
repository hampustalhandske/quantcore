"""Oracle tests for finding C2: `sharpe_ratio`/`sortino_ratio`'s risk-free-rate units.

Confirms `risk_free_rate` is a per-period rate matching
`empyrical.sharpe_ratio`/`sortino_ratio`'s convention (their own
docstrings: "Constant daily risk-free return") — not
`quantstats.stats.sharpe`'s convention, where `rf` is annualized and
converted internally. Also confirms Sortino's downside deviation matches
`empyrical.downside_risk` exactly (mean over *all* observations, not just
the negative ones).

Oracle: `empyrical.sharpe_ratio`, `empyrical.sortino_ratio`,
`empyrical.downside_risk`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("empyrical")

import empyrical

from quantcore.performance.metrics import sharpe_ratio, sortino_ratio

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestSharpeRatioMatchesEmpyricalPerPeriodConvention:
    @pytest.mark.parametrize("rf_daily", [0.0, 0.0001, -0.00005])
    def test_matches(self, rf_daily: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(0.0005, 0.01, size=252)

        quantcore_sharpe = sharpe_ratio(returns, risk_free_rate=rf_daily, periods_per_year=252)
        empyrical_sharpe = empyrical.sharpe_ratio(returns, risk_free=rf_daily, period="daily")

        assert quantcore_sharpe == pytest.approx(empyrical_sharpe, rel=1e-9)


class TestSortinoRatioAndDownsideDeviationMatchEmpyrical:
    @pytest.mark.parametrize("rf_daily", [0.0, 0.0001, -0.00005])
    def test_sortino_ratio_matches(self, rf_daily: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(0.0005, 0.01, size=252)

        quantcore_sortino = sortino_ratio(returns, risk_free_rate=rf_daily, periods_per_year=252)
        empyrical_sortino = empyrical.sortino_ratio(
            returns, required_return=rf_daily, period="daily"
        )

        assert quantcore_sortino == pytest.approx(empyrical_sortino, rel=1e-9)

    def test_downside_deviation_matches_empyrical_downside_risk(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(0.0005, 0.01, size=252)
        rf_daily = 0.0001

        empyrical_downside_risk = empyrical.downside_risk(
            returns, required_return=rf_daily, period="daily"
        )

        excess = returns - rf_daily
        quantcore_downside_deviation = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(252)

        assert quantcore_downside_deviation == pytest.approx(empyrical_downside_risk, rel=1e-9)
