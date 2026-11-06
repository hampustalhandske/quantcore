"""Tier-1 performance metrics: `sharpe_ratio` and `maximum_drawdown`.

`SR = mean(r - rf_p) / std(r - rf_p, ddof=1) * sqrt(p)` with `rf_p` the risk-free rate
de-annualised geometrically to the periodicity of `r`. Maximum drawdown is the largest peak-to-
trough fall of `cumprod(1 + r)` as a positive fraction in `[0, 1]`; `empyrical.max_drawdown`
reports it as a negative fraction, so the oracle flips its sign.
"""

import numpy as np
import numpy.typing as npt
from empyrical import max_drawdown as _empyrical_max_drawdown
from empyrical import sharpe_ratio as _empyrical_sharpe_ratio

import quantcore.performance as qc_performance
from conformance.compare import ConformanceTarget, ScalarComparator


def _sharpe_ratio_quantcore(
    returns: npt.NDArray[np.float64], risk_free_per_period: float, periods_per_year: int
) -> float:
    return float(qc_performance.sharpe_ratio(returns, risk_free_per_period, periods_per_year))


def _sharpe_ratio_reference(
    returns: npt.NDArray[np.float64], risk_free_per_period: float, periods_per_year: int
) -> float:
    return float(
        _empyrical_sharpe_ratio(
            returns, risk_free=risk_free_per_period, annualization=periods_per_year
        )
    )


_SHARPE_RATIO = ConformanceTarget(
    name="sharpe_ratio",
    tier=1,
    oracle="empyrical-reloaded",
    citation="Sharpe, W.F. (1966), 'Mutual Fund Performance.' Journal of Business, 39(1), 119-138.",
    compare=ScalarComparator(),
    quantcore=_sharpe_ratio_quantcore,
    reference=_sharpe_ratio_reference,
)


def _maximum_drawdown_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return float(qc_performance.maximum_drawdown(returns))


def _maximum_drawdown_reference(returns: npt.NDArray[np.float64]) -> float:
    return float(-_empyrical_max_drawdown(returns))


_MAXIMUM_DRAWDOWN = ConformanceTarget(
    name="maximum_drawdown",
    tier=1,
    oracle="empyrical-reloaded",
    citation="https://en.wikipedia.org/wiki/Drawdown_(economics); standard tear-sheet convention.",
    compare=ScalarComparator(),
    quantcore=_maximum_drawdown_quantcore,
    reference=_maximum_drawdown_reference,
)
