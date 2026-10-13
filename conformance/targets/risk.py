"""Tier-1 risk metrics: parametric Gaussian Value at Risk and Conditional VaR.

Both use the sample mean and `ddof=1` standard deviation of the returns and are reported as
positive losses: `VaR = -(mu + z_alpha * sigma)`, `CVaR = -mu + sigma * phi(z_alpha) / (1 - alpha)`
with `z_alpha = Phi^-1(1 - alpha)`. Cornish-Fisher VaR is the same VaR formula with `z_alpha`
replaced by its skewness/excess-kurtosis-adjusted quantile.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

import quantcore.risk as qc_risk
from conformance.compare import ConformanceTarget, ScalarComparator


def _value_at_risk_quantcore(returns: npt.NDArray[np.float64], confidence_level: float) -> float:
    return float(qc_risk.value_at_risk(returns, confidence_level, method="gaussian"))


def _value_at_risk_reference(returns: npt.NDArray[np.float64], confidence_level: float) -> float:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    return float(-(mu + z_alpha * sigma))


_VALUE_AT_RISK = ConformanceTarget(
    name="value_at_risk",
    tier=1,
    oracle="closed form (scipy.stats.norm)",
    citation="J.P. Morgan/Reuters (1996), RiskMetrics Technical Document, 4th ed.",
    compare=ScalarComparator(),
    quantcore=_value_at_risk_quantcore,
    reference=_value_at_risk_reference,
)


def _conditional_value_at_risk_quantcore(
    returns: npt.NDArray[np.float64], confidence_level: float
) -> float:
    return float(qc_risk.conditional_value_at_risk(returns, confidence_level, method="gaussian"))


def _conditional_value_at_risk_reference(
    returns: npt.NDArray[np.float64], confidence_level: float
) -> float:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    pdf_z = float(norm.pdf(z_alpha))
    return float(-mu + sigma * pdf_z / (1.0 - confidence_level))


_CONDITIONAL_VALUE_AT_RISK = ConformanceTarget(
    name="conditional_value_at_risk",
    tier=1,
    oracle="closed form (scipy.stats.norm)",
    citation=(
        "Rockafellar, R.T. and Uryasev, S. (2000), 'Optimization of Conditional Value-at-Risk.' "
        "Journal of Risk, 2(3), 21-41."
    ),
    compare=ScalarComparator(),
    quantcore=_conditional_value_at_risk_quantcore,
    reference=_conditional_value_at_risk_reference,
)


def _cornish_fisher_z(returns: npt.NDArray[np.float64], z: float) -> float:
    """Cornish-Fisher (1938) skewness/excess-kurtosis-adjusted normal quantile.

    Uses population-moment (``bias=True``) skewness and excess kurtosis, matching
    ``scipy.stats.skew``/``kurtosis``'s default convention: reduces to ``z`` exactly when both
    are 0.
    """
    demeaned = returns - returns.mean()
    sigma = float(returns.std(ddof=0))
    skewness = float(np.mean(demeaned**3)) / sigma**3
    excess_kurtosis = float(np.mean(demeaned**4)) / sigma**4 - 3.0
    return float(
        z
        + (z**2 - 1.0) * skewness / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skewness**2 / 36.0
    )


def _value_at_risk_cornish_fisher_quantcore(
    returns: npt.NDArray[np.float64], confidence_level: float
) -> float:
    return float(qc_risk.value_at_risk(returns, confidence_level, method="cornish_fisher"))


def _value_at_risk_cornish_fisher_reference(
    returns: npt.NDArray[np.float64], confidence_level: float
) -> float:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = float(norm.ppf(1.0 - confidence_level))
    z_cf = _cornish_fisher_z(returns, z)
    return float(-(mu + z_cf * sigma))


_VALUE_AT_RISK_CORNISH_FISHER = ConformanceTarget(
    name="value_at_risk_cornish_fisher",
    tier=1,
    oracle="hand-computed (Cornish-Fisher 1938 expansion)",
    citation=(
        "Cornish, E.A. and Fisher, R.A. (1938), 'Moments and Cumulants in the Specification of "
        "Distributions.' Revue de l'Institut International de Statistique, 5(4), 307-320; "
        "Zangari, P. (1996), 'An Improved Methodology for Measuring VaR.' RiskMetrics Monitor."
    ),
    compare=ScalarComparator(),
    quantcore=_value_at_risk_cornish_fisher_quantcore,
    reference=_value_at_risk_cornish_fisher_reference,
)
