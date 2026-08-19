"""Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).

VaR at confidence level alpha is the loss threshold not exceeded with
probability alpha:

    P(Loss > VaR_alpha) = 1 - alpha

``value_at_risk`` implements the parametric (delta-normal) VaR from the
RiskMetrics methodology, which assumes portfolio returns R ~ N(mu, sigma^2):

    VaR_alpha = -(mu + z_alpha * sigma)

where z_alpha = Phi^-1(1 - alpha) is the standard normal quantile.

CVaR (Expected Shortfall) is the expected loss conditional on exceeding VaR:

    CVaR_alpha = E[Loss | Loss > VaR_alpha]

This module implements the Gaussian-parametric special case, not the fully
empirical historical tail mean. Rockafellar & Uryasev (2000) show the
general CVaR can be computed via the convex optimization

    CVaR_alpha = min_z { z + (1 / (1 - alpha)) * E[max(Loss - z, 0)] }

for a sample of historical or simulated losses, but for a single return
series under a normality assumption the closed form becomes

    CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)

with z_alpha = Phi^-1(1 - alpha). This is the quantity implemented here.

References:
    J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document* (4th
    ed.). (VaR methodology.)

    Rockafellar, R.T. and Uryasev, S. (2000). "Optimization of Conditional
    Value-at-Risk." *Journal of Risk*, 3, 21-41. (CVaR / Expected
    Shortfall.)

    See REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def _validate_returns_and_confidence(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    volatility: float | None = None,
) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if returns.size < 2:
        raise ValueError("returns must contain at least two observations")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if volatility is not None and volatility < 0.0:
        raise ValueError("volatility must be non-negative")


def value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
) -> float:
    """Compute parametric (delta-normal) Value at Risk.

        VaR_alpha = -(mu + z_alpha * sigma),  z_alpha = Phi^-1(1 - alpha)

    Args:
        returns: Historical or simulated portfolio returns.
        confidence_level: Confidence level alpha, e.g. 0.95 or 0.99.
        mean: Optional override for mu. If None (default), mu is estimated
            from `returns` as `np.mean(returns)`. If provided, this value is
            used directly and the contents of `returns` are ignored for the
            purpose of computing mu (only its validity, e.g. length, still
            matters).
        volatility: Optional override for sigma. If None (default), sigma is
            estimated from `returns` as `np.std(returns, ddof=1)`. If
            provided, this value is used directly and the contents of
            `returns` are ignored for the purpose of computing sigma. Must
            be non-negative.

    Returns:
        VaR as a positive loss figure, in the same units as `returns`.

    References:
        J.P. Morgan/Reuters (1996), RiskMetrics Technical Document.
        See REFERENCES.md.
    """
    _validate_returns_and_confidence(returns, confidence_level, volatility)
    return _value_at_risk(returns, confidence_level, mean, volatility)


def _value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
) -> float:
    mu = float(np.mean(returns)) if mean is None else mean
    sigma = float(np.std(returns, ddof=1)) if volatility is None else volatility
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    return float(-(mu + z_alpha * sigma))


def conditional_value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

        CVaR_alpha = E[Loss | Loss > VaR_alpha]

    Args:
        returns: Historical or simulated portfolio returns.
        confidence_level: Confidence level alpha, e.g. 0.95 or 0.99.
        mean: Optional override for mu. If None (default), mu is estimated
            from `returns` as `np.mean(returns)`. If provided, this value is
            used directly and the contents of `returns` are ignored for the
            purpose of computing mu (only its validity, e.g. length, still
            matters).
        volatility: Optional override for sigma. If None (default), sigma is
            estimated from `returns` as `np.std(returns, ddof=1)`. If
            provided, this value is used directly and the contents of
            `returns` are ignored for the purpose of computing sigma. Must
            be non-negative.

    Returns:
        CVaR as a positive loss figure, in the same units as `returns`.

    References:
        Rockafellar & Uryasev (2000), "Optimization of Conditional
        Value-at-Risk." See REFERENCES.md.
    """
    _validate_returns_and_confidence(returns, confidence_level, volatility)
    return _conditional_value_at_risk(returns, confidence_level, mean, volatility)


def _conditional_value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
) -> float:
    # This module implements the parametric (delta-normal) methodology
    # throughout: `value_at_risk` assumes R ~ N(mu, sigma^2) rather than
    # using the empirical/historical quantile. CVaR is computed under the
    # same distributional assumption, using the closed-form Gaussian
    # expected-shortfall formula (the truncated-normal tail mean), which is
    # the E[Loss | Loss > VaR_alpha] the module docstring defines, evaluated
    # under R ~ N(mu, sigma^2):
    #
    #     CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
    #
    # where phi is the standard normal density and z_alpha = Phi^-1(1-alpha),
    # the same z_alpha used by `_value_at_risk`.
    mu = float(np.mean(returns)) if mean is None else mean
    sigma = float(np.std(returns, ddof=1)) if volatility is None else volatility
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    return float(-mu + sigma * norm.pdf(z_alpha) / (1.0 - confidence_level))
