"""Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).

VaR at confidence level alpha is the loss threshold not exceeded with
probability alpha:

    P(Loss > VaR_alpha) = 1 - alpha

``value_at_risk`` implements the parametric (delta-normal) VaR from the
RiskMetrics methodology, which assumes portfolio returns R ~ N(mu, sigma^2):

    VaR_alpha = -(mu + z_alpha * sigma)

where z_alpha = Phi^-1(1 - alpha) is the standard normal quantile.

`performance/metrics.py`'s `component_var` uses z_alpha = Phi^-1(alpha)
instead -- the opposite sign, since Phi^-1(alpha) = -Phi^-1(1-alpha) for
the standard normal. This is not an inconsistency: `component_var` has no
mean term and multiplies z_alpha directly rather than subtracting it, so
the two conventions' opposite signs cancel exactly. Concretely,
`component_var(weights, cov_matrix, alpha).sum()` equals
`value_at_risk(returns, alpha, mean=0.0, volatility=sqrt(weights @
cov_matrix @ weights))` to floating-point precision for every confidence
level -- see `tests/unit/risk/test_var.py`'s
`test_component_var_sums_to_portfolio_var_across_confidence_levels`, which
pins this down so it can't be "fixed" into actually breaking by a future
reader who (reasonably, given the differing z_alpha sign) suspects a bug
without checking the algebra.

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

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

VarMethod = Literal["gaussian", "historical", "cornish_fisher"]


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


def _validate_method(method: VarMethod, mean: float | None, volatility: float | None) -> None:
    if method not in ("gaussian", "historical", "cornish_fisher"):
        raise ValueError('method must be "gaussian", "historical", or "cornish_fisher"')
    if method == "historical" and (mean is not None or volatility is not None):
        raise ValueError(
            'method="historical" ignores mean/volatility overrides (it uses the '
            "empirical distribution of returns directly) -- pass neither, or use "
            'method="gaussian"/"cornish_fisher" if you need to override mu/sigma'
        )


def value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
    method: VarMethod = "gaussian",
) -> float:
    """Compute Value at Risk.

    Args:
        returns: Historical or simulated portfolio returns.
        confidence_level: Confidence level alpha, e.g. 0.95 or 0.99.
        mean: Optional override for mu (Gaussian/Cornish-Fisher methods
            only). If None (default), mu is estimated from `returns` as
            `np.mean(returns)`. If provided, this value is used directly
            and the contents of `returns` are ignored for the purpose of
            computing mu (only its validity, e.g. length, still matters).
        volatility: Optional override for sigma (Gaussian/Cornish-Fisher
            methods only). If None (default), sigma is estimated from
            `returns` as `np.std(returns, ddof=1)`. If provided, this
            value is used directly and the contents of `returns` are
            ignored for the purpose of computing sigma. Must be
            non-negative.
        method: `"gaussian"` (default; the original, only behavior) —
            parametric delta-normal VaR from the RiskMetrics methodology,
            assuming R ~ N(mu, sigma^2):
            `VaR_alpha = -(mu + z_alpha*sigma)`, `z_alpha = Phi^-1(1-alpha)`.
            `"historical"` — the empirical (1-alpha)-quantile of `returns`
            directly, no distributional assumption; `mean`/`volatility`
            must not be passed with this method. `"cornish_fisher"` —
            Gaussian VaR with `z_alpha` replaced by the Cornish-Fisher
            (1938) expansion's skewness/kurtosis-adjusted quantile
            `z_cf`, which corrects the Gaussian quantile for the sample's
            own skewness and excess kurtosis (Zangari 1996; Favre &
            Galeano 2002) — reduces exactly to `"gaussian"` when skewness
            and excess kurtosis are both 0.

    Returns:
        VaR as a positive loss figure, in the same units as `returns`.

    References:
        J.P. Morgan/Reuters (1996), RiskMetrics Technical Document
        ("gaussian"). Cornish, E.A. and Fisher, R.A. (1938), "Moments and
        Cumulants in the Specification of Distributions." Zangari, P.
        (1996), "An Improved Methodology for Measuring VaR." Favre, L. and
        Galeano, J.-A. (2002), "Mean-Modified Value-at-Risk Optimization
        with Hedge Funds." ("cornish_fisher"). See REFERENCES.md.
    """
    _validate_returns_and_confidence(returns, confidence_level, volatility)
    _validate_method(method, mean, volatility)
    return _value_at_risk(returns, confidence_level, mean, volatility, method)


def _cornish_fisher_z(returns: npt.NDArray[np.float64], z: float) -> float:
    """Cornish-Fisher (1938) skewness/excess-kurtosis-adjusted normal quantile.

    z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36

    where S is the (population-moment, i.e. `scipy.stats.skew(bias=True)`
    convention) sample skewness and K is the excess kurtosis
    (`scipy.stats.kurtosis(bias=True)`'s convention, kurtosis - 3). Reduces
    to z exactly when S=K=0.
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


def _value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None,
    volatility: float | None,
    method: VarMethod,
) -> float:
    if method == "historical":
        return float(-np.quantile(returns, 1.0 - confidence_level))

    mu = float(np.mean(returns)) if mean is None else mean
    sigma = float(np.std(returns, ddof=1)) if volatility is None else volatility
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    if method == "cornish_fisher":
        z_alpha = _cornish_fisher_z(returns, z_alpha)
    return float(-(mu + z_alpha * sigma))


def conditional_value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None = None,
    volatility: float | None = None,
    method: VarMethod = "gaussian",
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

        CVaR_alpha = E[Loss | Loss > VaR_alpha]

    Args:
        returns: Historical or simulated portfolio returns.
        confidence_level: Confidence level alpha, e.g. 0.95 or 0.99.
        mean: Optional override for mu (Gaussian/Cornish-Fisher methods
            only) — see `value_at_risk`.
        volatility: Optional override for sigma (Gaussian/Cornish-Fisher
            methods only) — see `value_at_risk`.
        method: `"gaussian"` (default; the original, only behavior) — the
            closed-form Gaussian expected-shortfall (truncated-normal tail
            mean): `CVaR_alpha = -mu + sigma*phi(z_alpha)/(1-alpha)`.
            `"historical"` — the empirical tail mean: the average of the
            observations at or below the historical-method VaR threshold,
            no distributional assumption. `"cornish_fisher"` — the same
            Gaussian expected-shortfall formula with `z_alpha` replaced by
            the Cornish-Fisher-adjusted quantile (see `value_at_risk`).
            This substitution is a widely used practical approximation,
            not an exact expected shortfall under the Cornish-Fisher
            expansion (the exact integral has a more involved closed form
            — see Boudt, Peterson & Croux (2008) — not implemented here).

    Returns:
        CVaR as a positive loss figure, in the same units as `returns`.

    References:
        Rockafellar & Uryasev (2000), "Optimization of Conditional
        Value-at-Risk." See REFERENCES.md. Cornish-Fisher references as in
        `value_at_risk`.
    """
    _validate_returns_and_confidence(returns, confidence_level, volatility)
    _validate_method(method, mean, volatility)
    return _conditional_value_at_risk(returns, confidence_level, mean, volatility, method)


def _conditional_value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    mean: float | None,
    volatility: float | None,
    method: VarMethod,
) -> float:
    if method == "historical":
        threshold = float(np.quantile(returns, 1.0 - confidence_level))
        tail = returns[returns <= threshold]
        if tail.size == 0:
            tail = np.array([threshold])
        return float(-np.mean(tail))

    # Gaussian and Cornish-Fisher: same closed-form expected-shortfall
    # structure (the truncated-normal tail mean), evaluated at the
    # method's own z_alpha -- see the module docstring for the Gaussian
    # derivation and `_cornish_fisher_z` for the adjustment.
    mu = float(np.mean(returns)) if mean is None else mean
    sigma = float(np.std(returns, ddof=1)) if volatility is None else volatility
    z_alpha = float(norm.ppf(1.0 - confidence_level))
    if method == "cornish_fisher":
        z_alpha = _cornish_fisher_z(returns, z_alpha)
    return float(-mu + sigma * norm.pdf(z_alpha) / (1.0 - confidence_level))
