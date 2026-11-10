"""Sampling inference on the Sharpe ratio: standard error, PSR, minimum track record, DSR.

All Sharpe ratios in the formulas are per-period, `SR = mean(x) / std(x, ddof=1)`, with
`gamma3` the population (divide-by-n) skewness and `gamma4` the population non-excess kurtosis:

    Var[SR] = (1 - gamma3 * SR + (gamma4 - 1) / 4 * SR^2) / (n - 1)
    PSR(SR*) = Phi((SR - SR*) / sqrt(Var[SR]))
    MinTRL = 1 + (1 - gamma3 * SR + (gamma4 - 1) / 4 * SR^2) * (z_{1-alpha} / (SR - SR*))^2
    SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))
    DSR = PSR(SR0)

with `g` the Euler-Mascheroni constant, `N` the number of independent trials and `V` the
variance of the trials' per-period Sharpe ratios.

A result that is undefined raises `quantcore.UndefinedResultError`, a `ValueError`: returns with
zero range (`np.ptp == 0`, so zero variance), a variance factor that is not positive, `SR` equal
to `SR*` in the minimum track record length, or a value that is not finite. Invalid input raises
a plain `ValueError`.

References:
    Mertens, E. (2002), "Comments on variance of the IID estimator in Lo (2002)." Opdyke, J.D.
    (2007), "Comparing Sharpe ratios: so where are the p-values?" Journal of Asset Management,
    8(5), 308-336. Bailey, D.H. and Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient
    Frontier." Journal of Risk, 15(2), 3-44. Bailey, D.H. and Lopez de Prado, M. (2014), "The
    Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
    Non-Normality." Journal of Portfolio Management, 40(5), 94-107. See docs/REFERENCES.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from quantcore.errors import UndefinedResultError, _finite_result
from quantcore.performance.metrics import _sharpe_ratio

_MIN_OBS = 10
_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class SharpeSE:
    """Annualised Sharpe ratio and its standard error.

    Attributes:
        sharpe: Per-period Sharpe ratio times `sqrt(periods_per_year)`.
        se: Per-period standard error times `sqrt(periods_per_year)`.
        n: Number of observations.
    """

    sharpe: float
    se: float
    n: int


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe ratio and the quantities that produced it.

    Attributes:
        dsr: `PSR(sr_zero)`, the probability that the best trial's true per-period Sharpe ratio
            exceeds `sr_zero`.
        sr_zero: The per-period hurdle `SR0` (or `sr_star_override` when given).
        n_effective: The number of independent trials used for `sr_zero`.
        var_sharpe: The variance of the trials' per-period Sharpe ratios that scaled `sr_zero`.
        psr_naive: `PSR(0)`, the probability against a zero hurdle.
    """

    dsr: float
    sr_zero: float
    n_effective: float
    var_sharpe: float
    psr_naive: float


def _sharpe_and_variance_factor(returns: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return `(SR, 1 - gamma3 SR + (gamma4 - 1) / 4 SR^2)` after validating `returns`.

    Raises:
        ValueError: `returns` is not a finite 1-D array of at least 10 observations.
        UndefinedResultError: `returns` has zero range (`np.ptp == 0`) or a computed standard
            deviation of 0, a power of the standard deviation overflows or underflows float64,
            or the factor is not positive.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 1 or x.size < _MIN_OBS:
        raise ValueError(f"returns must be 1-D with at least {_MIN_OBS} observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    d = x - x.mean()
    sigma = float(d.std(ddof=0))
    if np.ptp(x) == 0.0 or sigma == 0.0 or float(x.std(ddof=1)) == 0.0:
        raise UndefinedResultError("returns have zero range; the Sharpe ratio is undefined")
    sr = _sharpe_ratio(x, 0.0, 1)
    try:
        gamma3 = float(np.mean(d**3)) / sigma**3
        gamma4 = float(np.mean(d**4)) / sigma**4
    except (OverflowError, ZeroDivisionError) as exc:
        raise UndefinedResultError(
            "the third or fourth power of the returns' standard deviation overflows or "
            "underflows float64; the Sharpe variance factor is undefined"
        ) from exc
    factor = 1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr**2
    if not (math.isfinite(factor) and factor > 0.0):
        raise UndefinedResultError("the Sharpe variance factor is not positive")
    return sr, factor


def sharpe_se_mertens(returns: npt.NDArray[np.float64], *, periods_per_year: int) -> SharpeSE:
    """Sharpe ratio and its standard error for non-normal, serially independent returns.

    The per-period variance is `(1 - gamma3 SR + (gamma4 - 1) / 4 SR^2) / (n - 1)`. Both the
    Sharpe ratio and the standard error are multiplied by `sqrt(periods_per_year)`.

    Args:
        returns: Per-period simple returns, at least 10 observations.
        periods_per_year: Periods per year, at least 1.

    Returns:
        A `SharpeSE` of the annualised Sharpe ratio, its standard error and the sample size.

    Raises:
        ValueError: `periods_per_year < 1`, or `returns` is not a finite 1-D array of at least 10
            observations.
        UndefinedResultError: `returns` has zero range (zero variance), or the variance factor
            is not positive.
    """
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    sr, factor = _sharpe_and_variance_factor(returns)
    n = len(returns)
    scale = math.sqrt(periods_per_year)
    return SharpeSE(sharpe=sr * scale, se=math.sqrt(factor / (n - 1)) * scale, n=n)


def probabilistic_sharpe_ratio(returns: npt.NDArray[np.float64], sr_star: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds `sr_star` (PSR).

    Args:
        returns: Per-period simple returns, at least 10 observations.
        sr_star: Per-period Sharpe ratio threshold.

    Returns:
        `Phi((SR - sr_star) sqrt(n - 1) / sqrt(factor))`, in `[0, 1]`; `1 - PSR` is the p-value of
        `H0: SR <= sr_star`.

    Raises:
        ValueError: `returns` is not a finite 1-D array of at least 10 observations, or
            `sr_star` is not finite.
        UndefinedResultError: `returns` has zero range (zero variance), or the variance factor
            is not positive.
    """
    if not math.isfinite(sr_star):
        raise ValueError("sr_star must be finite")
    sr, factor = _sharpe_and_variance_factor(returns)
    z = (sr - sr_star) * math.sqrt(len(returns) - 1) / math.sqrt(factor)
    return _finite_result(float(norm.cdf(z)), "probabilistic Sharpe ratio")


def minimum_track_record_length(
    returns: npt.NDArray[np.float64], sr_star: float, alpha: float
) -> float:
    """Observations needed for PSR to reach `1 - alpha`, given the observed Sharpe and moments.

    Args:
        returns: Per-period simple returns, at least 10 observations; only their Sharpe ratio and
            moments are used.
        sr_star: Per-period Sharpe ratio threshold.
        alpha: One-sided significance level, in `(0, 1)`.

    Returns:
        `1 + factor (z_{1-alpha} / (SR - sr_star))^2`, in periods.

    Raises:
        ValueError: `alpha` outside `(0, 1)`, or any input condition of
            `probabilistic_sharpe_ratio`.
        UndefinedResultError: `SR` equal to `sr_star` (relative 1e-9, absolute 1e-12), any
            undefined case of `probabilistic_sharpe_ratio`, or a result that overflows float64.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not math.isfinite(sr_star):
        raise ValueError("sr_star must be finite")
    sr, factor = _sharpe_and_variance_factor(returns)
    if math.isclose(sr, sr_star, rel_tol=1e-9, abs_tol=1e-12):
        raise UndefinedResultError(
            "the Sharpe ratio equals sr_star; the track record length is undefined"
        )
    return _finite_result(
        1.0 + factor * (float(norm.ppf(1.0 - alpha)) / (sr - sr_star)) ** 2,
        "minimum track record length",
    )


def expected_max_sharpe(
    n_trials: float, var_sharpe: float, *, euler_gamma: float = _EULER_GAMMA
) -> float:
    """Expected maximum Sharpe ratio of `n_trials` skill-less strategies, `SR0`.

    Implements the asymptotic formula of Bailey and Lopez de Prado (2014), eq. for `SR0`,
    `sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`, with `g` the Euler-Mascheroni
    constant, as published. It approximates the exact expectation of the maximum of `N` standard
    Normal draws. Measured against numerical integration of `N x phi(x) Phi(x)^(N-1)`, the
    approximation is 7.9% low at `N=2`, 2.5% high at `N=5`, 2.3% high at `N=10`, 0.9% high at
    `N=100` and 0.4% high at `N=1000`.

    Args:
        n_trials: Number of independent trials `N`, finite and at least 2; need not be an
            integer.
        var_sharpe: Variance of the trials' per-period Sharpe ratios, finite and non-negative.
        euler_gamma: The Euler-Mascheroni constant.

    Returns:
        `SR0`, in the units of `sqrt(var_sharpe)`.

    Raises:
        ValueError: `n_trials < 2` or not finite, or `var_sharpe` negative or not finite.
        UndefinedResultError: `n_trials` is so large (above about 1e16) that `1 - 1/N` rounds
            to 1 in float64 and `Phi^-1(1 - 1/N)` is infinite.
    """
    if not (math.isfinite(n_trials) and n_trials >= 2.0):
        raise ValueError("n_trials must be at least 2")
    if not (math.isfinite(var_sharpe) and var_sharpe >= 0.0):
        raise ValueError("var_sharpe must be a finite, non-negative value")
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    if not (math.isfinite(z1) and math.isfinite(z2)):
        raise UndefinedResultError(
            "n_trials is so large that 1 - 1/n_trials rounds to 1; the expected maximum Sharpe "
            "ratio is not finite"
        )
    return _finite_result(
        math.sqrt(var_sharpe) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2),
        "expected maximum Sharpe ratio",
    )


def deflated_sharpe_ratio(
    returns_best: npt.NDArray[np.float64],
    family_sharpes: Sequence[float] | npt.NDArray[np.float64],
    *,
    n_effective: float,
    sr_star_override: float | None = None,
    trial_variance: Literal["family", "no_skill"] = "family",
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio of the best trial in a family of trials.

    The hurdle is `SR0 = expected_max_sharpe(n_effective, V)`, or 0 when `n_effective <= 1`, or
    `sr_star_override` when given; `DSR = PSR(SR0)`.

    Args:
        returns_best: Per-period returns of the best trial, at least 10 observations.
        family_sharpes: Per-period Sharpe ratios of every trial, including the best; used only
            for `trial_variance="family"`.
        n_effective: Number of independent trials, finite.
        sr_star_override: If given, used as `SR0` directly.
        trial_variance: `"family"` uses the sample variance (`ddof=1`, 0 for one trial) of
            `family_sharpes`; `"no_skill"` uses `1 / (T - 1)`, `T = len(returns_best)`.

    Returns:
        A `DeflatedSharpeResult`; `dsr <= psr_naive` whenever `sr_zero >= 0`.

    Raises:
        ValueError: unknown `trial_variance`, empty or non-finite `family_sharpes`, non-finite
            `n_effective` or `sr_star_override`, `returns_best` shorter than 2 (`"no_skill"`),
            or any input condition of `probabilistic_sharpe_ratio` and `expected_max_sharpe`.
        UndefinedResultError: any undefined case of `probabilistic_sharpe_ratio` (zero-range
            `returns_best`, a non-positive variance factor) or of `expected_max_sharpe`.
    """
    if trial_variance not in ("family", "no_skill"):
        raise ValueError(f"trial_variance must be 'family' or 'no_skill', got {trial_variance!r}")
    values = np.asarray(family_sharpes, dtype=np.float64)
    if values.size < 1:
        raise ValueError("family_sharpes must contain at least 1 value")
    if not np.all(np.isfinite(values)):
        raise ValueError("family_sharpes must be finite")
    if not math.isfinite(n_effective):
        raise ValueError("n_effective must be finite")
    if trial_variance == "no_skill":
        if len(returns_best) < 2:
            raise ValueError("returns_best needs at least 2 observations for the no-skill variance")
        var_sharpe = 1.0 / (len(returns_best) - 1)
    else:
        var_sharpe = float(values.var(ddof=1)) if values.size >= 2 else 0.0

    if sr_star_override is not None:
        sr_zero = sr_star_override
    elif n_effective <= 1.0:
        sr_zero = 0.0
    else:
        sr_zero = expected_max_sharpe(n_effective, var_sharpe)
    return DeflatedSharpeResult(
        dsr=probabilistic_sharpe_ratio(returns_best, sr_zero),
        sr_zero=sr_zero,
        n_effective=n_effective,
        var_sharpe=var_sharpe,
        psr_naive=probabilistic_sharpe_ratio(returns_best, 0.0),
    )
