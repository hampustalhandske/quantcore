"""Stationary-bootstrap confidence intervals and Hansen's test of superior predictive ability.

`optimal_block_length` is the Politis-White (2004) automatic expected block length for the
stationary bootstrap, as corrected by Patton, Politis and White (2009). `stationary_bootstrap_ci`
resamples a series with the stationary bootstrap of Politis and Romano (1994), evaluates a
statistic on every resample and returns a percentile interval or Efron's (1987) bias-corrected
and accelerated (BCa) interval. `spa_test` is the bootstrap test of superior predictive ability
of Hansen (2005), which extends White's (2000) Reality Check.

The block length, the resampling and the SPA test are computed by `arch.bootstrap`, which is
imported on first use. Every random draw comes from the caller's `numpy.random.Generator`, so the
same generator state gives the same result.

References:
    Politis, D.N. and Romano, J.P. (1994), "The Stationary Bootstrap." Journal of the American
    Statistical Association, 89(428), 1303-1313. Politis, D.N. and White, H. (2004), "Automatic
    Block-Length Selection for the Dependent Bootstrap." Econometric Reviews, 23(1), 53-70, with
    Patton, A., Politis, D.N. and White, H. (2009), "Correction to ...", Econometric Reviews,
    28(4), 372-375. Efron, B. (1987), "Better Bootstrap Confidence Intervals." Journal of the
    American Statistical Association, 82(397), 171-185. Hansen, P.R. (2005), "A Test for Superior
    Predictive Ability." Journal of Business & Economic Statistics, 23(4), 365-380. White, H.
    (2000), "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126. See
    docs/REFERENCES.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from quantcore.errors import UndefinedResultError, _finite_result

_PROPORTION_CLIP = 1e-10

# Shortest series `arch`'s block-length rule accepts: it computes autocorrelations up to lag
# ceil(sqrt(n)) + max(5, floor(log10(n))), which must not exceed n.
_MIN_BLOCK_LENGTH_OBS = 8


def _as_series(x: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    """`x` as a new C-contiguous float64 array; ValueError unless 1-D, 2+ points and finite."""
    array = np.array(x, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got {array.ndim} dimensions")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 observations")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def optimal_block_length(x: npt.NDArray[np.float64]) -> float:
    """Politis-White (2004) automatic expected block length for the stationary bootstrap.

    This is the `"stationary"` column of `arch.bootstrap.optimal_block_length`, which includes the
    Patton, Politis and White (2009) correction. The value is continuous: the mean of the
    geometric block-length distribution, not an integer.

    Args:
        x: 1-D finite series, at least 8 observations.

    Returns:
        The expected block length.

    Raises:
        ValueError: `x` is not 1-D, has fewer than 8 observations or is not finite.
        UndefinedResultError: `x` has zero range (`np.ptp(x) == 0`), so its autocorrelations
            are undefined; or `arch` returns a value that is not finite.
    """
    from arch.bootstrap import optimal_block_length as _arch_optimal_block_length

    array = _as_series(x, "x")
    if array.shape[0] < _MIN_BLOCK_LENGTH_OBS:
        raise ValueError(f"x must have at least {_MIN_BLOCK_LENGTH_OBS} observations")
    if np.ptp(array) == 0.0:
        raise UndefinedResultError(
            "x has zero range, so its autocorrelations are undefined; "
            "the optimal block length is undefined"
        )
    table = _arch_optimal_block_length(array)
    return _finite_result(float(table["stationary"].iloc[0]), "optimal block length")


@dataclass(frozen=True)
class BootstrapInterval:
    """Result of `stationary_bootstrap_ci`.

    Attributes:
        estimate: `stat` on the full series.
        ci_low: Lower end of the interval.
        ci_high: Upper end of the interval.
        ci_level: Confidence level of the interval.
        block_length: Expected block length the resampling used, before it is rounded to the
            integer `arch` takes.
        n_boot: Number of resamples drawn.
        n_dropped: Number of resamples left out because `stat` raised `ValueError` or returned a
            non-finite value on them.
    """

    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    block_length: float
    n_boot: int
    n_dropped: int


def _percentile_ci(replicates: npt.NDArray[np.float64], ci: float) -> tuple[float, float]:
    """Linear-interpolation percentiles of `replicates` at `(1 - ci) / 2` and `(1 + ci) / 2`."""
    alpha = 1.0 - ci
    lo = float(np.percentile(replicates, 100.0 * alpha / 2.0))
    hi = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _bca_ci(
    values: npt.NDArray[np.float64],
    point_estimate: float,
    replicates: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    ci: float,
) -> tuple[float, float]:
    """Efron's (1987) bias-corrected and accelerated interval from bootstrap replicates.

    The bias correction is `z0 = Phi^-1(p)`, where `p` is the share of replicates strictly below
    `point_estimate`, clipped to `[1e-10, 1 - 1e-10]`. The acceleration is
    `a = sum(d^3) / (6 * (sum(d^2))^1.5)` with `d = mean(theta_(.)) - theta_(i)` over the
    leave-one-out jackknife values `theta_(i)` of `stat`, and 0 when the denominator is 0. The
    endpoints are `np.percentile` of the replicates at `Phi(z0 + (z0 + z) / (1 - a * (z0 + z)))`
    for `z = Phi^-1((1 - ci) / 2)` and `Phi^-1((1 + ci) / 2)`, each level clipped to `[0, 1]`.
    An exception `stat` raises on a jackknife sample propagates.
    """
    alpha = 1.0 - ci
    proportion_below = float(np.mean(replicates < point_estimate))
    proportion_below = min(max(proportion_below, _PROPORTION_CLIP), 1.0 - _PROPORTION_CLIP)
    z0 = float(norm.ppf(np.array([proportion_below]))[0])

    n = values.shape[0]
    jackknife = np.empty(n, dtype=np.float64)
    for i in range(n):
        jackknife[i] = stat(np.delete(values, i))
    jack_mean = float(jackknife.mean())
    numerator = float(np.sum((jack_mean - jackknife) ** 3))
    denominator = 6.0 * float(np.sum((jack_mean - jackknife) ** 2)) ** 1.5
    acceleration = numerator / denominator if denominator != 0.0 else 0.0

    z_lo = float(norm.ppf(np.array([alpha / 2.0]))[0])
    z_hi = float(norm.ppf(np.array([1.0 - alpha / 2.0]))[0])
    p_lo = float(norm.cdf(np.array([z0 + (z0 + z_lo) / (1.0 - acceleration * (z0 + z_lo))]))[0])
    p_hi = float(norm.cdf(np.array([z0 + (z0 + z_hi) / (1.0 - acceleration * (z0 + z_hi))]))[0])
    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)
    lo = float(np.percentile(replicates, 100.0 * p_lo))
    hi = float(np.percentile(replicates, 100.0 * p_hi))
    return lo, hi


def _stat_or_nan(
    stat: Callable[[npt.NDArray[np.float64]], float], x: npt.NDArray[np.float64]
) -> float:
    """`stat(x)` as a float; NaN when `stat` raises `ValueError` or returns a non-finite value."""
    try:
        value = float(stat(x))
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def stationary_bootstrap_ci(
    x: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
    ci: float = 0.95,
    method: Literal["percentile", "bca"] = "percentile",
    block_length: float | None = None,
) -> BootstrapInterval:
    """Stationary-bootstrap confidence interval for an arbitrary statistic of a series.

    `n_boot` resamples are drawn by `arch.bootstrap.StationaryBootstrap` with integer block size
    `max(1, round(block_length))`, seeded with `rng`, in the order `.bootstrap(n_boot)` yields
    them. `stat` is evaluated on each; a resample on which it raises `ValueError` or returns a
    non-finite value is dropped and counted in `n_dropped`. The interval is built from the
    remaining replicates:

    - `"percentile"`: the `(1 - ci) / 2` and `(1 + ci) / 2` percentiles (linear interpolation).
    - `"bca"`: Efron's (1987) bias-corrected and accelerated interval. The bias correction `z0` is
      the normal quantile of the share of replicates below the estimate, that share clipped to
      `[1e-10, 1 - 1e-10]`; the acceleration comes from the leave-one-out jackknife of `stat` on
      `x` (`n` extra evaluations, whose exceptions propagate) and is 0 when the jackknife values
      are all equal.

    `ci_low` and `ci_high` are the smaller and larger of the two percentiles.

    Args:
        x: 1-D finite series, at least 2 observations.
        stat: Statistic of a 1-D float64 array, e.g. `lambda v: float(v.mean())`.
        rng: Generator the resampling draws from; it is advanced by the draws.
        n_boot: Number of resamples.
        ci: Confidence level, strictly between 0 and 1.
        method: `"percentile"` or `"bca"`.
        block_length: Expected block length, positive and finite. None uses
            `optimal_block_length(x)`, which needs at least 8 observations and a non-constant
            `x`.

    Returns:
        `BootstrapInterval` with the estimate `stat(x)`, the interval, the level, the block length
        used, `n_boot` and the number of dropped resamples.

    Raises:
        ValueError: `x` is not 1-D, has fewer than 2 observations (fewer than 8 when
            `block_length` is None) or is not finite; `n_boot` is not positive; `ci` is not in
            `(0, 1)`; `method` is neither `"percentile"` nor `"bca"`; `block_length` is not
            positive and finite; or every resample is dropped.
        UndefinedResultError: `block_length` is None and `optimal_block_length(x)` is
            undefined (a constant `x`).
    """
    from arch.bootstrap import StationaryBootstrap

    values = _as_series(x, "x")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie strictly between 0 and 1")
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    if block_length is None:
        used_block_length = optimal_block_length(values)
    else:
        used_block_length = float(block_length)
        if not (math.isfinite(used_block_length) and used_block_length > 0.0):
            raise ValueError("block_length must be positive and finite")

    point_estimate = float(stat(values))

    bootstrap = StationaryBootstrap(max(1, round(used_block_length)), values, seed=rng)
    raw_replicates = np.empty(n_boot, dtype=np.float64)
    for i, (positional, _keyword) in enumerate(bootstrap.bootstrap(n_boot)):
        (resample,) = positional
        raw_replicates[i] = _stat_or_nan(stat, np.asarray(resample, dtype=np.float64))
    finite = np.isfinite(raw_replicates)
    n_dropped = int(n_boot - int(np.sum(finite)))
    if n_dropped == n_boot:
        raise ValueError("stat raised ValueError or was non-finite on every resample")
    replicates = raw_replicates[finite]

    if method == "percentile":
        lo, hi = _percentile_ci(replicates, ci)
    else:
        lo, hi = _bca_ci(values, point_estimate, replicates, stat, ci)

    return BootstrapInterval(
        estimate=point_estimate,
        ci_low=min(lo, hi),
        ci_high=max(lo, hi),
        ci_level=ci,
        block_length=used_block_length,
        n_boot=n_boot,
        n_dropped=n_dropped,
    )


@dataclass(frozen=True)
class SpaPValues:
    """Result of `spa_test`: the p-values of the three re-centring conventions.

    Attributes:
        lower: p-value whose bootstrap distribution re-centres only the models with a positive
            mean loss differential (benchmark loss minus model loss); the smallest of the three.
        consistent: Hansen's (2005) consistent p-value, which re-centres every model except
            those whose mean loss differential is below `-sqrt(2 * var / T * ln ln T)`.
        upper: p-value that re-centres every model (the least favourable configuration of
            White's Reality Check); the largest of the three.
    """

    lower: float
    consistent: float
    upper: float


def spa_test(
    benchmark_losses: npt.NDArray[np.float64],
    model_losses: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
    block_size: int,
    reps: int,
    studentize: bool,
) -> SpaPValues:
    """Hansen's (2005) test of superior predictive ability (SPA) of `k` models over a benchmark.

    The null is that no model has a lower expected loss than the benchmark; a small p-value means
    at least one model is better. Losses, not returns, go in: for returns, pass their negatives.
    Computed by `arch.bootstrap.SPA` with a stationary bootstrap of expected block size
    `block_size`, `nested=False`, seeded with `rng`, and every other option given explicitly.

    In arch 8.0.0 the test statistic and its bootstrap distribution are the largest mean loss
    differential and are not divided by its standard deviation, whatever `studentize` is: the
    p-values are the same for `studentize=True` and `False`. The variances of the loss
    differentials enter only through the threshold that decides which models are re-centred for
    the consistent p-value.

    Args:
        benchmark_losses: `T` finite losses of the benchmark, `T >= 2`.
        model_losses: `T x k` finite losses, one column per candidate model.
        rng: Generator the bootstrap draws from; it is advanced by the draws.
        block_size: Expected block size of the stationary bootstrap, positive.
        reps: Number of bootstrap replications, positive.
        studentize: Passed to `arch.bootstrap.SPA` as `studentize`.

    Returns:
        `SpaPValues` with the lower, consistent and upper p-values.

    Raises:
        ValueError: `block_size` or `reps` is not positive; `benchmark_losses` is not 1-D with
            at least 2 observations; `model_losses` is not 2-D with at least 1 column; the two
            differ in length; or either is not finite.
        UndefinedResultError: A column of the loss differential `benchmark_losses -
            model_losses[:, j]` has zero range (`np.ptp == 0`): a model identical to the
            benchmark, or differing from it by a constant. Its differential has no variance,
            so the re-centring threshold and the bootstrap distribution are degenerate. The
            message names the column(s); `rng` is not advanced.
    """
    from arch.bootstrap import SPA

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if reps <= 0:
        raise ValueError("reps must be positive")
    benchmark = _as_series(benchmark_losses, "benchmark_losses")
    models = np.array(model_losses, dtype=np.float64, order="C", copy=True)
    if models.ndim != 2:
        raise ValueError(f"model_losses must be a 2-D array (T x k), got {models.ndim} dimensions")
    if models.shape[0] != benchmark.shape[0]:
        raise ValueError("benchmark_losses and model_losses must have the same number of rows")
    if models.shape[1] < 1:
        raise ValueError("model_losses must have at least 1 column")
    if not np.isfinite(models).all():
        raise ValueError("model_losses must be finite")
    differential = benchmark[:, np.newaxis] - models
    constant = np.flatnonzero(np.ptp(differential, axis=0) == 0.0)
    if constant.size > 0:
        raise UndefinedResultError(
            f"the loss differential of model column(s) {constant.tolist()} against the "
            "benchmark has zero range; the SPA p-values are undefined"
        )

    spa = SPA(
        benchmark,
        models,
        block_size=block_size,
        reps=reps,
        bootstrap="stationary",
        studentize=studentize,
        nested=False,
        seed=rng,
    )
    spa.compute()
    pvalues = spa.pvalues
    return SpaPValues(
        lower=float(pvalues["lower"]),
        consistent=float(pvalues["consistent"]),
        upper=float(pvalues["upper"]),
    )
