"""Oracle tests for `stationary_bootstrap_ci`: normal theory and the BCa reduction.

The resampling is `arch`'s, so no hand stationary bootstrap can reproduce its draws. These tests
check the interval against results that do not depend on the draws. The comparison of both
intervals with a hand Efron (1987) computation on the same `arch` replicates is the conformance
target `stationary_bootstrap_ci` (`conformance/targets/statistics_extra.py`), which runs it on
generated inputs of every return family.

Normal theory: for iid data and block length 1 the resampling is the iid bootstrap, whose
distribution of the mean has standard error `s_pop / sqrt(n)` with `s_pop` the ddof=0 standard
deviation, and the mean of `n = 2000` normal draws is normal to within the Monte Carlo error.
So each percentile endpoint should equal `mean -/+ z * s_pop / sqrt(n)`. With `B` replicates the
sample `q`-quantile has standard deviation `sqrt(q(1-q)/B) / phi(z_q)` in units of the standard
error: 0.060 at `q = 0.025` and `B = 2000`. The test allows 0.3 standard errors, five Monte Carlo
standard deviations; measured on the five seeds here: at most 0.093. Over 200 repeated iid
samples of `n = 200` both intervals should cover the true mean at about the nominal 95%; the
binomial standard deviation of the coverage is 0.015, and the test allows [0.89, 0.99]. Measured:
0.935 (percentile) and 0.930 (BCa).

BCa reduction: with `z0 = 0` and `a = 0` the BCa levels are `Phi(Phi^-1(alpha / 2))`, the
percentile levels, so the two intervals agree to the rounding of `Phi(Phi^-1(p))`. The cases are
constructed: `stat` is 0 on every jackknife sample (equal jackknife values give `a = 0`), and on
the full series it returns a value strictly between the two middle replicates of an even `B`,
computed here from `arch` with the same seed, so exactly half the replicates lie below it and
`z0 = Phi^-1(0.5) = 0`. The test allows 1e-14 times the data range; measured: 2.4e-17.
"""

import numpy as np
import pytest
from arch.bootstrap import StationaryBootstrap
from scipy.stats import norm

from quantcore.statistics import stationary_bootstrap_ci

pytestmark = pytest.mark.oracle

RNG_SEED = 20261105


def _mean(v: np.ndarray) -> float:
    return float(v.mean())


# --- normal theory --------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_percentile_interval_of_the_mean_matches_normal_theory(seed: int) -> None:
    n, n_boot = 2000, 2000
    x = np.random.default_rng(RNG_SEED + seed).normal(0.0005, 0.01, n)
    result = stationary_bootstrap_ci(
        x, _mean, rng=np.random.default_rng(seed), n_boot=n_boot, ci=0.95, block_length=1.0
    )
    se = float(x.std(ddof=0)) / np.sqrt(n)
    z = float(norm.ppf(0.975))
    assert abs(result.ci_low - (x.mean() - z * se)) <= 0.3 * se
    assert abs(result.ci_high - (x.mean() + z * se)) <= 0.3 * se


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_interval_of_the_mean_covers_at_the_nominal_rate_for_iid_data(method: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    true_mean = 0.0005
    covered = 0
    n_samples = 200
    for _ in range(n_samples):
        x = rng.normal(true_mean, 0.01, 200)
        result = stationary_bootstrap_ci(
            x, _mean, rng=rng, n_boot=300, ci=0.95, method=method, block_length=1.0
        )
        covered += result.ci_low <= true_mean <= result.ci_high
    assert 0.89 <= covered / n_samples <= 0.99


# --- BCa reduces to the percentile interval when z0 = 0 and a = 0 ---------------------------------


def _median_splitting_stat(x: np.ndarray, block_size: int, seed: int, n_boot: int):
    """Statistic with a = 0 and z0 = 0 for `stationary_bootstrap_ci(x, ..., seed, n_boot)`.

    0 on any shorter array (the jackknife samples); on `x` itself the midpoint of the two middle
    replicate means of the resamples `arch` draws from `default_rng(seed)`; the mean elsewhere.
    """
    bootstrap = StationaryBootstrap(block_size, x, seed=np.random.default_rng(seed))
    replicates = np.sort([_mean(data[0]) for data, _kw in bootstrap.bootstrap(n_boot)])
    below, above = replicates[n_boot // 2 - 1], replicates[n_boot // 2]
    assert below < above
    split = 0.5 * (below + above)
    assert below < split < above

    def stat(v: np.ndarray) -> float:
        if v.shape[0] != x.shape[0]:
            return 0.0
        if np.array_equal(v, x):
            return split
        return _mean(v)

    return stat


@pytest.mark.parametrize(
    ("n", "kind", "block_length", "ci", "seed"),
    [
        (30, "iid", 1.0, 0.95, 1),
        (120, "ar1", 4.0, 0.9, 2),
        (400, "skewed", 2.0, 0.99, 3),
        (1000, "ar1", 9.0, 0.8, 4),
    ],
)
def test_bca_equals_percentile_when_z0_and_acceleration_are_zero(
    n: int, kind: str, block_length: float, ci: float, seed: int
) -> None:
    assert norm.ppf(0.5) == 0.0
    r = np.random.default_rng(RNG_SEED + n)
    if kind == "skewed":
        x = r.lognormal(0.0, 0.8, n) * 0.01
    else:
        e = r.normal(0.0, 0.01, n)
        x = e.copy()
        if kind == "ar1":
            for t in range(1, n):
                x[t] = 0.6 * x[t - 1] + e[t]
    n_boot = 600
    stat = _median_splitting_stat(x, round(block_length), seed, n_boot)
    kwargs = {"n_boot": n_boot, "ci": ci, "block_length": block_length}
    bca = stationary_bootstrap_ci(x, stat, rng=np.random.default_rng(seed), method="bca", **kwargs)
    pct = stationary_bootstrap_ci(
        x, stat, rng=np.random.default_rng(seed), method="percentile", **kwargs
    )
    scale = float(np.ptp(x))
    assert abs(bca.ci_low - pct.ci_low) <= 1e-14 * scale
    assert abs(bca.ci_high - pct.ci_high) <= 1e-14 * scale
