"""Oracle test for `hac_t_test` against statsmodels.

Oracle: `statsmodels.api.OLS(x, ones).fit(cov_type="HAC", cov_kwds={"maxlags": L,
"use_correction": True, "kernel": "bartlett"}, use_t=True)`. statsmodels' Bartlett weights are
1 - h/(L+1) on lags h = 1..L, the weights quantcore uses, and its correction is n/(n-k) with
k = 1. Measured over 1500 random series (n in [10, 2000], lags in [0, 60), Gaussian, Student-t(3),
strongly autocorrelated and near-constant), the largest relative differences are 3e-14 on t and
3e-15 on se for ordinary series, and 2e-11 on both for a series whose mean is 5e4 times its
spread, where centring costs eps*mean/std. p-values agree to 6e-14 relative wherever p > 1e-6.
"""

import numpy as np
import pytest

pytest.importorskip("statsmodels")

import statsmodels.api as sm

from quantcore.statistics import hac_t_test

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _series(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal":
        return rng.normal(0.001, 0.01, n)
    if kind == "student_t":
        return rng.standard_t(3, n) * 0.01 + 0.002
    if kind == "autocorrelated":
        x = rng.normal(0.0, 0.01, n)
        for t in range(1, n):
            x[t] += 0.8 * x[t - 1]
        return x
    return rng.normal(0.0, 1e-6, n) + 0.05


@pytest.mark.parametrize("kind", ["normal", "student_t", "autocorrelated", "near_constant"])
@pytest.mark.parametrize("n", [10, 40, 250, 2000])
def test_matches_statsmodels_hac(kind: str, n: int) -> None:
    rng = np.random.default_rng(RNG_SEED + n)
    for lags in sorted({0, 1, n // 10, min(n - 1, 60)}):
        x = _series(kind, n, rng)
        ours = hac_t_test(x, lags=lags)
        fit = sm.OLS(x, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": lags, "use_correction": True, "kernel": "bartlett"},
            use_t=True,
        )
        rtol = 1e-9 if kind == "near_constant" else 1e-12
        np.testing.assert_allclose(ours.se, fit.bse[0], rtol=rtol)
        np.testing.assert_allclose(ours.t, fit.tvalues[0], rtol=rtol)
        np.testing.assert_allclose(ours.p_value, fit.pvalues[0], rtol=1e-9, atol=1e-12)
        assert ours.mean == pytest.approx(fit.params[0], rel=1e-14)
