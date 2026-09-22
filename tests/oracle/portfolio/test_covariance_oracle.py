"""Oracle test for `ledoit_wolf_shrinkage` vs `sklearn.covariance.LedoitWolf`.

Covers Workstream D priority 2 (owner follow-up, decision 4): "mind the
1/n vs 1/(n-1) ... conventions."

Confirmed defect (found this pass, not merely a documentation gap):
`ledoit_wolf_shrinkage` used the unbiased sample covariance (ddof=1,
1/(T-1)) throughout its Theorem 1 computation, where Ledoit & Wolf (2004)'s
own derivation — and `sklearn`'s implementation of it — uses the biased/
population normalization (1/T) consistently for the sample covariance S,
the shrinkage target's scale mu, and the shrinkage-intensity estimate.
Mixing 1/(T-1) into the a Theorem 1 whose derivation assumes 1/T made
`ledoit_wolf_shrinkage`'s output a different (not Theorem-1-optimal)
estimator, not just the same estimator on a slightly different scale —
before the fix, the shrunk covariance differed from `sklearn`'s by ~1%
of typical variance magnitudes on a 5-asset synthetic panel, not
numerical noise. Fixed by switching the internal sample covariance to
1/T; `sample_covariance` (the standalone public function) is unaffected
and keeps its own 1/(T-1) convention, since it isn't part of the Theorem 1
computation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.covariance import LedoitWolf

from quantcore.portfolio.covariance import ledoit_wolf_shrinkage

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestLedoitWolfMatchesSklearn:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n_assets", [2, 5, 10])
    def test_shrunk_covariance_matches(self, n_assets: int, seed: int) -> None:
        rng = np.random.default_rng(RNG_SEED + seed)
        n_obs = 200
        a = rng.normal(size=(n_assets, n_assets))
        true_cov = a @ a.T + np.eye(n_assets)
        returns = rng.multivariate_normal(mean=np.zeros(n_assets), cov=true_cov, size=n_obs)

        quantcore_cov = ledoit_wolf_shrinkage(returns)
        sklearn_cov = LedoitWolf().fit(returns).covariance_

        np.testing.assert_allclose(quantcore_cov, sklearn_cov, rtol=1e-8, atol=1e-12)

    def test_shrinkage_intensity_matches(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_assets, n_obs = 6, 150
        a = rng.normal(size=(n_assets, n_assets))
        true_cov = a @ a.T + np.eye(n_assets)
        returns = rng.multivariate_normal(mean=np.zeros(n_assets), cov=true_cov, size=n_obs)

        sklearn_result = LedoitWolf().fit(returns)
        quantcore_cov = ledoit_wolf_shrinkage(returns)

        # Recover quantcore's implied alpha from its output and compare to
        # sklearn's reported shrinkage_ directly.
        sample = np.cov(returns.T, bias=True)
        mu_hat = np.trace(sample) / n_assets
        target = mu_hat * np.eye(n_assets)
        # quantcore_cov = alpha*target + (1-alpha)*sample -- solve for alpha
        # via the trace of (quantcore_cov - sample) / (target - sample),
        # using the Frobenius inner product to average over all entries.
        numerator = np.sum((quantcore_cov - sample) * (target - sample))
        denominator = np.sum((target - sample) ** 2)
        implied_alpha = numerator / denominator

        assert implied_alpha == pytest.approx(sklearn_result.shrinkage_, rel=1e-6)
