"""Oracle tests for `ewma_variance`/`ewma_covariance` vs `pandas.Series.ewm`.

Covers Workstream D: "ewma_variance / ewma_covariance vs pandas ewm and
RiskMetrics."

quantcore's RiskMetrics (1996) recursion --
`sigma_t^2 = lambda*sigma_{t-1}^2 + (1-lambda)*epsilon_{t-1}^2` -- is a
one-step-ahead *forecast*: sigma_t^2 depends on epsilon_{t-1}, not
epsilon_t, deliberately avoiding lookahead (today's variance forecast uses
only data through yesterday). This is NOT the same quantity as
`pandas.Series.ewm(...).var()`, which is a backward-looking smoother of
the series' own value at each t (using data through t, including t
itself) -- comparing quantcore's output directly to `.ewm().var()` would
be comparing two different definitions and finding them different the way
finding C2 found is not itself informative. The correct apples-to-apples
comparison, confirmed below, is that quantcore's recursion is exactly
`pandas.Series(squared_returns).ewm(alpha=1-lambda, adjust=False).mean()`,
shifted by one lag -- a one-step-ahead EWMA forecast, not `.var()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.portfolio.covariance import ewma_covariance
from quantcore.risk.egarch import ewma_variance

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestEwmaVarianceMatchesPandasOneStepAheadEwma:
    @pytest.mark.parametrize("lambda_", [0.90, 0.94, 0.97])
    def test_matches_shifted_pandas_ewma_of_squared_returns(self, lambda_: float) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(scale=0.01, size=300)

        quantcore_variance = ewma_variance(returns, lambda_=lambda_)

        squared = pd.Series(returns**2)
        pandas_ewma_mean = squared.ewm(alpha=1.0 - lambda_, adjust=False).mean()
        shifted = pandas_ewma_mean.shift(1)
        shifted.iloc[0] = squared.iloc[0]  # matches quantcore's seed: returns[0]**2

        np.testing.assert_allclose(quantcore_variance, shifted.to_numpy(), rtol=1e-10)

    def test_does_not_match_pandas_var_directly(self) -> None:
        # Confirms these really are different quantities (the point of
        # this file's docstring), not that one side is subtly miscomputed.
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(scale=0.01, size=300)
        lambda_ = 0.94

        quantcore_variance = ewma_variance(returns, lambda_=lambda_)
        pandas_var = pd.Series(returns).ewm(alpha=1.0 - lambda_, adjust=False).var(bias=True)

        assert not np.allclose(quantcore_variance, pandas_var.to_numpy(), rtol=1e-2)


class TestEwmaCovarianceMatchesPandasOneStepAheadEwma:
    def test_matches_shifted_pandas_ewma_of_outer_products_after_burn_in(self) -> None:
        # ewma_covariance seeds at the *sample* covariance of the full
        # series (a different seed convention than ewma_variance's
        # returns[0]**2), so a long series is used here to let that seed
        # choice's influence decay away (lambda^T ~= 0), rather than trying
        # to inject the same seed into pandas' recursion directly.
        rng = np.random.default_rng(RNG_SEED)
        n_obs, n_assets = 3000, 3
        returns = rng.normal(scale=0.01, size=(n_obs, n_assets))
        lambda_ = 0.94

        quantcore_cov = ewma_covariance(returns, lambda_=lambda_)

        products = pd.DataFrame(
            {
                (i, j): returns[:, i] * returns[:, j]
                for i in range(n_assets)
                for j in range(n_assets)
            }
        )
        pandas_ewma_mean = products.ewm(alpha=1.0 - lambda_, adjust=False).mean()
        shifted = pandas_ewma_mean.shift(1)
        shifted.iloc[0] = products.iloc[0]
        terminal = shifted.iloc[-1]
        pandas_cov = np.array(
            [[terminal[(i, j)] for j in range(n_assets)] for i in range(n_assets)]
        )

        np.testing.assert_allclose(quantcore_cov, pandas_cov, rtol=1e-8)
