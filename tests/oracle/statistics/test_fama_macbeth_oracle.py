"""Oracle test for finding C10: `fama_macbeth_regression`'s NW-adjusted standard errors.

Confirms `n_lags_nw` gives Newey-West HAC standard errors of the mean risk
premia matching `linearmodels.panel.FamaMacBeth`'s `cov_type="kernel"`,
`kernel="bartlett"` (with `debiased=False`, since `linearmodels` applies a
small-sample debiasing correction by default that quantcore's
`fama_macbeth_regression` does not implement — a documented, deliberate
scope limit, not a mismatch in the Newey-West computation itself; see
`fama_macbeth_regression`'s docstring).

The mean risk premia themselves (`lambdas.mean(axis=0)`) are unaffected by
`n_lags_nw` and already matched exactly in this comparison, confirming the
two-pass procedure's first stage is correct independent of the SE method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("linearmodels")

from linearmodels.panel import FamaMacBeth

from quantcore.statistics.regression import fama_macbeth_regression

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _panel_frames(returns: np.ndarray, betas: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    n_periods, n_assets = returns.shape
    n_factors = betas.shape[1]
    entities = [f"a{i}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product(
        [entities, pd.RangeIndex(n_periods)], names=["entity", "time"]
    )
    exog_rows = np.repeat(betas, n_periods, axis=0)
    dep_rows = returns.T.ravel()
    exog_df = pd.DataFrame(exog_rows, index=index, columns=[f"f{j}" for j in range(n_factors)])
    dep_s = pd.Series(dep_rows, index=index, name="ret")
    return exog_df, dep_s


class TestFamaMacBethNewelWestMatchesLinearmodels:
    def test_mean_and_nw_tstats_match(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n_periods, n_assets, n_factors = 200, 20, 2
        true_lambda = np.array([0.02, -0.01])
        betas = rng.normal(scale=1.0, size=(n_assets, n_factors))

        returns = np.empty((n_periods, n_assets))
        for t in range(n_periods):
            lambda_t = true_lambda + rng.normal(scale=0.02, size=n_factors)
            returns[t, :] = betas @ lambda_t + rng.normal(scale=0.01, size=n_assets)

        n_lags = 3
        quantcore_mean, quantcore_t = fama_macbeth_regression(returns, betas, n_lags_nw=n_lags)

        exog_df, dep_s = _panel_frames(returns, betas)
        model = FamaMacBeth(dep_s, exog_df)
        result = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=n_lags, debiased=False)

        np.testing.assert_allclose(quantcore_mean, result.params.to_numpy(), rtol=1e-8)
        np.testing.assert_allclose(quantcore_t, result.tstats.to_numpy(), rtol=1e-6)

    def test_default_none_matches_plain_time_series_se_not_nw(self) -> None:
        # The default (n_lags_nw=None) is NOT the NW-adjusted SE -- confirms
        # the two paths actually differ when lambda_t has serial
        # correlation, i.e. that n_lags_nw does something.
        rng = np.random.default_rng(RNG_SEED)
        n_periods, n_assets, n_factors = 200, 20, 2
        true_lambda = np.array([0.02, -0.01])
        betas = rng.normal(scale=1.0, size=(n_assets, n_factors))
        # Autocorrelated lambda_t via an AR(1) on the true premium itself.
        lambda_path = np.empty((n_periods, n_factors))
        lambda_path[0] = true_lambda
        for t in range(1, n_periods):
            lambda_path[t] = (
                0.8 * lambda_path[t - 1]
                + 0.2 * true_lambda
                + rng.normal(scale=0.01, size=n_factors)
            )
        returns = np.empty((n_periods, n_assets))
        for t in range(n_periods):
            returns[t, :] = betas @ lambda_path[t] + rng.normal(scale=0.01, size=n_assets)

        _, t_plain = fama_macbeth_regression(returns, betas)
        _, t_nw = fama_macbeth_regression(returns, betas, n_lags_nw=5)

        assert not np.allclose(t_plain, t_nw, rtol=1e-3)
