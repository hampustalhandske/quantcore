"""Oracle tests for `arima_fit` vs `statsmodels.tsa.arima.model.ARIMA`.

Covers Workstream D: "ARIMA, VAR, IRF, Granger vs statsmodels."

`arima_fit` uses conditional sum-of-squares (CSS) estimation (Box, Jenkins
& Reinsel 2015); `statsmodels`' `ARIMA` uses full (Kalman-filter state-
space) maximum likelihood by default, and no longer offers a pure-CSS fit
method to compare against directly (the old `statsmodels.tsa.arima_model.ARMA`
class, which did, is removed). CSS and ML are different, both valid,
asymptotically-equivalent estimators for the same ARMA model -- exact
agreement isn't the right bar here, and treating a numeric mismatch on its
own as a defect would be wrong; what's checked is that quantcore's CSS
estimate converges close to statsmodels' ML estimate on a well-behaved,
reasonably long synthetic series, as the large-sample equivalence predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.arima.model import ARIMA

from quantcore.time_series.arima import arima_fit

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestArimaFitClosesToStatsmodelsMle:
    def test_ar1_matches_closely(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 1000
        phi = 0.6
        series = np.empty(n)
        series[0] = rng.normal()
        for t in range(1, n):
            series[t] = phi * series[t - 1] + rng.normal(scale=0.3)

        ar_coefs, _, sigma2 = arima_fit(series, p=1, d=0, q=0)
        sm_result = ARIMA(series, order=(1, 0, 0), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=0.02)
        assert sigma2 == pytest.approx(sm_result.params[1], rel=0.05)

    def test_arma11_matches_closely(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        n = 1500
        phi, theta = 0.5, 0.3
        eps = rng.normal(scale=0.2, size=n)
        series = np.empty(n)
        series[0] = eps[0]
        for t in range(1, n):
            series[t] = phi * series[t - 1] + eps[t] + theta * eps[t - 1]

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=1)
        sm_result = ARIMA(series, order=(1, 0, 1), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=0.02)
        assert ma_coefs[0] == pytest.approx(sm_result.params[1], abs=0.02)
        assert sigma2 == pytest.approx(sm_result.params[2], rel=0.05)

    def test_ar1_with_differencing_matches_closely(self) -> None:
        rng = np.random.default_rng(RNG_SEED + 1)
        n = 1000
        phi = 0.4
        diffs = np.empty(n)
        diffs[0] = 0.0
        for t in range(1, n):
            diffs[t] = phi * diffs[t - 1] + rng.normal(scale=0.2)
        series = np.cumsum(diffs) + 100.0

        ar_coefs, _, sigma2 = arima_fit(series, p=1, d=1, q=0)
        sm_result = ARIMA(series, order=(1, 1, 0), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=0.02)
        assert sigma2 == pytest.approx(sm_result.params[1], rel=0.05)
