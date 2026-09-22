"""Oracle test for finding C11: `ljung_box_test`'s `model_df` parameter.

Oracle: `statsmodels.stats.diagnostic.acorr_ljungbox`'s `model_df`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.stats.diagnostic import acorr_ljungbox

from quantcore.time_series.arima import ljung_box_test

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestModelDfMatchesStatsmodels:
    @pytest.mark.parametrize("model_df", [0, 1, 2, 5])
    def test_statistic_and_pvalue_match(self, model_df: int) -> None:
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(size=300)

        quantcore_stat, quantcore_p = ljung_box_test(residuals, n_lags=10, model_df=model_df)
        sm_result = acorr_ljungbox(residuals, lags=[10], model_df=model_df)

        assert quantcore_stat == pytest.approx(sm_result["lb_stat"].to_numpy()[0], rel=1e-8)
        assert quantcore_p == pytest.approx(sm_result["lb_pvalue"].to_numpy()[0], rel=1e-8)

    def test_arma_residuals_case_matches(self) -> None:
        # The documented use case: residuals from a fitted ARMA(p, q),
        # model_df = p + q.
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(size=300)
        p, q = 2, 1

        quantcore_stat, quantcore_p = ljung_box_test(residuals, n_lags=15, model_df=p + q)
        sm_result = acorr_ljungbox(residuals, lags=[15], model_df=p + q)

        assert quantcore_stat == pytest.approx(sm_result["lb_stat"].to_numpy()[0], rel=1e-8)
        assert quantcore_p == pytest.approx(sm_result["lb_pvalue"].to_numpy()[0], rel=1e-8)
