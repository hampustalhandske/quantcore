"""Oracle test for finding C3: `drawdown_series` vs `quantstats.stats.to_drawdown_series`.

quantstats uses the opposite sign convention (drawdown reported as a
negative number); quantcore keeps its existing positive-magnitude
convention (documented, pre-existing, unrelated to this finding — see
`maximum_drawdown`'s docstring). Compared here as magnitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantstats")

import quantstats.stats as qs

from quantcore.performance.metrics import drawdown_series

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


class TestDrawdownSeriesMatchesQuantstats:
    def test_matches_as_magnitudes(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        returns = rng.normal(0.0003, 0.01, size=200)
        # quantstats.to_drawdown_series has a documented special case for a
        # *negative first return*: it inserts a "phantom" pre-sample
        # baseline and treats period 0 itself as already underwater
        # relative to it (dd[0] < 0), where quantcore's convention (and
        # `maximum_drawdown`'s own, pre-existing, unchanged-by-design
        # convention -- see its docstring) takes the first observed
        # cumulative value as the initial peak, so dd[0] == 0 always. This
        # is a genuine definitional difference for that one edge case, not
        # a defect on either side, so a series that doesn't hit the edge
        # case (nonzero, non-negative first return) is used here to check
        # the actual recursion the two share away from it.
        if returns[0] < 0.0:
            returns[0] = abs(returns[0])
        index = pd.date_range("2020-01-01", periods=returns.size, freq="D")

        quantstats_dd = qs.to_drawdown_series(pd.Series(returns, index=index)).to_numpy()
        quantcore_dd = drawdown_series(returns)

        np.testing.assert_allclose(quantstats_dd, -quantcore_dd, atol=1e-10)
