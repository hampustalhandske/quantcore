"""Oracle tests for `drawdown_series` / `maximum_drawdown` vs quantstats and empyrical.

Both oracles seed the running peak with the starting capital before the
first return is applied (`empyrical.stats.drawdown_series` prepends the
start value; `quantstats.stats.to_drawdown_series` inserts a pre-sample
baseline), so a loss on the first period is a drawdown from initial
capital. quantcore keeps its positive-magnitude sign convention
(documented on `maximum_drawdown`); both oracles report drawdowns as
negative numbers, so they are compared as magnitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantstats")
pytest.importorskip("empyrical")

import empyrical
import quantstats.stats as qs

from quantcore.performance.metrics import drawdown_series, maximum_drawdown

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_paths(n_paths: int) -> list[np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    return [
        np.clip(rng.normal(0.0003, 0.02, size=int(rng.integers(1, 200))), -0.999, None)
        for _ in range(n_paths)
    ]


class TestDrawdownSeriesMatchesQuantstats:
    def test_matches_as_magnitudes(self) -> None:
        for returns in _random_paths(200):
            index = pd.date_range("2020-01-01", periods=returns.size, freq="D")
            quantstats_dd = qs.to_drawdown_series(pd.Series(returns, index=index)).to_numpy()
            np.testing.assert_allclose(quantstats_dd, -drawdown_series(returns), atol=1e-10)

    def test_negative_first_return_is_underwater(self) -> None:
        returns = np.array([-0.05, 0.01, 0.02])
        index = pd.date_range("2020-01-01", periods=returns.size, freq="D")
        quantstats_dd = qs.to_drawdown_series(pd.Series(returns, index=index)).to_numpy()
        np.testing.assert_allclose(quantstats_dd, -drawdown_series(returns), atol=1e-10)
        assert drawdown_series(returns)[0] == pytest.approx(0.05)


class TestMaximumDrawdownMatchesEmpyrical:
    def test_matches_on_random_paths_including_length_one(self) -> None:
        for returns in _random_paths(500):
            assert maximum_drawdown(returns) == pytest.approx(
                -empyrical.max_drawdown(returns), abs=1e-12
            )

    def test_single_loss(self) -> None:
        returns = np.array([-0.5])
        assert maximum_drawdown(returns) == pytest.approx(-empyrical.max_drawdown(returns))
        assert maximum_drawdown(returns) == pytest.approx(0.5)
