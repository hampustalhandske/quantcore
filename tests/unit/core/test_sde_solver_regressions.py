from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.sde_solver import simulate_gbm_paths


def test_gbm_paths_remain_positive_under_high_volatility() -> None:
    paths = simulate_gbm_paths(
        spot=100.0,
        rate=0.05,
        volatility=2.5,
        time_to_maturity=1.0,
        num_steps=12,
        num_paths=2000,
        seed=1,
    )
    assert np.all(paths > 0.0)


def test_gbm_negative_time_to_maturity_raises() -> None:
    with pytest.raises(ValueError):
        simulate_gbm_paths(
            spot=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=-0.5,
            num_steps=10,
            num_paths=50,
            seed=1,
        )
