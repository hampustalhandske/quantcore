from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from quantcore.pricing.monte_carlo import monte_carlo_call_price, monte_carlo_put_price


def test_single_path_antithetic_standard_error_is_finite() -> None:
    price, stderr = monte_carlo_call_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_paths=1,
        num_steps=10,
        seed=1,
        antithetic=True,
    )
    assert price >= 0.0
    assert math.isfinite(price)
    assert math.isfinite(stderr)


class TestSingleAntitheticPairStandardError:
    """One mirrored pair has no sample standard deviation; the result is 0.0, not NaN."""

    @pytest.mark.parametrize("pricer", [monte_carlo_call_price, monte_carlo_put_price])
    def test_two_antithetic_paths_give_a_finite_standard_error(self, pricer) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            price, std_error = pricer(100.0, 100.0, 0.05, 0.2, 1.0, 2, 10, seed=1)
        assert np.isfinite(price)
        assert std_error == 0.0
