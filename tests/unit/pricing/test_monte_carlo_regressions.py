from __future__ import annotations

import math

from quantcore.pricing.monte_carlo import monte_carlo_call_price


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
