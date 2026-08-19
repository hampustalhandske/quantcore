"""Monte Carlo option pricing.

Prices a European option by simulating terminal asset values under the
risk-neutral measure and averaging discounted payoffs:

    V_0 = exp(-r * T) * E[payoff(S_T)]
        ~= exp(-r * T) * (1 / N) * sum_{i=1}^{N} payoff(S_T^(i))

Path simulation is delegated to
``quantcore.core.sde_solver.simulate_gbm_paths`` (Euler-Maruyama applied to
GBM dynamics). Variance reduction via antithetic variates pairs each
Brownian increment dW with its mirror -dW, which reduces the variance of the
payoff average for a given number of simulated paths.

Results are validated against the closed-form Black-Scholes price in
tests/pricing/test_monte_carlo.py, since Black-Scholes is the known
closed-form solution for the same GBM dynamics.

References:
    Boyle, P. (1977). "Options: A Monte Carlo Approach." *Journal of
    Financial Economics*, 4(3), 323-338. (Base method.)

    Boyle, P., Broadie, M., and Glasserman, P. (1997). "Monte Carlo Methods
    for Security Pricing." *Journal of Economic Dynamics and Control*,
    21(8), 1267-1321. (Variance reduction: antithetic variates, control
    variates.)

    See REFERENCES.md.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from quantcore.core.sde_solver import (
    _generate_gbm_increments,
    _simulate_gbm_paths_numba_kernel,
    simulate_gbm_paths,
)


def _validate_monte_carlo_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if strike <= 0.0:
        raise ValueError("strike must be strictly positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")
    if num_paths <= 0:
        raise ValueError("num_paths must be a positive integer")
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")


def _price_via_monte_carlo(
    spot: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
    payoff: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> tuple[float, float]:
    """Shared MC pricing engine: simulate terminal prices, average discounted payoffs.

    Non-antithetic paths are simulated via
    ``quantcore.core.sde_solver.simulate_gbm_paths`` (Boyle 1977 base method).

    For antithetic variates (Boyle, Broadie & Glasserman 1997), each drawn
    Brownian increment dW is paired with its mirror -dW. ``simulate_gbm_paths``
    does not expose a hook for supplying/mirroring the underlying increments,
    so this draws the increments directly via the same sde_solver internals
    (``_generate_gbm_increments`` / ``_simulate_gbm_paths_numba_kernel``) that
    back ``simulate_gbm_paths``, builds the mirrored pairs, and feeds both
    halves through the identical GBM Euler-Maruyama kernel.
    """
    discount = float(np.exp(-rate * time_to_maturity))

    if not antithetic:
        terminal = simulate_gbm_paths(
            spot, rate, volatility, time_to_maturity, num_steps, num_paths, seed
        )[:, -1]
        discounted_payoffs = discount * payoff(terminal)
        price = float(np.mean(discounted_payoffs))
        if num_paths == 1:
            std_error = 0.0
        else:
            std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(num_paths))
        return price, std_error

    # Antithetic variates: draw n_pairs independent increment sets and mirror
    # them (-dW) to build negatively-correlated path pairs.
    n_pairs = (num_paths + 1) // 2
    dt = time_to_maturity / num_steps
    dw = _generate_gbm_increments(dt, num_steps, n_pairs, seed)
    dw_full = np.concatenate([dw, -dw], axis=0)[:num_paths]
    terminal = _simulate_gbm_paths_numba_kernel(spot, rate, volatility, dt, dw_full)[:, -1]
    discounted_payoffs = discount * payoff(terminal)
    price = float(np.mean(discounted_payoffs))

    if num_paths == 2 * n_pairs:
        # Every path has its mirror present: compute the standard error from
        # the per-pair averages, which correctly reflects the variance
        # reduction from the negative correlation induced by antithetic
        # pairing (Boyle, Broadie & Glasserman 1997).
        pair_averages = (discounted_payoffs[:n_pairs] + discounted_payoffs[n_pairs:]) / 2.0
        std_error = float(np.std(pair_averages, ddof=1) / np.sqrt(n_pairs))
    else:
        # Odd num_paths: one path has no mirror, so fall back to the naive
        # (unpaired) standard error estimate. For a single path, the sample
        # variance is undefined (ddof=1), so guard the estimator to return a
        # finite zero-variance result rather than propagating NaN.
        if num_paths == 1:
            std_error = 0.0
        else:
            std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(num_paths))

    return price, std_error


def monte_carlo_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Price a European call option via Monte Carlo simulation.

        V_0 = exp(-r * T) * E[max(S_T - K, 0)]

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        num_paths: Number of simulated paths.
        num_steps: Number of time discretization steps per path.
        seed: Optional seed for reproducibility.
        antithetic: Whether to use antithetic variates for variance
            reduction.

    Returns:
        Tuple of (price estimate, standard error of the estimate).

    References:
        Boyle (1977); Boyle, Broadie & Glasserman (1997) for antithetic
        variates. See REFERENCES.md.
    """
    _validate_monte_carlo_inputs(
        spot,
        strike,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
    )
    return _monte_carlo_call_price(
        spot,
        strike,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
        seed,
        antithetic,
    )


def _monte_carlo_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    def call_payoff(terminal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(terminal - strike, 0.0)

    return _price_via_monte_carlo(
        spot,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
        seed,
        antithetic,
        call_payoff,
    )


def monte_carlo_put_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Price a European put option via Monte Carlo simulation.

        V_0 = exp(-r * T) * E[max(K - S_T, 0)]

    Args:
        spot: Current price of the underlying asset (S0).
        strike: Strike price (K).
        rate: Risk-free interest rate, continuously compounded (r).
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Time to expiry in years (T).
        num_paths: Number of simulated paths.
        num_steps: Number of time discretization steps per path.
        seed: Optional seed for reproducibility.
        antithetic: Whether to use antithetic variates for variance
            reduction.

    Returns:
        Tuple of (price estimate, standard error of the estimate).

    References:
        Boyle (1977); Boyle, Broadie & Glasserman (1997) for antithetic
        variates. See REFERENCES.md.
    """
    _validate_monte_carlo_inputs(
        spot,
        strike,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
    )
    return _monte_carlo_put_price(
        spot,
        strike,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
        seed,
        antithetic,
    )


def _monte_carlo_put_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_paths: int,
    num_steps: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[float, float]:
    def put_payoff(terminal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(strike - terminal, 0.0)

    return _price_via_monte_carlo(
        spot,
        rate,
        volatility,
        time_to_maturity,
        num_paths,
        num_steps,
        seed,
        antithetic,
        put_payoff,
    )
