"""Price-path simulators driven by a caller-supplied `numpy.random.Generator`.

Six processes, one function each, with the common signature

    simulate_<process>_price_paths(rng, n_paths, n_steps, dt, *, s0=..., <process parameters>)

Each returns prices (not returns) as a float64 array of shape (n_paths, n_steps + 1), one row per
path, column 0 the starting price. `dt` is the length of one step in the time unit of the rate
parameters (years when drift and volatility are annual). With x = log(S) and Z ~ N(0, 1):

    GBM:      x_{k+1} = x_k + (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z
    exp-OU:   x_{k+1} = theta + (x_k - theta)*exp(-kappa*dt)
                        + sqrt(sigma^2/(2*kappa) * (1 - exp(-2*kappa*dt)))*Z
    Heston:   v+ = max(v_k, 0)
              x_{k+1} = x_k + (mu - v+/2)*dt + sqrt(v+)*sqrt(dt)*Z1
              v_{k+1} = v_k + kappa*(theta - v+)*dt
                        + xi*sqrt(v+)*sqrt(dt)*(rho*Z1 + sqrt(1 - rho^2)*Z2)
    GARCH-t:  r_k = mu*dt + sqrt(h_k)*z_k,  z_k = T_k*sqrt((nu - 2)/nu),  T_k ~ Student-t(nu)
              h_{k+1} = omega + alpha*h_k*z_k^2 + beta*h_k,  h_0 = omega/(1 - alpha - beta)
              x_{k+1} = x_k + r_k
    Regime:   s_{k+1} in {0, 1} is a Markov chain with s_0 = 0, P(0 -> 1) = p01, P(1 -> 0) = p10
              x_{k+1} = x_k + (mu_s - sigma_s^2/2)*dt + sigma_s*sqrt(dt)*Z,  s = s_{k+1}
    Jump:     N ~ Poisson(lambda*dt),  J ~ Normal(N*m, N*s^2),  J = 0 when N = 0
              x_{k+1} = x_k + (mu - sigma^2/2 - lambda*c)*dt + sigma*sqrt(dt)*Z + J,
              c = exp(m + s^2/2) - 1

GBM, exp-OU and the jump diffusion are sampled from their exact transition law on the grid. The
Heston scheme is an Euler discretisation and carries a bias that shrinks with dt. The GARCH and
regime-switching models are discrete-time models by definition.

These functions sit beside the seed-taking simulators in `sde_solver.py` and `stochastic.py` and
differ from them in taking the generator itself, so that a caller can draw from one stream before
the call, inside it and after it.

Reproducibility:
    For a given state of `rng`, on a given platform and NumPy build, the array returned and the
    state `rng` is left in are fixed, and callers may rely on both (for instance by hashing data
    derived from the paths, or by continuing to draw from `rng`). Each function documents which
    variates it draws and in what order. That order, and the grouping of every floating-point
    product and sum in this module, are part of the contract: writing `sigma * (sqrt(dt) * z)`
    for `(sigma * sqrt(dt)) * z`, or drawing one (n_paths, n_steps) block where the function
    draws a vector per step, returns a different array. Arguments are validated before the first
    draw, so a call rejected by that validation leaves `rng` untouched. NumPy does not promise
    that its samplers or `exp` round identically on different platforms, so two machines can
    disagree in the last bits.

Why NumPy:
    The simulators are plain NumPy, an exception to this package's use of Numba for path
    simulation. The variates have to come from NumPy's own samplers, because Numba's versions
    of `Generator.standard_t` and `Generator.normal` do not return the same values as NumPy's.
    The arithmetic stays in NumPy as well, because a compiled kernel calls the C library's
    `exp`, which is not guaranteed to round like NumPy's elementwise `exp` on every platform.

Non-finite inputs and overflow:
    Every real-valued parameter, including the drifts and levels (`mu`, `mu0`, `mu1`, `theta`,
    `jump_mean_log`), is checked finite before the first draw. Prices are the exponential of a
    float64 log-price, so finite parameters can still overflow; if any simulated price is not
    finite the call raises `FloatingPointError` after all draws, so `rng` has been advanced.
    Prices that underflow to `0.0` are returned as they are.

References:
    Black, F. and Scholes, M. (1973), "The Pricing of Options and Corporate Liabilities"
    (geometric Brownian motion). Glasserman, P. (2004), "Monte Carlo Methods in Financial
    Engineering", Chapter 3 (exact simulation of geometric Brownian motion, Gaussian processes
    and jump diffusions on a fixed time grid). Uhlenbeck, G.E. and Ornstein, L.S. (1930), "On
    the Theory of the Brownian Motion" (OU transition law); Schwartz, E.S. (1997), "The
    Stochastic Behavior of Commodity Prices: Implications for Valuation and Hedging" (OU process
    on the log price). Heston, S.L. (1993), "A Closed-Form Solution for Options with Stochastic
    Volatility" (model); Lord, R., Koekkoek, R. and van Dijk, D. (2010), "A Comparison of Biased
    Simulation Schemes for Stochastic Volatility Models" (full-truncation Euler scheme).
    Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity" (variance
    recursion); Bollerslev, T. (1987), "A Conditionally Heteroskedastic Time Series Model for
    Speculative Prices and Rates of Return" (standardised Student-t innovations). Hamilton, J.D.
    (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the
    Business Cycle" (Markov regime switching). Merton, R.C. (1976), "Option Pricing When
    Underlying Stock Returns Are Discontinuous" (jump diffusion). See docs/REFERENCES.md.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import numpy.typing as npt


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_finite_prices(
    prices: npt.NDArray[np.float64], log_prices: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if not np.isfinite(log_prices).all():
        raise FloatingPointError(
            "a simulated log-price is not finite: a drift or variance term exceeds the "
            "float64 range"
        )
    if not np.isfinite(prices).all():
        raise FloatingPointError(
            "simulated prices overflowed: exp of the log-price exceeds the float64 range"
        )
    return prices


_LOG_FLOAT_MAX = math.log(sys.float_info.max)


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be strictly positive and finite")


def _require_non_negative(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be non-negative and finite")


def _require_open_unit_interval(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1")


def _validate_grid(n_paths: int, n_steps: int, dt: float, s0: float) -> None:
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    _require_positive("dt", dt)
    _require_positive("s0", s0)


def _validate_simulate_gbm_price_paths(
    n_paths: int, n_steps: int, dt: float, s0: float, mu: float, sigma: float
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    _require_finite("mu", mu)
    _require_positive("sigma", sigma)


def simulate_gbm_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
) -> npt.NDArray[np.float64]:
    """Simulate geometric Brownian motion price paths with the exact log-normal step.

    Each step adds `(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z` to the log-price, which is the exact
    transition for constant `mu` and `sigma`. Over `T = n_steps*dt` the log-return has mean
    `(mu - sigma^2/2)*T` and variance `sigma^2*T`, and `E[S_T] = s0*exp(mu*T)`.

    Random draws: one call `rng.standard_normal((n_paths, n_steps))`; element `[i, k]` drives
    step `k` of path `i`.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of `mu` and `sigma` (strictly positive, finite).
        s0: Starting price (strictly positive, finite).
        mu: Drift per unit time, so that `E[S_t] = s0*exp(mu*t)` (finite).
        sigma: Volatility per square root of unit time (strictly positive, finite).

    Returns:
        Prices of shape (n_paths, n_steps + 1); column 0 equals `s0` exactly.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive, `mu` is not finite, or `dt`, `s0`
            or `sigma` is not strictly positive and finite.
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    References:
        Black & Scholes (1973); Glasserman (2004), Chapter 3. See docs/REFERENCES.md.
    """
    _validate_simulate_gbm_price_paths(n_paths, n_steps, dt, s0, mu, sigma)
    return _simulate_gbm_price_paths(rng, n_paths, n_steps, dt, s0, mu, sigma)


def _simulate_gbm_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    log_increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * rng.standard_normal(
        (n_paths, n_steps)
    )
    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    np.cumsum(log_increments, axis=1, out=log_prices[:, 1:])
    prices: npt.NDArray[np.float64] = s0 * np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)


def _validate_simulate_exp_ou_price_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    kappa: float,
    theta: float | None,
    sigma: float,
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    if theta is not None:
        _require_finite("theta", theta)
    _require_positive("kappa", kappa)
    _require_positive("sigma", sigma)


def simulate_exp_ou_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    kappa: float = 1.0,
    theta: float | None = None,
    sigma: float = 0.2,
) -> npt.NDArray[np.float64]:
    """Simulate mean-reverting price paths: an Ornstein-Uhlenbeck process on the log-price.

    With `X = log(S)` and `dX = kappa*(theta - X)*dt + sigma*dW`, each step samples the exact
    Gaussian transition, mean `theta + (X_k - theta)*exp(-kappa*dt)` and variance
    `sigma^2/(2*kappa) * (1 - exp(-2*kappa*dt))`, and the prices are `exp(X)`. `theta` is the
    long-run mean of `log(S)`, not of `S`. In the stationary regime `log(S)` has mean `theta`,
    variance `sigma^2/(2*kappa)` and lag-one autocorrelation `exp(-kappa*dt)`. The process has
    no drift parameter.

    Random draws: for each step in turn, one call `rng.standard_normal(n_paths)`, element `i`
    driving path `i`.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of `kappa` and `sigma` (strictly positive,
            finite).
        s0: Starting price (strictly positive, finite).
        kappa: Mean-reversion speed per unit time (strictly positive, finite).
        theta: Long-run mean of `log(S)`. None (default) uses `log(s0)`, so the process starts
            at its long-run mean. Finite when given.
        sigma: Volatility of `log(S)` per square root of unit time (strictly positive, finite).

    Returns:
        Prices of shape (n_paths, n_steps + 1). Column 0 is `exp(log(s0))` evaluated in floating
        point, which is the same number in every row but not always bit-equal to `s0`: it can
        differ from `s0` in the last few bits, by a relative amount of at most about
        `(|ln(s0)| + 2) * 2.2e-16`.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive, `theta` is given and not finite,
            or `dt`, `s0`, `kappa` or `sigma` is not strictly positive and finite.
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    Notes:
        The step variance is formed from `1 - exp(-2*kappa*dt)`, which loses relative precision
        as `kappa*dt` shrinks (a relative error of up to about `1.1e-16 / (2*kappa*dt)`) and is
        exactly zero, giving paths without noise, once `2*kappa*dt` is below about `5.6e-17`.

    References:
        Uhlenbeck & Ornstein (1930); Schwartz (1997); Glasserman (2004), Chapter 3. See
        docs/REFERENCES.md.
    """
    _validate_simulate_exp_ou_price_paths(n_paths, n_steps, dt, s0, kappa, theta, sigma)
    return _simulate_exp_ou_price_paths(rng, n_paths, n_steps, dt, s0, kappa, theta, sigma)


def _simulate_exp_ou_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    kappa: float,
    theta: float | None,
    sigma: float,
) -> npt.NDArray[np.float64]:
    theta_value = math.log(s0) if theta is None else theta
    decay = math.exp(-kappa * dt)
    variance = (sigma * sigma) / (2.0 * kappa) * (1.0 - math.exp(-2.0 * kappa * dt))
    std = math.sqrt(variance)

    log_prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_prices[:, 0] = math.log(s0)
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        log_prices[:, step + 1] = (
            theta_value + (log_prices[:, step] - theta_value) * decay + std * z
        )
    prices: npt.NDArray[np.float64] = np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)


def _validate_simulate_heston_price_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    _require_finite("mu", mu)
    _require_non_negative("v0", v0)
    _require_positive("kappa", kappa)
    _require_positive("theta", theta)
    _require_positive("xi", xi)
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")


def simulate_heston_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    mu: float = 0.05,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.5,
) -> npt.NDArray[np.float64]:
    """Simulate Heston stochastic-volatility price paths with the full-truncation Euler scheme.

    The model is `dS = mu*S*dt + sqrt(v)*S*dW1`, `dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2`,
    `corr(dW1, dW2) = rho`. With `v+ = max(v, 0)`, each step advances

        log S:  + (mu - v+/2)*dt + sqrt(v+)*sqrt(dt)*Z1
        v:      + kappa*(theta - v+)*dt + xi*sqrt(v+)*sqrt(dt)*(rho*Z1 + sqrt(1 - rho^2)*Z2)

    The variance itself is not floored and may be negative between steps; only `v+` enters the
    drift and the diffusion. The variance path is not returned. `2*kappa*theta < xi^2` (Feller
    condition violated) is accepted. The log-price step is conditionally log-normal with mean
    growth `exp(mu*dt)`, so `E[S_T] = s0*exp(mu*T)` holds for the scheme without discretisation
    error. The expected average variance over `T = n_steps*dt` in the continuous-time model is
    `A = theta + (v0 - theta)*(1 - exp(-kappa*T))/(kappa*T)`, with `E[log(S_T/s0)] = mu*T - A*T/2`;
    the scheme matches both up to a discretisation bias that shrinks with `dt`.

    Random draws: for each step in turn, `rng.standard_normal(n_paths)` for the price shock `Z1`,
    then `rng.standard_normal(n_paths)` for `Z2`, the part of the variance shock independent of
    `Z1`. Both are drawn at every step, also when `rho` is 1 or -1.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of the rate parameters (strictly positive,
            finite).
        s0: Starting price (strictly positive, finite).
        mu: Drift of the price per unit time (the drift of the log-price is `mu - v/2`; finite).
        v0: Initial variance per unit time (non-negative, finite).
        kappa: Mean-reversion speed of the variance (strictly positive, finite).
        theta: Long-run variance (strictly positive, finite).
        xi: Volatility of the variance (strictly positive, finite).
        rho: Correlation between the price and variance shocks, in [-1, 1].

    Returns:
        Prices of shape (n_paths, n_steps + 1); column 0 equals `s0` exactly.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive; `mu` is not finite; `dt`, `s0`,
            `kappa`, `theta` or `xi` is not strictly positive and finite; `v0` is negative or not
            finite; or `rho` is outside [-1, 1].
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    References:
        Heston (1993); Lord, Koekkoek & van Dijk (2010). See docs/REFERENCES.md.
    """
    _validate_simulate_heston_price_paths(n_paths, n_steps, dt, s0, mu, v0, kappa, theta, xi, rho)
    return _simulate_heston_price_paths(
        rng, n_paths, n_steps, dt, s0, mu, v0, kappa, theta, xi, rho
    )


def _simulate_heston_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> npt.NDArray[np.float64]:
    sqrt_dt = math.sqrt(dt)
    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    variance = np.full(n_paths, v0, dtype=np.float64)
    for step in range(n_steps):
        v_pos = np.maximum(variance, 0.0)
        sqrt_v_pos = np.sqrt(v_pos)
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.standard_normal(n_paths)
        log_prices[:, step + 1] = (
            log_prices[:, step] + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z1
        )
        variance = variance + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * sqrt_dt * z2
    prices: npt.NDArray[np.float64] = s0 * np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)


def _validate_simulate_garch_t_price_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    _require_finite("mu", mu)
    _require_positive("omega", omega)
    _require_non_negative("alpha", alpha)
    _require_non_negative("beta", beta)
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be less than 1")
    if not (math.isfinite(nu) and nu > 2.0):
        raise ValueError("nu must be finite and greater than 2")


def simulate_garch_t_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    mu: float = 0.0,
    omega: float = 1e-5,
    alpha: float = 0.05,
    beta: float = 0.90,
    nu: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Simulate GARCH(1,1) price paths with standardised Student-t innovations.

    The log-return of step `k` is `r_k = mu*dt + sqrt(h_k)*z_k`, where
    `z_k = T_k*sqrt((nu - 2)/nu)` with `T_k` Student-t distributed with `nu` degrees of freedom,
    so that `z_k` has unit variance. The conditional variance follows
    `h_{k+1} = omega + alpha*h_k*z_k^2 + beta*h_k` from `h_0 = omega/(1 - alpha - beta)`, the
    unconditional variance, and the prices are `s0*exp(cumulative sum of r)`.

    `omega`, `alpha` and `beta` are per-step quantities: the variance of a log-return is
    `omega/(1 - alpha - beta)` whatever `dt` is, and `dt` scales the drift only. `r_k` is a
    log-return without a convexity correction, so `E[S_T]` is not `s0*exp(mu*T)`.

    Random draws: for each step in turn, one call `rng.standard_t(nu, size=n_paths)`, element
    `i` driving path `i`.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of `mu` (strictly positive, finite).
        s0: Starting price (strictly positive, finite).
        mu: Drift of the log-price per unit time (finite).
        omega: Intercept of the variance recursion, per step (strictly positive, finite).
        alpha: Weight of the squared innovation (non-negative, finite).
        beta: Weight of the previous conditional variance (non-negative, finite).
        nu: Degrees of freedom of the Student-t innovation (finite, greater than 2).

    Returns:
        Prices of shape (n_paths, n_steps + 1); column 0 equals `s0` exactly.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive; `mu` is not finite; `dt`, `s0` or
            `omega` is not strictly positive and finite; `alpha` or `beta` is negative or not
            finite; `alpha + beta >= 1`; or `nu` is not finite and greater than 2.
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    References:
        Bollerslev (1986); Bollerslev (1987). See docs/REFERENCES.md.
    """
    _validate_simulate_garch_t_price_paths(n_paths, n_steps, dt, s0, mu, omega, alpha, beta, nu)
    return _simulate_garch_t_price_paths(rng, n_paths, n_steps, dt, s0, mu, omega, alpha, beta, nu)


def _simulate_garch_t_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
) -> npt.NDArray[np.float64]:
    t_scale = math.sqrt((nu - 2.0) / nu)
    unconditional_variance = omega / (1.0 - alpha - beta)

    log_returns = np.empty((n_paths, n_steps), dtype=np.float64)
    h = np.full(n_paths, unconditional_variance, dtype=np.float64)
    for step in range(n_steps):
        z = rng.standard_t(nu, size=n_paths) * t_scale
        log_returns[:, step] = mu * dt + np.sqrt(h) * z
        h = omega + alpha * (h * z * z) + beta * h

    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    np.cumsum(log_returns, axis=1, out=log_prices[:, 1:])
    prices: npt.NDArray[np.float64] = s0 * np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)


def _validate_simulate_regime_switching_price_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu0: float,
    sigma0: float,
    mu1: float,
    sigma1: float,
    p01: float,
    p10: float,
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    _require_finite("mu0", mu0)
    _require_finite("mu1", mu1)
    _require_positive("sigma0", sigma0)
    _require_positive("sigma1", sigma1)
    _require_open_unit_interval("p01", p01)
    _require_open_unit_interval("p10", p10)


def simulate_regime_switching_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    mu0: float = 0.10,
    sigma0: float = 0.15,
    mu1: float = -0.20,
    sigma1: float = 0.40,
    p01: float = 0.02,
    p10: float = 0.10,
) -> npt.NDArray[np.float64]:
    """Simulate price paths that switch between two geometric Brownian motion regimes.

    A two-state Markov chain starts every path in regime 0. At each step the transition is drawn
    first (a path in regime 0 moves to 1 with probability `p01`, one in regime 1 moves to 0 with
    probability `p10`), and the log-price then takes the exact GBM step of the regime just
    entered, `(mu_s - sigma_s^2/2)*dt + sigma_s*sqrt(dt)*Z`. The first return is therefore
    already in regime 1 with probability `p01`.

    `p01` and `p10` are probabilities per step and are not scaled by `dt`. The stationary regime
    probabilities are `pi0 = p10/(p01 + p10)` and `pi1 = p01/(p01 + p10)`, and the stationary
    variance of a one-step log-return is
    `pi0*sigma0^2*dt + pi1*sigma1^2*dt + pi0*pi1*(m0 - m1)^2` with
    `m_s = (mu_s - sigma_s^2/2)*dt`.

    Random draws: for each step in turn, `rng.random(n_paths)` for the switch decision (path `i`
    switches when its uniform is below its switch probability), then
    `rng.standard_normal(n_paths)` for the return.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of the drifts and volatilities (strictly
            positive, finite).
        s0: Starting price (strictly positive, finite).
        mu0: Drift per unit time in regime 0 (finite).
        sigma0: Volatility per square root of unit time in regime 0 (strictly positive, finite).
        mu1: Drift per unit time in regime 1 (finite).
        sigma1: Volatility per square root of unit time in regime 1 (strictly positive, finite).
        p01: Probability per step of moving from regime 0 to regime 1, strictly between 0 and 1.
        p10: Probability per step of moving from regime 1 to regime 0, strictly between 0 and 1.

    Returns:
        Prices of shape (n_paths, n_steps + 1); column 0 equals `s0` exactly.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive; `mu0` or `mu1` is not finite;
            `dt`, `s0`, `sigma0` or `sigma1` is not strictly positive and finite; or `p01` or
            `p10` is not strictly between 0 and 1.
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    References:
        Hamilton (1989). See docs/REFERENCES.md.
    """
    _validate_simulate_regime_switching_price_paths(
        n_paths, n_steps, dt, s0, mu0, sigma0, mu1, sigma1, p01, p10
    )
    return _simulate_regime_switching_price_paths(
        rng, n_paths, n_steps, dt, s0, mu0, sigma0, mu1, sigma1, p01, p10
    )


def _simulate_regime_switching_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu0: float,
    sigma0: float,
    mu1: float,
    sigma1: float,
    p01: float,
    p10: float,
) -> npt.NDArray[np.float64]:
    sqrt_dt = math.sqrt(dt)
    state = np.zeros(n_paths, dtype=np.int64)
    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    for step in range(n_steps):
        u = rng.random(n_paths)
        switch_prob = np.where(state == 0, p01, p10)
        state = np.where(u < switch_prob, 1 - state, state)
        mu_t = np.where(state == 0, mu0, mu1)
        sigma_t = np.where(state == 0, sigma0, sigma1)
        z = rng.standard_normal(n_paths)
        log_prices[:, step + 1] = (
            log_prices[:, step] + (mu_t - 0.5 * sigma_t * sigma_t) * dt + sigma_t * sqrt_dt * z
        )
    prices: npt.NDArray[np.float64] = s0 * np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)


def _validate_simulate_jump_diffusion_price_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean_log: float,
    jump_std_log: float,
) -> None:
    _validate_grid(n_paths, n_steps, dt, s0)
    _require_finite("mu", mu)
    _require_positive("sigma", sigma)
    _require_non_negative("jump_intensity", jump_intensity)
    _require_finite("jump_mean_log", jump_mean_log)
    _require_non_negative("jump_std_log", jump_std_log)
    if jump_mean_log + 0.5 * jump_std_log * jump_std_log > _LOG_FLOAT_MAX:
        raise ValueError(
            "jump_mean_log and jump_std_log give an expected jump size "
            "exp(jump_mean_log + 0.5*jump_std_log**2) beyond float64"
        )


def simulate_jump_diffusion_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    jump_intensity: float = 1.0,
    jump_mean_log: float = 0.0,
    jump_std_log: float = 0.1,
) -> npt.NDArray[np.float64]:
    """Simulate Merton jump-diffusion price paths with a compensated drift.

    Jumps arrive as a Poisson process of intensity `lambda = jump_intensity` and multiply the
    price by `exp(Y)` with `Y ~ Normal(m, s^2)`, `m = jump_mean_log`, `s = jump_std_log`. Each
    step draws the number of jumps `N ~ Poisson(lambda*dt)` and their summed log-size
    `J ~ Normal(N*m, N*s^2)` (zero when `N = 0`), and adds
    `(mu - sigma^2/2 - lambda*c)*dt + sigma*sqrt(dt)*Z + J` to the log-price, which is the exact
    transition on the grid. The compensator `c = exp(m + s^2/2) - 1` makes
    `E[S_T] = s0*exp(mu*T)`, and `Var[log(S_T/s0)] = (sigma^2 + lambda*(m^2 + s^2))*T`.

    Random draws: for each step in turn, `rng.standard_normal(n_paths)` for the diffusion, then
    `rng.poisson(jump_intensity*dt, size=n_paths)` for the jump counts, then one call
    `rng.normal(loc, scale)` with `loc = N*m` and `scale = s*sqrt(max(N, 1))` for the jump
    sizes. The jump-size normal is drawn for every path at every step and discarded where
    `N = 0`.

    Args:
        rng: Generator the variates are drawn from; it is advanced by the call.
        n_paths: Number of independent paths (positive).
        n_steps: Number of time steps per path (positive).
        dt: Length of one step, in the time unit of the rate parameters (strictly positive,
            finite).
        s0: Starting price (strictly positive, finite).
        mu: Total drift per unit time, jumps included, so that `E[S_t] = s0*exp(mu*t)` (finite).
        sigma: Diffusive volatility per square root of unit time (strictly positive, finite).
        jump_intensity: Expected number of jumps per unit time (non-negative, finite).
        jump_mean_log: Mean of the log jump size (finite).
        jump_std_log: Standard deviation of the log jump size (non-negative, finite).

    Returns:
        Prices of shape (n_paths, n_steps + 1); column 0 equals `s0` exactly.

    Raises:
        ValueError: If `n_paths` or `n_steps` is not positive; `mu` or `jump_mean_log` is not
            finite; `dt`, `s0` or `sigma` is not strictly positive and finite; or
            `jump_intensity` or `jump_std_log` is negative or not finite; or the expected jump
            size `exp(jump_mean_log + 0.5*jump_std_log**2)` exceeds float64. NumPy's Poisson sampler
            also raises `ValueError` when `jump_intensity*dt` exceeds the largest mean it accepts
            (about 9.2e18); that happens after the first draw, so `rng` has been advanced.
        FloatingPointError: If a simulated price overflows float64, or a log-price is not
            finite because a drift or variance term overflowed. The generator has been
            advanced by then.

    References:
        Merton (1976); Glasserman (2004), Chapter 3. See docs/REFERENCES.md.
    """
    _validate_simulate_jump_diffusion_price_paths(
        n_paths, n_steps, dt, s0, mu, sigma, jump_intensity, jump_mean_log, jump_std_log
    )
    return _simulate_jump_diffusion_price_paths(
        rng, n_paths, n_steps, dt, s0, mu, sigma, jump_intensity, jump_mean_log, jump_std_log
    )


def _simulate_jump_diffusion_price_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean_log: float,
    jump_std_log: float,
) -> npt.NDArray[np.float64]:
    compensator = math.exp(jump_mean_log + 0.5 * jump_std_log * jump_std_log) - 1.0
    drift = (mu - 0.5 * sigma * sigma - jump_intensity * compensator) * dt
    sqrt_dt = math.sqrt(dt)

    log_prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(jump_intensity * dt, size=n_paths)
        jump_sum = np.where(
            n_jumps > 0,
            rng.normal(n_jumps * jump_mean_log, jump_std_log * np.sqrt(np.maximum(n_jumps, 1))),
            0.0,
        )
        log_prices[:, step + 1] = log_prices[:, step] + drift + sigma * sqrt_dt * z + jump_sum
    prices: npt.NDArray[np.float64] = s0 * np.exp(log_prices)
    return _require_finite_prices(prices, log_prices)
