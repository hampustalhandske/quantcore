"""Mean-reverting and stochastic-volatility SDE path simulators.

    OU:     dX_t = theta*(mu - X_t)*dt + sigma*dW_t
                 exact step: X_{t+dt} = X_t*exp(-theta*dt) + mu*(1-exp(-theta*dt))
                             + sigma*sqrt((1-exp(-2*theta*dt))/(2*theta))*Z

    Heston: dS_t = mu*S_t*dt + sqrt(V_t)*S_t*dW_t^S
            dV_t = kappa*(theta - V_t)*dt + xi*sqrt(V_t)*dW_t^V,  corr(dW^S, dW^V) = rho
            Milstein correction for V: + 0.5*xi^2*(dW_t^V)^2 - 0.5*xi^2*dt

    CIR:    dr_t = kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*dW_t
            Milstein: r_{t+dt} = r_t + kappa*(theta-r_t)*dt + sigma*sqrt(r_t)*Z*sqrt(dt)
                                + 0.25*sigma^2*dt*(Z^2 - 1)

References:
    Uhlenbeck, G.E. and Ornstein, L.S. (1930), "On the Theory of the Brownian
    Motion"; Vasicek, O. (1977), "An Equilibrium Characterization of the Term
    Structure" (OU / Vasicek exact transition). Heston, S.L. (1993), "A
    Closed-Form Solution for Options with Stochastic Volatility" (Heston SDE).
    Cox, J.C., Ingersoll, J.E., and Ross, S.A. (1985), "A Theory of the Term
    Structure of Interest Rates" (CIR square-root diffusion). See
    docs/REFERENCES.md.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt


def _validate_simulate_ou_paths(
    theta: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> None:
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")


def simulate_ou_paths(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Simulate Ornstein-Uhlenbeck paths using the exact conditional-Gaussian transition.

    Args:
        theta: Mean-reversion speed (must be strictly positive).
        mu: Long-run mean.
        sigma: Diffusion coefficient (must be non-negative).
        x0: Initial value X_0.
        t: Total time horizon (must be non-negative).
        n_paths: Number of independent sample paths.
        n_steps: Number of discretization steps.
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) of simulated OU paths, including
        X_0 at column 0.

    References:
        Uhlenbeck & Ornstein (1930); Vasicek (1977). See docs/REFERENCES.md.
    """
    _validate_simulate_ou_paths(theta, sigma, t, n_paths, n_steps)
    return _simulate_ou_paths(theta, mu, sigma, x0, t, n_paths, n_steps, seed)


@numba.njit(cache=True)
def _simulate_ou_paths_kernel(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_paths, n_steps = z.shape
    paths = np.empty((n_paths, n_steps + 1))
    decay = np.exp(-theta * dt)
    diffusion_scale = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
    for i in range(n_paths):
        x = x0
        paths[i, 0] = x0
        for k in range(n_steps):
            x = x * decay + mu * (1.0 - decay) + diffusion_scale * z[i, k]
            paths[i, k + 1] = x
    return paths


def _simulate_ou_paths(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    dt = t / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))
    return _simulate_ou_paths_kernel(theta, mu, sigma, x0, dt, z)


def _validate_simulate_heston_paths(
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> None:
    if s0 <= 0.0:
        raise ValueError("s0 must be strictly positive")
    if v0 < 0.0:
        raise ValueError("v0 must be non-negative")
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive")
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive")
    if xi < 0.0:
        raise ValueError("xi must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")


def simulate_heston_paths(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate Heston stochastic-volatility paths via log-Euler (S) / Milstein (V).

    Args:
        s0: Initial spot price (must be strictly positive).
        v0: Initial variance (must be non-negative).
        mu: Drift of the log-price process.
        kappa: Mean-reversion speed of variance (must be strictly positive).
        theta: Long-run variance (must be strictly positive).
        xi: Vol-of-vol (must be non-negative).
        rho: Correlation between dW^S and dW^V, in [-1, 1].
        t: Total time horizon (must be non-negative).
        n_paths: Number of independent sample paths.
        n_steps: Number of discretization steps.
        seed: Seed for reproducibility.

    Returns:
        Tuple (S_paths, V_paths), each of shape (n_paths, n_steps + 1),
        including S_0/V_0 at column 0. Variance is reflected at 0.

    References:
        Heston (1993). See docs/REFERENCES.md.
    """
    _validate_simulate_heston_paths(s0, v0, kappa, theta, xi, rho, t, n_paths, n_steps)
    return _simulate_heston_paths(s0, v0, mu, kappa, theta, xi, rho, t, n_paths, n_steps, seed)


@numba.njit(cache=True)
def _simulate_heston_paths_kernel(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dt: float,
    z1: npt.NDArray[np.float64],
    z2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_paths, n_steps = z1.shape
    s_paths = np.empty((n_paths, n_steps + 1))
    v_paths = np.empty((n_paths, n_steps + 1))
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)

    for i in range(n_paths):
        log_s = np.log(s0)
        v = v0
        s_paths[i, 0] = s0
        v_paths[i, 0] = v0
        for k in range(n_steps):
            dw_v = z1[i, k] * sqrt_dt
            dw_s = (rho * z1[i, k] + sqrt_one_minus_rho2 * z2[i, k]) * sqrt_dt

            sqrt_v = np.sqrt(v) if v > 0.0 else 0.0
            log_s = log_s + (mu - 0.5 * v) * dt + sqrt_v * dw_s

            v_next = (
                v
                + kappa * (theta - v) * dt
                + xi * sqrt_v * dw_v
                + 0.5 * xi * xi * (dw_v * dw_v - dt)
            )
            v = max(v_next, 0.0)

            s_paths[i, k + 1] = np.exp(log_s)
            v_paths[i, k + 1] = v

    return s_paths, v_paths


def _simulate_heston_paths(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dt = t / n_steps
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(size=(n_paths, n_steps))
    z2 = rng.standard_normal(size=(n_paths, n_steps))
    return _simulate_heston_paths_kernel(s0, v0, mu, kappa, theta, xi, rho, dt, z1, z2)


def _validate_simulate_cir_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
) -> None:
    if r0 < 0.0:
        raise ValueError("r0 must be non-negative")
    if kappa <= 0.0:
        raise ValueError("kappa must be strictly positive")
    if theta <= 0.0:
        raise ValueError("theta must be strictly positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")


def simulate_cir_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Simulate CIR square-root diffusion paths via Milstein discretization.

    Args:
        r0: Initial rate (must be non-negative).
        kappa: Mean-reversion speed (must be strictly positive).
        theta: Long-run mean (must be strictly positive).
        sigma: Diffusion coefficient (must be non-negative).
        t: Total time horizon (must be non-negative).
        n_paths: Number of independent sample paths.
        n_steps: Number of discretization steps.
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_paths, n_steps + 1) of simulated CIR paths,
        including r_0 at column 0. Values are reflected at 0.

    References:
        Cox, Ingersoll & Ross (1985). See docs/REFERENCES.md.
    """
    _validate_simulate_cir_paths(r0, kappa, theta, sigma, t, n_paths, n_steps)
    return _simulate_cir_paths(r0, kappa, theta, sigma, t, n_paths, n_steps, seed)


@numba.njit(cache=True)
def _simulate_cir_paths_kernel(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_paths, n_steps = z.shape
    paths = np.empty((n_paths, n_steps + 1))
    sqrt_dt = np.sqrt(dt)
    for i in range(n_paths):
        r = r0
        paths[i, 0] = r0
        for k in range(n_steps):
            sqrt_r = np.sqrt(r) if r > 0.0 else 0.0
            zk = z[i, k]
            r_next = (
                r
                + kappa * (theta - r) * dt
                + sigma * sqrt_r * zk * sqrt_dt
                + 0.25 * sigma * sigma * dt * (zk * zk - 1.0)
            )
            r = max(r_next, 0.0)
            paths[i, k + 1] = r
    return paths


def _simulate_cir_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    dt = t / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))
    return _simulate_cir_paths_kernel(r0, kappa, theta, sigma, dt, z)
