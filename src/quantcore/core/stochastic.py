"""Mean-reverting SDE path simulators and the CIR fit.

    OU:     dX_t = theta*(mu - X_t)*dt + sigma*dW_t
                 exact step: X_{t+dt} = X_t*exp(-theta*dt) + mu*(1-exp(-theta*dt))
                             + sigma*sqrt((1-exp(-2*theta*dt))/(2*theta))*Z

    CIR:    dr_t = kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*dW_t
            Milstein: r_{t+dt} = r_t + kappa*(theta-r_t)*dt + sigma*sqrt(r_t)*Z*sqrt(dt)
                                + 0.25*sigma^2*dt*(Z^2 - 1)

References:
    Uhlenbeck, G.E. and Ornstein, L.S. (1930), "On the Theory of the Brownian
    Motion"; Vasicek, O. (1977), "An Equilibrium Characterization of the Term
    Structure" (OU / Vasicek exact transition).
    Cox, J.C., Ingersoll, J.E., and Ross, S.A. (1985), "A Theory of the Term
    Structure of Interest Rates" (CIR square-root diffusion). See
    docs/REFERENCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    seed: int | None = None,
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
        seed: Seed for reproducibility. None (default) draws fresh
            entropy each call (non-reproducible unless you pass a seed),
            matching `simulate_gbm_paths`'s convention.

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
    seed: int | None,
) -> npt.NDArray[np.float64]:
    dt = t / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))
    return _simulate_ou_paths_kernel(theta, mu, sigma, x0, dt, z)


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
    seed: int | None = None,
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
        seed: Seed for reproducibility. None (default) draws fresh
            entropy each call (non-reproducible unless you pass a seed),
            matching `simulate_gbm_paths`'s convention.

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
    seed: int | None,
) -> npt.NDArray[np.float64]:
    dt = t / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_paths, n_steps))
    return _simulate_cir_paths_kernel(r0, kappa, theta, sigma, dt, z)


class CIRNoMeanReversionError(RuntimeError):
    """Raised by `cir_fit` when the estimated mean-reversion speed kappa is <= 0.

    The input series shows no mean reversion, so no CIR model describes it. A subclass of
    `RuntimeError`, so code that catches `RuntimeError` also catches it. The other inadmissible
    fits (theta <= 0, sigma^2 <= 0) raise plain `RuntimeError`.
    """


@dataclass(frozen=True)
class CIRParams:
    """Fitted CIR parameters, ready to pass straight into `simulate_cir_paths`.

    Attributes:
        kappa: Mean-reversion speed.
        theta: Long-run mean.
        sigma: Diffusion coefficient.
    """

    kappa: float
    theta: float
    sigma: float


# A CIR fit needs at least this many observed transitions (rate_series.size -
# 1) so the no-intercept, 2-regressor GLS regression below has strictly
# positive residual degrees of freedom (n_obs - 2 >= 2) to estimate a
# meaningful residual variance from, rather than an exact-fit degenerate one.
_CIR_FIT_MIN_OBSERVATIONS = 5


def _validate_cir_fit_inputs(rate_series: npt.NDArray[np.float64], dt: float) -> None:
    if rate_series.size < _CIR_FIT_MIN_OBSERVATIONS:
        raise ValueError(
            f"rate_series must contain at least {_CIR_FIT_MIN_OBSERVATIONS} observations"
        )
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive")
    if np.any(rate_series <= 0.0):
        raise ValueError(
            "rate_series must be strictly positive -- the GLS estimator divides "
            "through by sqrt(r_t) to homoskedasticize the regression"
        )


def cir_fit(rate_series: npt.NDArray[np.float64], dt: float) -> CIRParams:
    """Estimate CIR parameters (kappa, theta, sigma) from an observed rate series.

    Uses the Chan-Karolyi-Longstaff-Sanders (CKLS 1992) style GLS moment
    estimator, applied to the CIR Euler discretization

        r_{t+dt} - r_t = kappa*theta*dt - kappa*r_t*dt + sigma*sqrt(r_t*dt)*Z_t

    Dividing through by sqrt(r_t) homoskedasticizes the residual (its
    variance becomes sigma^2*dt, no longer a function of r_t), giving the
    no-intercept linear regression

        (r_{t+dt} - r_t)/sqrt(r_t) = b1*[dt/sqrt(r_t)] + b2*[dt*sqrt(r_t)] + eps_t,
        eps_t ~ N(0, sigma^2*dt)

    with b1 = kappa*theta and b2 = -kappa, solved by ordinary least squares
    (`numpy.linalg.lstsq`) on the two regressors. kappa and theta are
    recovered from (b1, b2), and sigma from the residual variance of the fit.
    This closed-form GLS/CKLS approach was chosen over Euler quasi-MLE
    because it has no optimizer to fail to converge and its parameter
    recovery can be validated exactly against a known-parameter synthetic
    series (see test_stochastic.py).

    Args:
        rate_series: Observed rate path, evenly spaced by `dt`, strictly
            positive (at least `_CIR_FIT_MIN_OBSERVATIONS` observations).
        dt: Time step between consecutive observations in `rate_series`
            (must be strictly positive).

    Returns:
        `CIRParams(kappa, theta, sigma)`, in the same parameter order as
        `simulate_cir_paths`, so the output can be piped straight in.

    Raises:
        ValueError: If `rate_series` is too short, contains a non-positive
            value, or `dt` is not strictly positive.
        CIRNoMeanReversionError: If the estimated kappa is <= 0: the input
            series shows no mean reversion. A `RuntimeError` subclass.
        RuntimeError: If the estimated theta or sigma^2 is <= 0, outside the
            admissible CIR parameter space. No silent fallback is returned in
            either case.

    References:
        Chan, K.C., Karolyi, G.A., Longstaff, F.A., and Sanders, A.B. (1992),
        "An Empirical Comparison of Alternative Models of the Short-Term
        Interest Rate" (CKLS GLS moment estimator). Cox, Ingersoll & Ross
        (1985) for the underlying CIR model. See docs/REFERENCES.md.
    """
    _validate_cir_fit_inputs(rate_series, dt)
    return _cir_fit(rate_series, dt)


def _cir_fit(rate_series: npt.NDArray[np.float64], dt: float) -> CIRParams:
    r = rate_series[:-1]
    dr = np.diff(rate_series)
    sqrt_r = np.sqrt(r)

    y = dr / sqrt_r
    x1 = dt / sqrt_r
    x2 = dt * sqrt_r
    design = np.column_stack((x1, x2))

    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    b1, b2 = coeffs

    n_obs = y.shape[0]
    residuals = y - design @ coeffs
    residual_variance = float(np.sum(residuals**2)) / (n_obs - 2)

    kappa = -float(b2)
    if kappa <= 0.0:
        raise CIRNoMeanReversionError(
            "cir_fit: estimated kappa <= 0 -- input series shows no discernible "
            "mean reversion, so no admissible CIR fit exists for this data"
        )
    theta = float(b1) / kappa
    if theta <= 0.0:
        raise RuntimeError("cir_fit: estimated theta <= 0 -- not an admissible CIR long-run mean")
    sigma_sq = residual_variance / dt
    if sigma_sq <= 0.0:
        raise RuntimeError(
            "cir_fit: estimated sigma^2 <= 0 -- not an admissible CIR diffusion coefficient"
        )
    sigma = float(np.sqrt(sigma_sq))

    return CIRParams(kappa=kappa, theta=theta, sigma=sigma)
