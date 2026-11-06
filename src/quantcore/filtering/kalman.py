"""Time-varying hedge ratio filter.

The hedge ratio beta_t is the state of a random walk observed through
y_t = beta_t * x_t + v_t; see `dynamic_hedge_ratio` for the recursion.

References:
    Chan, E.P. (2013), *Algorithmic Trading: Winning Strategies and Their
    Rationale*, ch. 3. (Hedge ratio estimated as the state of a Kalman
    filter.)

    Hamilton, J.D. (1994), *Time Series Analysis*, ch. 13. (Regression with
    time-varying coefficients in state-space form.)

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_dynamic_hedge_ratio_inputs(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    obs_var: float,
    proc_var: float,
    initial_beta: float,
) -> None:
    if y.size == 0 or x.size == 0:
        raise ValueError("y and x must be non-empty")
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not (np.isfinite(obs_var) and obs_var > 0.0):
        raise ValueError("obs_var must be strictly positive and finite")
    if not (np.isfinite(proc_var) and proc_var > 0.0):
        raise ValueError("proc_var must be strictly positive and finite")
    if not np.isfinite(initial_beta):
        raise ValueError("initial_beta must be finite")


def dynamic_hedge_ratio(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    obs_var: float = 1e-3,
    proc_var: float = 1e-5,
    *,
    initial_beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Estimate a time-varying hedge ratio beta_t via a 1-D Kalman filter.

    The hedge ratio is the single state of a regression of y on x without
    an intercept, and it follows a random walk:

        y_t = beta_t * x_t + v_t,     v_t ~ N(0, obs_var)
        beta_t = beta_{t-1} + w_t,    w_t ~ N(0, proc_var)

    Before the first observation beta has the prior N(initial_beta, 1): the
    prior mean is ``initial_beta`` and the prior variance is fixed at 1.
    The default mean of 0 does not depend on the data. The single-state
    model and this prior are choices of this function. One predict/update
    step runs per observation, the first included, with P the variance of
    beta:

        P_pred = P + proc_var
        K_t    = P_pred * x_t / (x_t^2 * P_pred + obs_var)
        beta_t = beta_{t-1} + K_t * (y_t - x_t * beta_{t-1})
        P      = (1 - K_t * x_t) * P_pred

    Each observation moves beta towards y_t / x_t by the fraction
    K_t * x_t, so an observation with x_t = 0 leaves it unchanged, and the
    prior mean matters less with every informative observation.

    Args:
        y: Dependent series, shape (T,).
        x: Independent series, shape (T,).
        obs_var: Observation noise variance R (must be strictly positive).
        proc_var: Process noise variance Q (must be strictly positive).
        initial_beta: Prior mean of beta before the first observation
            (must be finite). Passing ``y[0] / x[0]`` starts the filter on
            the first observed ratio, which then is the first value
            returned.

    Returns:
        Filtered hedge ratio sequence beta_t, shape (T,). beta_t uses
        observations 0..t only.

    Raises:
        ValueError: ``y`` or ``x`` is empty or contains a NaN or infinite
            value, their shapes differ, ``obs_var`` or ``proc_var`` is not
            strictly positive and finite, or ``initial_beta`` is not finite.

    References:
        Chan (2013), ch. 3, for estimating a hedge ratio as the state of a
        Kalman filter (dynamic linear regression). Kalman (1960) for the
        predict/update equations; Hamilton (1994), ch. 13, for regression
        with time-varying coefficients in state-space form. See
        docs/REFERENCES.md.
    """
    _validate_dynamic_hedge_ratio_inputs(y, x, obs_var, proc_var, initial_beta)
    return _dynamic_hedge_ratio(y, x, obs_var, proc_var, initial_beta)


def _dynamic_hedge_ratio(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    obs_var: float,
    proc_var: float,
    initial_beta: float,
) -> npt.NDArray[np.float64]:
    t_steps = y.shape[0]
    beta = np.empty(t_steps, dtype=np.float64)

    # Prior on beta before the first observation: N(initial_beta, 1).
    beta_prev = initial_beta
    p_prev = 1.0

    for t in range(t_steps):
        beta_pred = beta_prev
        p_pred = p_prev + proc_var

        h_t = x[t]
        innovation_cov = h_t * p_pred * h_t + obs_var
        kalman_gain = p_pred * h_t / innovation_cov
        innovation = y[t] - h_t * beta_pred

        beta_upd = beta_pred + kalman_gain * innovation
        p_upd = (1.0 - kalman_gain * h_t) * p_pred

        beta[t] = beta_upd
        beta_prev, p_prev = beta_upd, p_upd

    return beta
