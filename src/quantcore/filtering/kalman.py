"""Linear Kalman filtering, RTS smoothing, and a time-varying hedge ratio filter.

State model:       x_t = A*x_{t-1} + w_t,   w_t ~ N(0, Q)
Observation model: y_t = H*x_t + v_t,        v_t ~ N(0, R)

Predict:
    x_{t|t-1} = A * x_{t-1|t-1}
    P_{t|t-1} = A * P_{t-1|t-1} * A^T + Q

Update:
    K_t = P_{t|t-1} * H^T * (H*P_{t|t-1}*H^T + R)^{-1}   (Kalman gain)
    x_{t|t} = x_{t|t-1} + K_t*(y_t - H*x_{t|t-1})
    P_{t|t} = (I - K_t*H) * P_{t|t-1}

RTS smoother backward pass, from t=T-1 down to 0:
    G_t = P_{t|t} * A^T * P_{t+1|t}^{-1}
    x_{t|T} = x_{t|t} + G_t*(x_{t+1|T} - x_{t+1|t})
    P_{t|T} = P_{t|t} + G_t*(P_{t+1|T} - P_{t+1|t})*G_t^T

References:
    Kalman, R.E. (1960), "A New Approach to Linear Filtering and Prediction
    Problems." (Predict/update equations.)

    Rauch, H.E., Tung, F., and Striebel, C.T. (1965), "Maximum Likelihood
    Estimates of Linear Dynamic Systems." (RTS smoother backward pass.)

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


def _validate_kalman_inputs(
    observations: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    P0: npt.NDArray[np.float64],
) -> None:
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("observations must be a non-empty 2D array of shape (T, m)")
    n = x0.shape[0]
    m = observations.shape[1]
    if A.shape != (n, n):
        raise ValueError(f"A must have shape ({n}, {n}) matching x0, got {A.shape}")
    if H.shape[1] != n:
        raise ValueError(f"H must have {n} columns matching the state dimension, got {H.shape}")
    if H.shape[0] != m:
        raise ValueError(f"H must have {m} rows matching observations' dimension, got {H.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape ({n}, {n}) matching x0, got {Q.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must have shape ({m}, {m}) matching observations, got {R.shape}")
    if P0.shape != (n, n):
        raise ValueError(f"P0 must have shape ({n}, {n}) matching x0, got {P0.shape}")


def kalman_filter(
    observations: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    P0: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the linear Kalman filter forward pass over a sequence of observations.

    Args:
        observations: Observed values, shape (T, m).
        A: State transition matrix, shape (n, n).
        H: Observation matrix, shape (m, n).
        Q: Process noise covariance, shape (n, n).
        R: Observation noise covariance, shape (m, m).
        x0: Initial state mean, shape (n,).
        P0: Initial state covariance, shape (n, n).

    Returns:
        Tuple (filtered_means, filtered_covs) of shape (T, n) and (T, n, n).
    """
    _validate_kalman_inputs(observations, A, H, Q, R, x0, P0)
    return _kalman_filter(observations, A, H, Q, R, x0, P0)


def _kalman_filter(
    observations: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    P0: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    t_steps, n = observations.shape[0], x0.shape[0]
    identity = np.eye(n)

    filtered_means = np.empty((t_steps, n), dtype=np.float64)
    filtered_covs = np.empty((t_steps, n, n), dtype=np.float64)

    x_prev, p_prev = x0, P0
    for t in range(t_steps):
        x_pred = A @ x_prev
        p_pred = A @ p_prev @ A.T + Q

        innovation_cov = H @ p_pred @ H.T + R
        kalman_gain = p_pred @ H.T @ np.linalg.inv(innovation_cov)
        innovation = observations[t] - H @ x_pred

        x_upd = x_pred + kalman_gain @ innovation
        p_upd = (identity - kalman_gain @ H) @ p_pred

        filtered_means[t] = x_upd
        filtered_covs[t] = p_upd
        x_prev, p_prev = x_upd, p_upd

    return filtered_means, filtered_covs


def kalman_smooth(
    observations: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    P0: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the Kalman forward pass followed by the RTS backward smoothing pass.

    Args:
        observations: Observed values, shape (T, m).
        A: State transition matrix, shape (n, n).
        H: Observation matrix, shape (m, n).
        Q: Process noise covariance, shape (n, n).
        R: Observation noise covariance, shape (m, m).
        x0: Initial state mean, shape (n,).
        P0: Initial state covariance, shape (n, n).

    Returns:
        Tuple (smoothed_means, smoothed_covs) of shape (T, n) and (T, n, n).
    """
    _validate_kalman_inputs(observations, A, H, Q, R, x0, P0)
    return _kalman_smooth(observations, A, H, Q, R, x0, P0)


def _kalman_smooth(
    observations: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    P0: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    filtered_means, filtered_covs = _kalman_filter(observations, A, H, Q, R, x0, P0)
    t_steps, n = observations.shape[0], x0.shape[0]

    # Predicted (one-step-ahead) means/covs are needed by the RTS backward
    # recursion; recompute them from the filtered sequence (P_{t|t-1} depends
    # only on P_{t-1|t-1} via A, Q).
    pred_means = np.empty((t_steps, n), dtype=np.float64)
    pred_covs = np.empty((t_steps, n, n), dtype=np.float64)
    prev_mean, prev_cov = x0, P0
    for t in range(t_steps):
        pred_means[t] = A @ prev_mean
        pred_covs[t] = A @ prev_cov @ A.T + Q
        prev_mean, prev_cov = filtered_means[t], filtered_covs[t]

    smoothed_means = np.empty_like(filtered_means)
    smoothed_covs = np.empty_like(filtered_covs)
    smoothed_means[-1] = filtered_means[-1]
    smoothed_covs[-1] = filtered_covs[-1]

    for t in range(t_steps - 2, -1, -1):
        gain = filtered_covs[t] @ A.T @ np.linalg.inv(pred_covs[t + 1])
        smoothed_means[t] = filtered_means[t] + gain @ (smoothed_means[t + 1] - pred_means[t + 1])
        smoothed_covs[t] = (
            filtered_covs[t] + gain @ (smoothed_covs[t + 1] - pred_covs[t + 1]) @ gain.T
        )

    return smoothed_means, smoothed_covs


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
