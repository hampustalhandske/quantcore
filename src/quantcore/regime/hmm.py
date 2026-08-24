"""Gaussian Hidden Markov Model: Baum-Welch fitting, Viterbi decoding, state posteriors.

Gaussian HMM with K hidden states, T observations:
    Transition matrix A_{ij} = P(s_t = j | s_{t-1} = i)
    Emission: p(y_t | s_t = k) = N(y_t; mu_k, sigma_k^2)
    Initial distribution: pi_k = P(s_1 = k)

Forward variable: alpha_t(k) = P(y_1,...,y_t, s_t=k)
Backward variable: beta_t(k) = P(y_{t+1},...,y_T | s_t=k)

Baum-Welch E-step (Baum et al. 1970):
    gamma_t(k) = alpha_t(k)*beta_t(k) / sum_j alpha_t(j)*beta_t(j)
    xi_t(i,j)  = alpha_t(i)*A_{ij}*b_j(y_{t+1})*beta_{t+1}(j) / sum_{i,j}(...)

Baum-Welch M-step: update pi, A, mu_k, sigma_k^2 from gamma and xi.

Viterbi (log-domain, Viterbi 1967):
    delta_t(k) = max over paths ending at k
    psi_t(k)   = argmax backpointer

All recursions are carried out in log-space with log-sum-exp for numerical
stability.

References:
    Baum, L.E., Petrie, T., Soules, G., and Weiss, N. (1970). "A Maximization
    Technique Occurring in the Statistical Analysis of Probabilistic
    Functions of Markov Chains." *Annals of Mathematical Statistics*, 41(1),
    164-171. (Baum-Welch EM.)

    Viterbi, A. (1967). "Error Bounds for Convolutional Codes and an
    Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on
    Information Theory*, 13(2), 260-269. (Viterbi decoding.)

    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2),
    357-384. (Markov-switching models applied to financial returns.)

    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt

_LOG_2PI = float(np.log(2.0 * np.pi))
_MIN_VARIANCE = 1e-8
_MIN_PROB = 1e-12


def _log_emission_matrix(
    observations: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    diff = observations[:, None] - means[None, :]
    return -0.5 * (_LOG_2PI + np.log(variances)[None, :] + diff**2 / variances[None, :])


def _logsumexp_rows(log_x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m = np.max(log_x, axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(log_x - m), axis=1, keepdims=True))


@numba.njit(cache=True)
def _forward_log(
    log_emission: npt.NDArray[np.float64],
    log_a: npt.NDArray[np.float64],
    log_pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_steps, n_states = log_emission.shape
    log_alpha = np.empty((n_steps, n_states))
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, n_steps):
        for j in range(n_states):
            vals = log_alpha[t - 1] + log_a[:, j]
            mx = np.max(vals)
            log_alpha[t, j] = mx + np.log(np.sum(np.exp(vals - mx))) + log_emission[t, j]
    return log_alpha


@numba.njit(cache=True)
def _backward_log(
    log_emission: npt.NDArray[np.float64],
    log_a: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_steps, n_states = log_emission.shape
    log_beta = np.zeros((n_steps, n_states))
    for t in range(n_steps - 2, -1, -1):
        for i in range(n_states):
            vals = log_a[i, :] + log_emission[t + 1] + log_beta[t + 1]
            mx = np.max(vals)
            log_beta[t, i] = mx + np.log(np.sum(np.exp(vals - mx)))
    return log_beta


@numba.njit(cache=True)
def _viterbi_log(
    log_emission: npt.NDArray[np.float64],
    log_a: npt.NDArray[np.float64],
    log_pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    n_steps, n_states = log_emission.shape
    delta = np.empty((n_steps, n_states))
    psi = np.zeros((n_steps, n_states), dtype=np.int64)
    delta[0] = log_pi + log_emission[0]
    for t in range(1, n_steps):
        for j in range(n_states):
            best_val = -np.inf
            best_i = 0
            for i in range(n_states):
                val = delta[t - 1, i] + log_a[i, j]
                if val > best_val:
                    best_val = val
                    best_i = i
            delta[t, j] = best_val + log_emission[t, j]
            psi[t, j] = best_i

    states = np.zeros(n_steps, dtype=np.int64)
    states[n_steps - 1] = np.argmax(delta[n_steps - 1])
    for t in range(n_steps - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _validate_hmm_fit_inputs(observations: npt.NDArray[np.float64], n_states: int) -> None:
    if observations.size == 0:
        raise ValueError("observations must be non-empty")
    if n_states < 1:
        raise ValueError("n_states must be a positive integer")


def hmm_fit(
    observations: npt.NDArray[np.float64],
    n_states: int,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Fit a Gaussian HMM via Baum-Welch EM.

    Args:
        observations: Univariate observation sequence, shape (T,).
        n_states: Number of hidden states K.
        n_iter: Maximum number of EM iterations.
        tol: Stop when log-likelihood improves by less than this.
        seed: Seed used to initialize the quantile-bucket means/variances.

    Returns:
        Tuple (transition_matrix, means, variances, initial_probs).

    References:
        Baum, Petrie, Soules & Weiss (1970). See docs/REFERENCES.md.
    """
    _validate_hmm_fit_inputs(observations, n_states)
    return _hmm_fit(observations, n_states, n_iter, tol, seed)


def _hmm_fit(
    observations: npt.NDArray[np.float64],
    n_states: int,
    n_iter: int,
    tol: float,
    seed: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    n_steps = observations.shape[0]
    k = n_states
    rng = np.random.default_rng(seed)

    sorted_obs = np.sort(observations)
    buckets = np.array_split(sorted_obs, k)
    global_var = max(float(np.var(observations)), _MIN_VARIANCE)
    means = np.array([float(np.mean(b)) for b in buckets], dtype=np.float64)
    variances = np.array(
        [max(float(np.var(b)), _MIN_VARIANCE) if b.size > 1 else global_var for b in buckets],
        dtype=np.float64,
    )
    # Break exact ties between bucket means (degenerate for small/constant
    # inputs) with a tiny seeded perturbation so states remain identifiable.
    means += rng.normal(0.0, 1e-6, size=k)

    transition_matrix = np.full((k, k), 1.0 / k, dtype=np.float64)
    initial_probs = np.full(k, 1.0 / k, dtype=np.float64)

    prev_log_likelihood = -np.inf
    for iteration in range(n_iter):
        log_emission = _log_emission_matrix(observations, means, variances)
        log_a = np.log(np.clip(transition_matrix, _MIN_PROB, None))
        log_pi = np.log(np.clip(initial_probs, _MIN_PROB, None))

        log_alpha = _forward_log(log_emission, log_a, log_pi)
        log_beta = _backward_log(log_emission, log_a)
        log_likelihood = float(_logsumexp_rows(log_alpha[-1:]).item())

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp_rows(log_gamma)
        gamma = np.exp(log_gamma)

        xi_sum = np.zeros((k, k), dtype=np.float64)
        for t in range(n_steps - 1):
            log_xi_t = (
                log_alpha[t][:, None]
                + log_a
                + log_emission[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi_t -= np.max(log_xi_t) + np.log(np.sum(np.exp(log_xi_t - np.max(log_xi_t))))
            xi_sum += np.exp(log_xi_t)

        initial_probs = gamma[0] / gamma[0].sum()
        denom = gamma[:-1].sum(axis=0)
        denom = np.clip(denom, _MIN_PROB, None)
        transition_matrix = xi_sum / denom[:, None]
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        weights = np.clip(gamma.sum(axis=0), _MIN_PROB, None)
        means = (gamma * observations[:, None]).sum(axis=0) / weights
        variances = (gamma * (observations[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        variances = np.maximum(variances, _MIN_VARIANCE)

        if iteration > 0 and (log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood

    return transition_matrix, means, variances, initial_probs


def _validate_hmm_params(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> None:
    if observations.size == 0:
        raise ValueError("observations must be non-empty")
    if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    n_states = transition_matrix.shape[0]
    if means.shape != (n_states,):
        raise ValueError("means must have shape (n_states,) matching transition_matrix")
    if variances.shape != (n_states,):
        raise ValueError("variances must have shape (n_states,) matching transition_matrix")
    if initial_probs.shape != (n_states,):
        raise ValueError("initial_probs must have shape (n_states,) matching transition_matrix")
    if np.any(variances < 0.0):
        raise ValueError("variances must be non-negative")
    if np.any(initial_probs < 0.0):
        raise ValueError("initial_probs must be non-negative")


def hmm_decode(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Decode the most likely hidden state sequence via Viterbi.

    Args:
        observations: Observation sequence, shape (T,).
        transition_matrix: State transition matrix, shape (K, K).
        means: Emission means per state, shape (K,).
        variances: Emission variances per state, shape (K,).
        initial_probs: Initial state distribution, shape (K,).

    Returns:
        Most likely state sequence, shape (T,).

    References:
        Viterbi (1967). See docs/REFERENCES.md.
    """
    _validate_hmm_params(observations, transition_matrix, means, variances, initial_probs)
    return _hmm_decode(observations, transition_matrix, means, variances, initial_probs)


def _hmm_decode(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    log_emission = _log_emission_matrix(observations, means, variances)
    log_a = np.log(np.clip(transition_matrix, _MIN_PROB, None))
    log_pi = np.log(np.clip(initial_probs, _MIN_PROB, None))
    return _viterbi_log(log_emission, log_a, log_pi)


def hmm_predict_proba(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute smoothed state posteriors gamma_t(k) via forward-backward.

    Args:
        observations: Observation sequence, shape (T,).
        transition_matrix: State transition matrix, shape (K, K).
        means: Emission means per state, shape (K,).
        variances: Emission variances per state, shape (K,).
        initial_probs: Initial state distribution, shape (K,).

    Returns:
        Smoothed state posteriors, shape (T, K), rows sum to 1.

    References:
        Baum, Petrie, Soules & Weiss (1970). See docs/REFERENCES.md.
    """
    _validate_hmm_params(observations, transition_matrix, means, variances, initial_probs)
    return _hmm_predict_proba(observations, transition_matrix, means, variances, initial_probs)


def _hmm_predict_proba(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    log_emission = _log_emission_matrix(observations, means, variances)
    log_a = np.log(np.clip(transition_matrix, _MIN_PROB, None))
    log_pi = np.log(np.clip(initial_probs, _MIN_PROB, None))

    log_alpha = _forward_log(log_emission, log_a, log_pi)
    log_beta = _backward_log(log_emission, log_a)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp_rows(log_gamma)
    return np.exp(log_gamma)
