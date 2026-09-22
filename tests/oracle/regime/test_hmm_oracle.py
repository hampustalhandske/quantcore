"""Oracle tests for finding 2c (owner follow-up): `hmm_predict_proba` is a smoother.

`hmm_predict_proba` computes SMOOTHED posteriors P(s_t=k | y_1..y_T) via
forward-backward -- it uses future observations, making it unsuitable for
any causal use (a live signal, walk-forward backtesting) despite its
`predict_proba`-style name. Renaming it would break the public API
contract other projects depend on, so it's kept (now with an explicit
docstring warning), and a new causal function, `hmm_filtered_proba`
(Hamilton 1989's filtered-probability recursion, forward pass only), is
added alongside it.

Oracles:
- `hmmlearn.hmm.GaussianHMM.predict_proba` for `hmm_predict_proba` (the
  smoother) -- same forward-backward algorithm, standard library.
- An independent, from-scratch plain-probability-space (not log-space,
  not quantcore's Numba kernel) forward recursion written directly in this
  test, implementing Hamilton (1989)'s filter equation, for
  `hmm_filtered_proba`. hmmlearn does not expose a public filtered-only
  API (its `predict_proba` is always the smoother), so this is the
  independent reference implementation ground rule 5 calls for rather
  than a second library.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

import hmmlearn.hmm as hl

from quantcore.regime.hmm import hmm_filtered_proba, hmm_predict_proba

pytestmark = pytest.mark.oracle

SEED = 20240921


def _synthetic_two_state(
    n_blocks: int = 6,
    block_size: int = 50,
    means: tuple[float, float] = (-2.0, 2.0),
    std: float = np.sqrt(0.1),
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_states = np.array(
        [i % 2 for i in range(n_blocks) for _ in range(block_size)], dtype=np.int64
    )
    means_arr = np.array(means)
    return rng.normal(loc=means_arr[true_states], scale=std)


def _reference_filtered_proba(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    initial_probs: np.ndarray,
) -> np.ndarray:
    """Hamilton (1989) filtered-probability recursion, plain probability
    space (no log-domain, no Numba) -- an independent reference
    implementation, not a copy of quantcore's `_forward_log`.
    """
    n_steps = observations.shape[0]
    n_states = means.shape[0]
    filtered = np.empty((n_steps, n_states))

    emission = np.exp(-0.5 * (observations[0] - means) ** 2 / variances) / np.sqrt(
        2.0 * np.pi * variances
    )
    joint = initial_probs * emission
    filtered[0] = joint / joint.sum()

    for t in range(1, n_steps):
        predicted = filtered[t - 1] @ transition_matrix  # P(s_t | y_1..y_{t-1})
        emission = np.exp(-0.5 * (observations[t] - means) ** 2 / variances) / np.sqrt(
            2.0 * np.pi * variances
        )
        joint = predicted * emission
        filtered[t] = joint / joint.sum()

    return filtered


TRANSITION_MATRIX = np.array([[0.98, 0.02], [0.02, 0.98]])
MEANS = np.array([-2.0, 2.0])
VARIANCES = np.array([0.1, 0.1])
INITIAL_PROBS = np.array([0.5, 0.5])


class TestSmoothedPosteriorsMatchHmmlearn:
    def test_predict_proba_matches_hmmlearn_exactly(self) -> None:
        observations = _synthetic_two_state()

        model = hl.GaussianHMM(n_components=2, covariance_type="diag", init_params="")
        model.startprob_ = INITIAL_PROBS
        model.transmat_ = TRANSITION_MATRIX
        model.means_ = MEANS.reshape(-1, 1)
        model.covars_ = VARIANCES.reshape(-1, 1)

        expected = model.predict_proba(observations.reshape(-1, 1))
        actual = hmm_predict_proba(observations, TRANSITION_MATRIX, MEANS, VARIANCES, INITIAL_PROBS)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestFilteredPosteriorsMatchIndependentReferenceImplementation:
    def test_filtered_proba_matches_hand_derived_forward_recursion(self) -> None:
        # Noisier / closer states than the 6-sigma-separated default, so
        # filtered and smoothed genuinely differ and this test isn't
        # trivially satisfied by both collapsing to near-certainty.
        observations = _synthetic_two_state(means=(-0.3, 0.3), std=1.0)
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        means = np.array([-0.3, 0.3])
        variances = np.array([1.0, 1.0])

        expected = _reference_filtered_proba(
            observations, transition_matrix, means, variances, INITIAL_PROBS
        )
        actual = hmm_filtered_proba(
            observations, transition_matrix, means, variances, INITIAL_PROBS
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_filtered_and_smoothed_agree_only_at_the_final_observation(self) -> None:
        # At t=T (the last observation), there's no future left to smooth
        # against, so gamma_T(k) = filtered_T(k) exactly. Everywhere before
        # that, they generally differ -- the whole point of the distinction
        # this finding is about.
        observations = _synthetic_two_state(means=(-0.3, 0.3), std=1.0)
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        means = np.array([-0.3, 0.3])
        variances = np.array([1.0, 1.0])

        filtered = hmm_filtered_proba(
            observations, transition_matrix, means, variances, INITIAL_PROBS
        )
        smoothed = hmm_predict_proba(
            observations, transition_matrix, means, variances, INITIAL_PROBS
        )
        np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-10)
        assert not np.allclose(filtered[:-1], smoothed[:-1], atol=1e-3)
