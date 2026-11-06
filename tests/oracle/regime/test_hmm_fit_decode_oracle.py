"""Oracle tests for `hmm_fit`/`hmm_decode` vs `hmmlearn`.

Covers Workstream D priority 4 (owner follow-up, decision 4).

Two kinds of check, for the reasons given in
`tests/oracle/risk/test_garch_family_oracle.py`'s module docstring (same
principle: fixed-parameter comparison isolates the algorithm from
optimizer confounds):

- Fixed parameters (`hmm_decode`, and `hmm_fit`'s underlying forward
  algorithm/log-likelihood): set identical (transition_matrix, means,
  variances, initial_probs) on both quantcore and an `hmmlearn.GaussianHMM`
  with `init_params=""`, and compare the Viterbi path and log-likelihood
  directly -- no fitting involved, so no optimizer-to-optimizer confound.
- Fitted parameters (`hmm_fit` end to end): both packages' Baum-Welch EM
  are local optimizers, so exact parameter agreement isn't guaranteed in
  general, but on well-separated synthetic data both should converge to
  essentially the same MLE -- checked with a generous but meaningful
  tolerance, plus states-up-to-permutation label matching.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

import hmmlearn.hmm as hl

from quantcore.regime.hmm import (
    _forward_log,
    _log_emission_matrix,
    _logsumexp_rows,
    hmm_decode,
    hmm_fit,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921

TRANSITION_MATRIX = np.array([[0.98, 0.02], [0.02, 0.98]])
MEANS = np.array([-2.0, 2.0])
VARIANCES = np.array([0.1, 0.1])
INITIAL_PROBS = np.array([0.5, 0.5])


def _synthetic_two_state(
    n_blocks: int, block_size: int, means: np.ndarray, std: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_states = np.array(
        [i % 2 for i in range(n_blocks) for _ in range(block_size)], dtype=np.int64
    )
    return rng.normal(loc=means[true_states], scale=std)


def _fixed_hmmlearn_model(
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    initial_probs: np.ndarray,
) -> hl.GaussianHMM:
    model = hl.GaussianHMM(
        n_components=means.shape[0], covariance_type="diag", init_params="", implementation="log"
    )
    model.startprob_ = initial_probs
    model.transmat_ = transition_matrix
    model.means_ = means.reshape(-1, 1)
    model.covars_ = variances.reshape(-1, 1)
    return model


class TestHmmDecodeMatchesHmmlearnAtFixedParameters:
    def test_viterbi_path_matches_exactly(self) -> None:
        observations = _synthetic_two_state(6, 50, MEANS, np.sqrt(0.1), seed=RNG_SEED)
        model = _fixed_hmmlearn_model(TRANSITION_MATRIX, MEANS, VARIANCES, INITIAL_PROBS)

        hmmlearn_states = model.predict(observations.reshape(-1, 1))
        quantcore_states = hmm_decode(
            observations, TRANSITION_MATRIX, MEANS, VARIANCES, INITIAL_PROBS
        )

        np.testing.assert_array_equal(quantcore_states, hmmlearn_states)


class TestForwardAlgorithmLogLikelihoodMatchesHmmlearnAtFixedParameters:
    def test_log_likelihood_matches(self) -> None:
        observations = _synthetic_two_state(6, 50, MEANS, np.sqrt(0.1), seed=RNG_SEED)
        model = _fixed_hmmlearn_model(TRANSITION_MATRIX, MEANS, VARIANCES, INITIAL_PROBS)
        hmmlearn_loglik = model.score(observations.reshape(-1, 1))

        log_emission = _log_emission_matrix(observations, MEANS, VARIANCES)
        log_a = np.log(TRANSITION_MATRIX)
        log_pi = np.log(INITIAL_PROBS)
        log_alpha = _forward_log(log_emission, log_a, log_pi)
        quantcore_loglik = float(_logsumexp_rows(log_alpha[-1:]).item())

        assert quantcore_loglik == pytest.approx(hmmlearn_loglik, abs=1e-8)


class TestHmmFitMatchesHmmlearnOnSyntheticData:
    def test_recovers_similar_mle_as_hmmlearn(self) -> None:
        means_true = np.array([-2.0, 2.0])
        observations = _synthetic_two_state(8, 80, means_true, std=np.sqrt(0.3), seed=RNG_SEED)

        transition_matrix, means, variances, initial_probs = hmm_fit(observations, n_states=2)

        log_emission = _log_emission_matrix(observations, means, variances)
        log_a = np.log(np.clip(transition_matrix, 1e-12, None))
        log_pi = np.log(np.clip(initial_probs, 1e-12, None))
        log_alpha = _forward_log(log_emission, log_a, log_pi)
        quantcore_loglik = float(_logsumexp_rows(log_alpha[-1:]).item())

        hmmlearn_model = hl.GaussianHMM(
            n_components=2, covariance_type="diag", n_iter=200, tol=1e-6, random_state=0
        )
        hmmlearn_model.fit(observations.reshape(-1, 1))
        hmmlearn_loglik = hmmlearn_model.score(observations.reshape(-1, 1))

        # Both are local optima of the same Gaussian-HMM likelihood surface
        # on well-separated synthetic data -- close, not necessarily
        # bit-identical (different EM initializations).
        assert quantcore_loglik == pytest.approx(hmmlearn_loglik, rel=1e-3)

        # quantcore's states are sorted by ascending variance (see
        # `hmm_fit`'s docstring); match hmmlearn's states to quantcore's by
        # nearest mean before comparing, since hmmlearn has no such
        # convention.
        hmmlearn_means = hmmlearn_model.means_.ravel()
        order = np.argsort(hmmlearn_means) if means[0] < means[1] else np.argsort(-hmmlearn_means)
        hmmlearn_means_matched = hmmlearn_means[order]

        np.testing.assert_allclose(means, hmmlearn_means_matched, atol=0.1)
