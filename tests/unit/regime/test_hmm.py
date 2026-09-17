"""Tests for Gaussian HMM fitting (Baum-Welch), Viterbi decoding, and state posteriors."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.regime.hmm import hmm_decode, hmm_fit, hmm_predict_proba, select_hmm_n_states

SEED = 42


def _two_state_synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    # Alternating blocks of 50 samples from N(-2, 0.1) and N(2, 0.1).
    rng = np.random.default_rng(SEED)
    n_blocks = 6
    block_size = 50
    true_states = np.array(
        [i % 2 for i in range(n_blocks) for _ in range(block_size)], dtype=np.int64
    )
    means = np.array([-2.0, 2.0])
    std = np.sqrt(0.1)
    observations = rng.normal(loc=means[true_states], scale=std)
    return observations, true_states


def _best_label_matched_accuracy(decoded: np.ndarray, true_states: np.ndarray) -> float:
    direct = float(np.mean(decoded == true_states))
    swapped = float(np.mean((1 - decoded) == true_states))
    return max(direct, swapped)


class TestHmmFit:
    def test_invalid_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            hmm_fit(np.array([]), n_states=2)

    def test_invalid_n_states_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError):
            hmm_fit(np.array([0.1, 0.2, 0.3]), n_states=0)

    def test_recovers_well_separated_means(self) -> None:
        observations, _ = _two_state_synthetic_data()
        _, means, _, _ = hmm_fit(observations, n_states=2, seed=SEED)
        recovered = sorted(means.tolist())
        assert recovered[0] == pytest.approx(-2.0, abs=0.3)
        assert recovered[1] == pytest.approx(2.0, abs=0.3)

    def test_output_shapes(self) -> None:
        observations, _ = _two_state_synthetic_data()
        transition_matrix, means, variances, initial_probs = hmm_fit(
            observations, n_states=2, seed=SEED
        )
        assert transition_matrix.shape == (2, 2)
        assert means.shape == (2,)
        assert variances.shape == (2,)
        assert initial_probs.shape == (2,)


class TestSelectHmmNStates:
    def test_invalid_criterion_raises(self) -> None:
        observations, _ = _two_state_synthetic_data()
        with pytest.raises(ValueError):
            select_hmm_n_states(observations, max_states=3, criterion="hqic")

    def test_invalid_max_states_raises(self) -> None:
        observations, _ = _two_state_synthetic_data()
        with pytest.raises(ValueError):
            select_hmm_n_states(observations, max_states=0)

    def test_does_not_crash_when_max_states_exceeds_true_count(self) -> None:
        observations, _ = _two_state_synthetic_data()
        n_states = select_hmm_n_states(observations, max_states=4)
        assert 1 <= n_states <= 4

    def test_recovers_true_number_of_states_with_bic(self) -> None:
        # State-count selection is noisy in finite samples (BIC can favor
        # one extra state that captures a spurious sub-cluster), so we only
        # assert the well-separated two-regime structure is not collapsed
        # to a single state and not wildly overfit.
        observations, _ = _two_state_synthetic_data()
        n_states = select_hmm_n_states(observations, max_states=4, criterion="bic")
        assert n_states == 2


class TestHmmDecode:
    def test_invalid_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            hmm_decode(
                np.array([]),
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_invalid_transition_matrix_shape_raises(self) -> None:
        observations = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            hmm_decode(
                observations,
                transition_matrix=np.eye(3),
                means=np.array([-2.0, 2.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_invalid_means_length_mismatch_raises(self) -> None:
        observations = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            hmm_decode(
                observations,
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0, 0.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_invalid_negative_variance_raises(self) -> None:
        observations = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            hmm_decode(
                observations,
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0]),
                variances=np.array([-0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_recovers_true_state_sequence(self) -> None:
        observations, true_states = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        decoded = hmm_decode(observations, transition_matrix, means, variances, initial_probs)
        accuracy = _best_label_matched_accuracy(decoded, true_states)
        assert accuracy > 0.95


class TestHmmPredictProba:
    def test_invalid_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            hmm_predict_proba(
                np.array([]),
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_invalid_initial_probs_length_mismatch_raises(self) -> None:
        observations = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            hmm_predict_proba(
                observations,
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.3, 0.2]),
            )

    def test_output_shape_and_rows_sum_to_one(self) -> None:
        observations, _ = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        posteriors = hmm_predict_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        assert posteriors.shape == (observations.shape[0], 2)
        assert np.allclose(posteriors.sum(axis=1), 1.0)

    def test_high_confidence_matches_true_state(self) -> None:
        observations, true_states = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        posteriors = hmm_predict_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        predicted_states = np.argmax(posteriors, axis=1)
        accuracy = _best_label_matched_accuracy(predicted_states, true_states)
        assert accuracy > 0.95
