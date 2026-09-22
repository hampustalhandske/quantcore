"""Tests for Gaussian HMM fitting (Baum-Welch), Viterbi decoding, and state posteriors."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.regime.hmm import (
    hmm_decode,
    hmm_filtered_proba,
    hmm_fit,
    hmm_predict_proba,
    select_hmm_n_states,
)

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


def _three_state_synthetic_data(
    seed: int, n_blocks: int = 9, block_size: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    # Cycling blocks from three regimes with distinct means AND distinct
    # variances (calm/mid/crisis), matching the downstream sort-by-variance
    # labeling convention: state 2 has both the largest mean spread and the
    # largest variance so both mean- and variance-based orderings agree.
    rng = np.random.default_rng(seed)
    means = np.array([-2.0, 0.0, 2.0])
    stds = np.array([np.sqrt(0.05), np.sqrt(0.2), np.sqrt(0.8)])
    true_states = np.array(
        [i % 3 for i in range(n_blocks) for _ in range(block_size)], dtype=np.int64
    )
    observations = rng.normal(loc=means[true_states], scale=stds[true_states])
    return observations, true_states


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

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    @pytest.mark.parametrize("n_states", [2, 3])
    def test_variances_always_sorted_ascending(self, seed: int, n_states: int) -> None:
        observations, _ = (
            _two_state_synthetic_data() if n_states == 2 else _three_state_synthetic_data(seed)
        )
        _, _, variances, _ = hmm_fit(observations, n_states=n_states, seed=seed)
        assert np.all(np.diff(variances) >= 0.0)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    @pytest.mark.parametrize("n_states", [2, 3])
    def test_transition_matrix_row_stochastic_after_relabeling(
        self, seed: int, n_states: int
    ) -> None:
        # Catches a transposition bug: permuting only one axis of
        # transition_matrix would break the row-stochastic property since
        # rows and columns index the same state permutation.
        observations, _ = (
            _two_state_synthetic_data() if n_states == 2 else _three_state_synthetic_data(seed)
        )
        transition_matrix, _, _, _ = hmm_fit(observations, n_states=n_states, seed=seed)
        assert np.allclose(transition_matrix.sum(axis=1), 1.0)

    def test_state_ordering_stable_across_refits_on_similar_data(self) -> None:
        # The actual bug being fixed: two fits of the same underlying
        # generative process (different seeds/sample windows) must land in
        # the same regime order, so downstream callers don't need to
        # re-identify "crisis" by variance themselves each time.
        obs_a, _ = _three_state_synthetic_data(seed=1, n_blocks=9)
        obs_b, _ = _three_state_synthetic_data(seed=2, n_blocks=12)

        _, means_a, variances_a, _ = hmm_fit(obs_a, n_states=3, seed=1)
        _, means_b, variances_b, _ = hmm_fit(obs_b, n_states=3, seed=2)

        assert np.all(np.diff(variances_a) >= 0.0)
        assert np.all(np.diff(variances_b) >= 0.0)
        # Both fits should recover the same calm -> mid -> crisis mean
        # ordering, not an arbitrary permutation.
        assert np.all(np.diff(means_a) > 0.0)
        assert np.all(np.diff(means_b) > 0.0)
        for i in range(3):
            assert means_a[i] == pytest.approx(means_b[i], abs=0.5)


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


class TestHmmFitDecodeIntegration:
    def test_decode_and_predict_proba_work_with_relabeled_fit_output(self) -> None:
        observations, true_states = _two_state_synthetic_data()
        transition_matrix, means, variances, initial_probs = hmm_fit(
            observations, n_states=2, seed=SEED
        )

        decoded = hmm_decode(observations, transition_matrix, means, variances, initial_probs)
        assert decoded.shape == observations.shape
        assert _best_label_matched_accuracy(decoded, true_states) > 0.9

        posteriors = hmm_predict_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        assert posteriors.shape == (observations.shape[0], 2)
        assert np.allclose(posteriors.sum(axis=1), 1.0)


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


class TestHmmFilteredProba:
    def test_invalid_empty_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            hmm_filtered_proba(
                np.array([]),
                transition_matrix=np.eye(2),
                means=np.array([-2.0, 2.0]),
                variances=np.array([0.1, 0.1]),
                initial_probs=np.array([0.5, 0.5]),
            )

    def test_output_shape_and_rows_sum_to_one(self) -> None:
        observations, _ = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        filtered = hmm_filtered_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        assert filtered.shape == (observations.shape[0], 2)
        assert np.allclose(filtered.sum(axis=1), 1.0)

    def test_high_confidence_matches_true_state(self) -> None:
        observations, true_states = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        filtered = hmm_filtered_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        predicted_states = np.argmax(filtered, axis=1)
        accuracy = _best_label_matched_accuracy(predicted_states, true_states)
        assert accuracy > 0.9

    def test_row_t_is_unaffected_by_observations_after_t(self) -> None:
        # The defining causal property: truncating the sequence to
        # observations[:t+1] and taking the last filtered row must give the
        # same answer as computing on the full sequence and reading row t --
        # a smoother (hmm_predict_proba) would NOT have this property, since
        # its row t depends on everything after t too.
        observations, _ = _two_state_synthetic_data()
        transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])
        means = np.array([-2.0, 2.0])
        variances = np.array([0.1, 0.1])
        initial_probs = np.array([0.5, 0.5])

        full = hmm_filtered_proba(observations, transition_matrix, means, variances, initial_probs)
        t = 30
        truncated = hmm_filtered_proba(
            observations[: t + 1], transition_matrix, means, variances, initial_probs
        )
        np.testing.assert_allclose(truncated[-1], full[t], atol=1e-10)

    def test_differs_from_smoothed_posteriors_mid_sequence(self) -> None:
        # The two functions' defining difference. Uses noisier, closer-
        # together states than _two_state_synthetic_data (whose 6-sigma
        # separation lets a single observation resolve the state almost
        # certainly on its own, leaving no room for filtered vs. smoothed
        # to visibly differ) so that context genuinely matters.
        rng = np.random.default_rng(SEED)
        n_blocks, block_size = 6, 50
        true_states = np.array(
            [i % 2 for i in range(n_blocks) for _ in range(block_size)], dtype=np.int64
        )
        means = np.array([-0.3, 0.3])
        variances = np.array([1.0, 1.0])
        observations = rng.normal(loc=means[true_states], scale=1.0)
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        initial_probs = np.array([0.5, 0.5])

        filtered = hmm_filtered_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        smoothed = hmm_predict_proba(
            observations, transition_matrix, means, variances, initial_probs
        )
        # Right at a regime switch (block boundary), the filter hasn't seen
        # the switch happen yet but the smoother has -- their answers must
        # differ measurably there.
        switch_point = 50
        assert not np.allclose(filtered[switch_point], smoothed[switch_point], atol=1e-2)
        # At the very last observation there is no future left to use, so
        # the two must agree exactly there.
        np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-8)
