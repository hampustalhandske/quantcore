"""Oracle test: `hmm_predict_proba` against `hmmlearn.hmm.GaussianHMM.predict_proba`.

`hmm_predict_proba` computes smoothed posteriors P(s_t=k | y_1..y_T) via
forward-backward; hmmlearn's `predict_proba` runs the same algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

import hmmlearn.hmm as hl

from quantcore.regime.hmm import hmm_predict_proba

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
