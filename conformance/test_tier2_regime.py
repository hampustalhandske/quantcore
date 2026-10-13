"""Differential tests for Gaussian HMM regime detection against `hmmlearn`.

`hmm_fit_log_likelihood`, `hmm_decode`, `hmm_filtered_proba` are Tier-1; the rest of this family
is Tier-2. `select_hmm_n_states` has no independent oracle (see its section in
`conformance/targets/regime.py`) and is exercised only by `tests/unit/regime/test_hmm.py`.
"""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

import quantcore.regime.hmm as qc_hmm
from conformance.compare import registry
from conformance.harness import check
from conformance.targets.regime import _forward_log_likelihood

_HMM_FIT_TRANSITION_MATRIX = registry()["hmm_fit_transition_matrix"]
_HMM_FIT_MEANS = registry()["hmm_fit_means"]
_HMM_FIT_VARIANCES = registry()["hmm_fit_variances"]
_HMM_FIT_INITIAL_PROBS = registry()["hmm_fit_initial_probs"]
_HMM_FIT_LOG_LIKELIHOOD = registry()["hmm_fit_log_likelihood"]
_HMM_DECODE = registry()["hmm_decode"]
_HMM_PREDICT_PROBA = registry()["hmm_predict_proba"]
_HMM_FILTERED_PROBA = registry()["hmm_filtered_proba"]

# Declared coverage. `hmm_fit*`'s family/bucket names describe the regime-switching generator
# below, not `conformance.generators.RETURN_FAMILIES` (HMM inputs are observation sequences of a
# discrete-state process, not returns).
_HMM_FAMILIES: tuple[str, ...] = ("two_state_separated", "three_state_separated")
_HMM_BUCKETS: tuple[str, ...] = ("medium", "long")

HMM_FIT_MIN_LEN = 60
HMM_FIT_FAMILIES = _HMM_FAMILIES
HMM_FIT_BUCKETS = _HMM_BUCKETS

HMM_DECODE_MIN_LEN = 60
HMM_DECODE_FAMILIES = _HMM_FAMILIES
HMM_DECODE_BUCKETS = _HMM_BUCKETS

HMM_PREDICT_PROBA_MIN_LEN = 60
HMM_PREDICT_PROBA_FAMILIES = _HMM_FAMILIES
HMM_PREDICT_PROBA_BUCKETS = _HMM_BUCKETS

HMM_FILTERED_PROBA_MIN_LEN = 60
HMM_FILTERED_PROBA_FAMILIES = _HMM_FAMILIES
HMM_FILTERED_PROBA_BUCKETS = _HMM_BUCKETS


def _length_bucket(n: int) -> str:
    return "medium" if n <= 500 else "long"


@st.composite
def regime_switching_observations(
    draw: st.DrawFn, min_len: int = 60, max_len: int = 800
) -> tuple[npt.NDArray[np.float64], str, int, int]:
    """Draw a well-separated 2- or 3-regime Gaussian Markov-switching observation sequence.

    Returns `(observations, family, seed, n_states)`. States are well separated (means several
    standard deviations apart) so `hmm_decode`'s Viterbi path is unambiguous and label-permutation
    concerns for `hmm_fit` (the oracle relabels states by ascending emission variance) stay small.
    """
    n_states = draw(st.sampled_from([2, 3]))
    family = "two_state_separated" if n_states == 2 else "three_state_separated"
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    means = np.arange(n_states, dtype=np.float64) * 4.0
    sigma = 0.5
    state = 0
    states = np.empty(n, dtype=np.int64)
    for t in range(n):
        if t > 0 and rng.random() < 0.05:
            state = int(rng.integers(0, n_states))
        states[t] = state
    observations = rng.normal(means[states], sigma).astype(np.float64)
    return observations, family, seed, n_states


@settings(max_examples=30)
@given(draw_result=regime_switching_observations())
def test_hmm_fit_matches_hmmlearn(
    draw_result: tuple[npt.NDArray[np.float64], str, int, int],
) -> None:
    observations, family, seed, n_states = draw_result
    bucket = _length_bucket(observations.shape[0])
    for fn in (
        _HMM_FIT_TRANSITION_MATRIX,
        _HMM_FIT_MEANS,
        _HMM_FIT_VARIANCES,
        _HMM_FIT_INITIAL_PROBS,
        _HMM_FIT_LOG_LIKELIHOOD,
    ):
        check(fn, observations, n_states, seed, family=family, length_bucket=bucket, seed=seed)


@settings(max_examples=30)
@given(draw_result=regime_switching_observations())
def test_hmm_decode_matches_hmmlearn(
    draw_result: tuple[npt.NDArray[np.float64], str, int, int],
) -> None:
    observations, family, seed, n_states = draw_result
    bucket = _length_bucket(observations.shape[0])
    transition_matrix, means, variances, initial_probs, _ll = _fit(observations, n_states, seed)
    check(
        _HMM_DECODE,
        observations,
        transition_matrix,
        means,
        variances,
        initial_probs,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )


@settings(max_examples=30)
@given(draw_result=regime_switching_observations())
def test_hmm_predict_proba_matches_hmmlearn(
    draw_result: tuple[npt.NDArray[np.float64], str, int, int],
) -> None:
    observations, family, seed, n_states = draw_result
    bucket = _length_bucket(observations.shape[0])
    transition_matrix, means, variances, initial_probs, _ll = _fit(observations, n_states, seed)
    check(
        _HMM_PREDICT_PROBA,
        observations,
        transition_matrix,
        means,
        variances,
        initial_probs,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )


@settings(max_examples=30)
@given(draw_result=regime_switching_observations())
def test_hmm_filtered_proba_matches_hand_forward_algorithm(
    draw_result: tuple[npt.NDArray[np.float64], str, int, int],
) -> None:
    observations, family, seed, n_states = draw_result
    bucket = _length_bucket(observations.shape[0])
    transition_matrix, means, variances, initial_probs, _ll = _fit(observations, n_states, seed)
    check(
        _HMM_FILTERED_PROBA,
        observations,
        transition_matrix,
        means,
        variances,
        initial_probs,
        family=family,
        length_bucket=bucket,
        seed=seed,
    )


def _fit(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """Fit once directly via quantcore, reused across decode/predict_proba/filtered_proba cases.

    Calls `quantcore.regime.hmm.hmm_fit` directly. Only used to build a shared parameter set for
    the downstream tests; `hmm_fit` itself is exercised by `test_hmm_fit_matches_hmmlearn`.
    """
    transition_matrix, means, variances, initial_probs = qc_hmm.hmm_fit(
        observations, n_states, seed=seed
    )
    log_likelihood = _forward_log_likelihood(
        observations, transition_matrix, means, variances, initial_probs
    )
    return transition_matrix, means, variances, initial_probs, log_likelihood


@settings(max_examples=50)
@given(draw_result=regime_switching_observations(min_len=80, max_len=300))
def test_hmm_filtered_proba_no_lookahead(
    draw_result: tuple[npt.NDArray[np.float64], str, int, int],
) -> None:
    """Property test of the look-ahead guard: perturbing the future must not change the past.

    `hmm_filtered_proba` row `t` is documented to depend only on `observations[:t+1]`. This
    perturbs every observation strictly after a cut point and asserts the filtered probabilities
    up to and including the cut point are unchanged -- the concrete guarantee a walk-forward
    backtest or live signal relies on (see `quantcore.regime.hmm.hmm_filtered_proba`).
    """
    observations, _family, seed, n_states = draw_result
    transition_matrix, means, variances, initial_probs, _ll = _fit(observations, n_states, seed)
    n = observations.shape[0]
    cut = n // 2

    baseline = qc_hmm.hmm_filtered_proba(
        observations, transition_matrix, means, variances, initial_probs
    )

    rng = np.random.default_rng(seed + 1)
    perturbed = observations.copy()
    perturbed[cut + 1 :] = rng.normal(0.0, 10.0, size=n - cut - 1)
    perturbed_filtered = qc_hmm.hmm_filtered_proba(
        perturbed, transition_matrix, means, variances, initial_probs
    )

    assert np.allclose(baseline[: cut + 1], perturbed_filtered[: cut + 1], atol=1e-12), (
        "hmm_filtered_proba's row t changed when observations strictly after t were perturbed -- "
        "a look-ahead leak"
    )
