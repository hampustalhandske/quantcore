"""Gaussian Hidden Markov Model regime detection: fit, decode, smoothed and filtered posteriors.

`hmm_filtered_proba` (causal) is the only posterior safe in a walk-forward backtest or live
signal; `hmm_predict_proba` conditions on the whole sequence, future included. `hmm_fit`'s
log-likelihood is checked through an independent forward algorithm and `hmmlearn`; its parameters
are compared after relabelling states by ascending emission variance. `hmm_decode` and
`hmm_predict_proba` are checked with the same parameters handed to `hmmlearn`.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM

import quantcore.regime.hmm as qc_hmm
from conformance.compare import ArrayComparator, ConformanceTarget, ScalarComparator

_MIN_PROB = 1e-300

type _HmmParams = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


def _log_emission(
    observations: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    diff = observations[:, None] - means[None, :]
    result = -0.5 * (
        np.log(2.0 * np.pi) + np.log(variances)[None, :] + diff**2 / variances[None, :]
    )
    return np.asarray(result, dtype=np.float64)


def _logsumexp_rows(log_x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m = np.max(log_x, axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(log_x - m), axis=1, keepdims=True))


def _forward_log_alpha(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hand forward-algorithm log-alpha recursion (Baum et al. 1970).

    Written independently of quantcore's own (private) forward pass, so it can serve as an
    oracle for `hmm_filtered_proba` and for cross-checking `hmm_fit`'s log-likelihood.
    """
    log_b = _log_emission(observations, means, variances)
    log_a = np.log(np.clip(transition_matrix, _MIN_PROB, None))
    log_pi = np.log(np.clip(initial_probs, _MIN_PROB, None))
    n, k = observations.shape[0], means.shape[0]
    log_alpha = np.empty((n, k), dtype=np.float64)
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, n):
        prev = log_alpha[t - 1][:, None] + log_a
        m = np.max(prev, axis=0)
        log_alpha[t] = m + np.log(np.sum(np.exp(prev - m[None, :]), axis=0)) + log_b[t]
    return log_alpha


def _forward_log_likelihood(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> float:
    """Total log-likelihood `log P(y_1..y_T)` via the hand forward algorithm above."""
    log_alpha = _forward_log_alpha(observations, transition_matrix, means, variances, initial_probs)
    return float(_logsumexp_rows(log_alpha[-1:]).item())


def _forward_filtered_proba(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hand causal (filtered) posterior `P(s_t=k | y_1..y_t)`, from the forward pass alone."""
    log_alpha = _forward_log_alpha(observations, transition_matrix, means, variances, initial_probs)
    log_filtered = log_alpha - _logsumexp_rows(log_alpha)
    return np.exp(log_filtered)


def _quantile_bucket_init(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mirror `quantcore.regime.hmm._hmm_fit`'s quantile-bucket mean/variance initialisation.

    Used to seed the independent `hmmlearn` fit from the *same* starting point as quantcore's own
    EM, so the `hmm_fit_*` parameter comparisons are between two optimizers converging from
    comparable starting conditions, not confounded by one starting far from the other.
    """
    rng = np.random.default_rng(seed)
    sorted_obs = np.sort(observations)
    buckets = np.array_split(sorted_obs, n_states)
    global_var = max(float(np.var(observations)), 1e-8)
    means = np.array([float(np.mean(b)) for b in buckets], dtype=np.float64)
    variances = np.array(
        [max(float(np.var(b)), 1e-8) if b.size > 1 else global_var for b in buckets],
        dtype=np.float64,
    )
    means += rng.normal(0.0, 1e-6, size=n_states)
    return means, variances


def _relabel_ascending_variance(
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> _HmmParams:
    order = np.argsort(variances)
    return (
        transition_matrix[np.ix_(order, order)],
        means[order],
        variances[order],
        initial_probs[order],
    )


# --- hmm_fit: params (independent hmmlearn EM, comparably seeded) + log-likelihood (hand forward)


def _hmm_fit_params_quantcore(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> _HmmParams:
    transition_matrix, means, variances, initial_probs = qc_hmm.hmm_fit(
        observations, n_states, seed=seed
    )
    return (
        np.asarray(transition_matrix, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(initial_probs, dtype=np.float64),
    )


def _hmm_fit_params_reference(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> _HmmParams:
    means0, variances0 = _quantile_bucket_init(observations, n_states, seed)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        tol=1e-6,
        init_params="",
        params="stmc",
        random_state=seed,
    )
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
    model.means_ = means0.reshape(-1, 1)
    model.covars_ = variances0.reshape(-1, 1)
    model.fit(observations.reshape(-1, 1))
    transition_matrix = np.asarray(model.transmat_, dtype=np.float64)
    means = np.asarray(model.means_, dtype=np.float64).reshape(-1)
    variances = np.asarray(model.covars_, dtype=np.float64).reshape(-1)
    initial_probs = np.asarray(model.startprob_, dtype=np.float64)
    return _relabel_ascending_variance(transition_matrix, means, variances, initial_probs)


# Four targets, one per fitted array: `TupleComparator` takes scalar sub-results only, so the
# arrays cannot be judged as one tuple (the same split as `ols_coefficients`/`ols_r_squared`).
_HMM_FIT_PARAMS_INDEX = {"transition_matrix": 0, "means": 1, "variances": 2, "initial_probs": 3}


def _hmm_fit_component_quantcore(
    key: str,
) -> Callable[[npt.NDArray[np.float64], int, int], npt.NDArray[np.float64]]:
    index = _HMM_FIT_PARAMS_INDEX[key]

    def _adapter(
        observations: npt.NDArray[np.float64], n_states: int, seed: int
    ) -> npt.NDArray[np.float64]:
        return _hmm_fit_params_quantcore(observations, n_states, seed)[index]

    return _adapter


def _hmm_fit_component_reference(
    key: str,
) -> Callable[[npt.NDArray[np.float64], int, int], npt.NDArray[np.float64]]:
    index = _HMM_FIT_PARAMS_INDEX[key]

    def _adapter(
        observations: npt.NDArray[np.float64], n_states: int, seed: int
    ) -> npt.NDArray[np.float64]:
        return _hmm_fit_params_reference(observations, n_states, seed)[index]

    return _adapter


_HMM_FIT_TRANSITION_MATRIX = ConformanceTarget(
    name="hmm_fit_transition_matrix",
    tier=2,
    oracle="hmmlearn.GaussianHMM (independent EM fit, comparably seeded)",
    citation="Baum, Petrie, Soules & Weiss (1970), 'A Maximization Technique...' Annals of "
    "Mathematical Statistics, 41(1), 164-171.",
    compare=ArrayComparator(),
    quantcore=_hmm_fit_component_quantcore("transition_matrix"),
    reference=_hmm_fit_component_reference("transition_matrix"),
)

_HMM_FIT_MEANS = ConformanceTarget(
    name="hmm_fit_means",
    tier=2,
    oracle="hmmlearn.GaussianHMM (independent EM fit, comparably seeded)",
    citation="Baum, Petrie, Soules & Weiss (1970). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_hmm_fit_component_quantcore("means"),
    reference=_hmm_fit_component_reference("means"),
)

_HMM_FIT_VARIANCES = ConformanceTarget(
    name="hmm_fit_variances",
    tier=2,
    oracle="hmmlearn.GaussianHMM (independent EM fit, comparably seeded)",
    citation="Baum, Petrie, Soules & Weiss (1970). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_hmm_fit_component_quantcore("variances"),
    reference=_hmm_fit_component_reference("variances"),
)

_HMM_FIT_INITIAL_PROBS = ConformanceTarget(
    name="hmm_fit_initial_probs",
    tier=2,
    oracle="hmmlearn.GaussianHMM (independent EM fit, comparably seeded)",
    citation="Baum, Petrie, Soules & Weiss (1970). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_hmm_fit_component_quantcore("initial_probs"),
    reference=_hmm_fit_component_reference("initial_probs"),
)


def _hmm_fit_log_likelihood_quantcore(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> float:
    params = qc_hmm.hmm_fit(observations, n_states, seed=seed)
    return _forward_log_likelihood(observations, *params)


def _hmm_fit_log_likelihood_reference(
    observations: npt.NDArray[np.float64], n_states: int, seed: int
) -> float:
    """`hmmlearn.GaussianHMM.score` evaluated on quantcore's *own* fitted parameters.

    Cross-validates quantcore's log-likelihood arithmetic against an independent forward-algorithm
    implementation (`hmmlearn`'s), on the exact same parameters.
    """
    transition_matrix, means, variances, initial_probs = qc_hmm.hmm_fit(
        observations, n_states, seed=seed
    )
    model = GaussianHMM(n_components=n_states, covariance_type="diag")
    model.startprob_ = initial_probs
    model.transmat_ = transition_matrix
    model.means_ = means.reshape(-1, 1)
    model.covars_ = variances.reshape(-1, 1)
    return float(model.score(observations.reshape(-1, 1)))


_HMM_FIT_LOG_LIKELIHOOD = ConformanceTarget(
    name="hmm_fit_log_likelihood",
    tier=1,
    oracle="hmmlearn.GaussianHMM.score, loaded with quantcore's own fitted parameters",
    citation="Baum, Petrie, Soules & Weiss (1970). Full citation in docs/REFERENCES.md.",
    compare=ScalarComparator(),
    quantcore=_hmm_fit_log_likelihood_quantcore,
    reference=_hmm_fit_log_likelihood_reference,
)


# --- hmm_decode: same (not re-fit) params fed to hmmlearn's Viterbi --------------------------


def _hmm_decode_quantcore(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    states = qc_hmm.hmm_decode(observations, transition_matrix, means, variances, initial_probs)
    return np.asarray(states, dtype=np.float64)


def _hmm_decode_reference(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_states = means.shape[0]
    model = GaussianHMM(n_components=n_states, covariance_type="diag")
    model.startprob_ = initial_probs
    model.transmat_ = transition_matrix
    model.means_ = means.reshape(-1, 1)
    model.covars_ = variances.reshape(-1, 1)
    states = model.predict(observations.reshape(-1, 1))
    return np.asarray(states, dtype=np.float64)


_HMM_DECODE = ConformanceTarget(
    name="hmm_decode",
    tier=1,
    oracle="hmmlearn.GaussianHMM.predict (Viterbi), given quantcore's own parameters directly",
    citation="Viterbi, A. (1967), 'Error Bounds for Convolutional Codes...' IEEE Transactions on "
    "Information Theory, 13(2), 260-269.",
    compare=ArrayComparator(),
    quantcore=_hmm_decode_quantcore,
    reference=_hmm_decode_reference,
)


# --- hmm_predict_proba: smoother, same injected params -----------------------------------------


def _hmm_predict_proba_quantcore(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_hmm.hmm_predict_proba(observations, transition_matrix, means, variances, initial_probs),
        dtype=np.float64,
    )


def _hmm_predict_proba_reference(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n_states = means.shape[0]
    model = GaussianHMM(n_components=n_states, covariance_type="diag")
    model.startprob_ = initial_probs
    model.transmat_ = transition_matrix
    model.means_ = means.reshape(-1, 1)
    model.covars_ = variances.reshape(-1, 1)
    proba: npt.NDArray[np.float64] = model.predict_proba(observations.reshape(-1, 1))
    return proba


_HMM_PREDICT_PROBA = ConformanceTarget(
    name="hmm_predict_proba",
    tier=2,
    oracle="hmmlearn.GaussianHMM.predict_proba, given quantcore's own parameters directly",
    citation="Baum, Petrie, Soules & Weiss (1970). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_hmm_predict_proba_quantcore,
    reference=_hmm_predict_proba_reference,
)


# --- hmm_filtered_proba: causal, hand forward-algorithm oracle -----------------------------------


def _hmm_filtered_proba_quantcore(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_hmm.hmm_filtered_proba(observations, transition_matrix, means, variances, initial_probs),
        dtype=np.float64,
    )


def _hmm_filtered_proba_reference(
    observations: npt.NDArray[np.float64],
    transition_matrix: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    initial_probs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return _forward_filtered_proba(observations, transition_matrix, means, variances, initial_probs)


_HMM_FILTERED_PROBA = ConformanceTarget(
    name="hmm_filtered_proba",
    tier=1,
    oracle="hand forward-algorithm implementation (Hamilton 1989)",
    citation="Hamilton, J.D. (1989), 'A New Approach to the Economic Analysis of Nonstationary "
    "Time Series and the Business Cycle.' Econometrica, 57(2), 357-384.",
    compare=ArrayComparator(),
    quantcore=_hmm_filtered_proba_quantcore,
    reference=_hmm_filtered_proba_reference,
)


# --- select_hmm_n_states: no independent oracle (quantcore adapter only) ------------------------


def _select_hmm_n_states_quantcore(
    observations: npt.NDArray[np.float64], max_states: int, criterion: str
) -> int:
    return int(qc_hmm.select_hmm_n_states(observations, max_states, criterion))


_SELECT_HMM_N_STATES = ConformanceTarget(
    name="select_hmm_n_states",
    tier=2,
    oracle=(
        "none (no independent library implements this selection rule); hand/invariant tests only"
    ),
    citation="Schwarz, G. (1978), 'Estimating the Dimension of a Model.' Annals of Statistics, "
    "6(2), 461-464. (BIC.) Akaike, H. (1974), 'A New Look at the Statistical Model Identification.'"
    " IEEE Transactions on Automatic Control, 19(6), 716-723. (AIC.)",
    compare=ScalarComparator(),
    quantcore=_select_hmm_n_states_quantcore,
)
