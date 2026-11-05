"""Oracle tests for `cscv_pbo` and `effective_n_eigen`.

PBO oracle: the CSCV procedure of Bailey, Borwein, Lopez de Prado and Zhu (2016), Sec. 2.2,
written independently. The train half of each combination is a boolean row mask (no block
concatenation), every trial's Sharpe ratio is computed at once along axis 0, and the
out-of-sample rank of the in-sample winner is `scipy.stats.rankdata(method="average")` among
all N trials. A PBO is `k / C(S, S/2)` for an integer `k`, so a single combination judged
differently moves it by at least `1 / 12870`. The two agree exactly on the 64 panels here and on
400 further random panels (S in {2, 4, 6, 8, 10}, N in [2, 12]).

Effective-N oracle: the correlation matrix `C` is positive semidefinite, so clipping its
eigenvalues at 0 only removes rounding-level negatives, and `sum lambda = trace(C)` and
`sum lambda^2 = ||C||_F^2`; hence `N_eff = trace(C)^2 / ||C||_F^2` with no eigendecomposition.
`C` is built from hand-standardised columns, with a constant column's row and column set to 0
(its correlation is undefined). The closed forms are `N` for orthogonal trials and `1` for
identical ones. Constant columns use the value 2**-9, which centres to exactly 0; a constant
whose mean is inexact leaves a 1e-19 residue whose treatment depends on summation order (see
`effective_n_eigen`). Over the 600 panels here (T in [3, 600], N in [2, 40], including T < N)
and 2000 further ones (N up to 60), the largest relative difference is 1.1e-15; the test allows
1e-13, the backward error of a symmetric eigensolver, about N * eps.
"""

import math
from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.stats import rankdata

from quantcore.statistics import cscv_pbo, effective_n_eigen

pytestmark = pytest.mark.oracle

RNG_SEED = 20261105


# --- PBO oracle ---------------------------------------------------------------------------------


def _sharpe_columns(x: np.ndarray) -> np.ndarray:
    sigma = x.std(axis=0, ddof=1)
    safe = np.where(sigma == 0.0, 1.0, sigma)
    return np.where(sigma == 0.0, 0.0, x.mean(axis=0) / safe)


def _pbo_reference(x: np.ndarray, n_splits: int) -> float:
    t_obs, n_trials = x.shape
    block = t_obs // n_splits
    m = x[: block * n_splits]
    block_of_row = np.arange(m.shape[0]) // block
    overfit = 0
    total = 0
    for train in combinations(range(n_splits), n_splits // 2):
        in_train = np.isin(block_of_row, train)
        r_is = _sharpe_columns(m[in_train])
        r_oos = _sharpe_columns(m[~in_train])
        best = int(np.argmax(r_is))
        omega = rankdata(r_oos, method="average")[best] / (n_trials + 1)
        overfit += math.log(omega / (1.0 - omega)) <= 0.0
        total += 1
    return overfit / total


def _panel(kind: str, t: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0003, 0.01, (t, n))
    if kind == "skill":
        x = rng.normal(0.0, 0.01, (t, n))
        x[:, 0] += 0.003
        return x
    if kind == "duplicates":
        base = rng.normal(0.0, 0.01, (t, max(2, n // 2)))
        return np.column_stack([base, base])[:, :n]
    # 2**-9 is exact under centring, so the column's standard deviation is exactly 0.
    x = rng.normal(0.0, 0.01, (t, n))
    x[:, n // 2] = 2.0**-9
    return x


@pytest.mark.parametrize("kind", ["gaussian", "skill", "duplicates", "constant_column"])
@pytest.mark.parametrize("n_splits", [2, 4, 8, 10])
def test_cscv_pbo_matches_paper_procedure(kind: str, n_splits: int) -> None:
    rng = np.random.default_rng(RNG_SEED + n_splits)
    for n in (2, 3, 7, 16):
        t = int(rng.integers(n_splits * 3, n_splits * 30))
        x = _panel(kind, t, n, rng)
        assert cscv_pbo(x, n_splits=n_splits).pbo == _pbo_reference(x, n_splits)


def test_cscv_pbo_hand_case_three_trials() -> None:
    # S = 2, blocks of 2 rows. Per-period Sharpe on block 0: (2.12, 0.71, -0.71); on block 1:
    # (0.0 for the constant -0.01 pair, 0.71, 1.41). Train on block 0: trial 0 wins and ranks
    # 1 of 3 on block 1. Train on block 1: trial 2 wins and ranks 1 of 3 on block 0. Both
    # omegas are 1/4, both logits negative, so PBO = 1.
    x = np.array(
        [
            [0.02, 0.00, -0.01],
            [0.01, 0.01, 0.00],
            [-0.01, 0.00, 0.03],
            [-0.01, 0.01, 0.01],
        ]
    )
    assert _pbo_reference(x, 2) == 1.0
    assert cscv_pbo(x, n_splits=2).pbo == 1.0


# --- effective N oracle ---------------------------------------------------------------------------


def _effective_n_reference(x: np.ndarray) -> float:
    centred = x - x.mean(axis=0)
    norms = np.sqrt((centred**2).sum(axis=0))
    unit = np.divide(centred, norms, out=np.zeros_like(centred), where=norms > 0.0)
    corr = unit.T @ unit
    np.fill_diagonal(corr, (norms > 0.0).astype(float))
    frobenius_sq = float((corr**2).sum())
    return float(np.trace(corr)) ** 2 / frobenius_sq if frobenius_sq > 0.0 else 1.0


def _trials(kind: str, t: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0, 0.01, (t, n))
    if kind == "near_identical":
        return rng.normal(0.0, 0.01, (t, 1)) + 1e-4 * rng.normal(0.0, 0.01, (t, n))
    if kind == "equicorrelated":
        return rng.normal(size=(t, 1)) * 0.8 + rng.normal(size=(t, n)) * 0.6
    # 2**-9 is exact under centring, so the column's correlations are undefined and count as 0.
    x = rng.normal(0.0, 0.01, (t, n))
    x[:, 0] = 2.0**-9
    return x


@pytest.mark.parametrize("kind", ["gaussian", "near_identical", "equicorrelated", "constant"])
def test_effective_n_eigen_matches_trace_identity(kind: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(150):
        t = int(rng.integers(3, 600))
        n = int(rng.integers(2, 40))
        x = _trials(kind, t, n, rng)
        assert effective_n_eigen(x) == pytest.approx(_effective_n_reference(x), rel=1e-13)


@pytest.mark.parametrize("n", [2, 3, 8, 16])
def test_effective_n_eigen_orthogonal_trials_equal_n(n: int) -> None:
    # Centred orthonormal columns are exactly uncorrelated: C = I, N_eff = N.
    q, _ = np.linalg.qr(np.random.default_rng(n).normal(size=(4 * n, n + 1)))
    ones = np.ones((4 * n, 1)) / math.sqrt(4 * n)
    q = q - ones @ (ones.T @ q)
    q, _ = np.linalg.qr(q)
    x = q[:, :n]
    assert effective_n_eigen(x) == pytest.approx(n, rel=1e-12)


@pytest.mark.parametrize("n", [2, 5, 30])
def test_effective_n_eigen_identical_and_scaled_trials_equal_one(n: int) -> None:
    column = np.random.default_rng(n).normal(0.0, 0.01, (50, 1))
    x = column * np.linspace(0.5, 3.0, n) + np.linspace(-0.01, 0.01, n)
    assert effective_n_eigen(x) == pytest.approx(1.0, rel=1e-12)
