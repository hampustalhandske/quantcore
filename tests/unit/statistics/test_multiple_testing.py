"""Unit tests for `cscv_pbo` and `effective_n_eigen`.

The pinned values were computed by edgelab's `cscv` and `effective_n_eigen` on the inputs the
helpers below build. A PBO is a ratio of small integers and is compared exactly; an effective
N goes through an eigendecomposition and is compared at relative 1e-12.
"""

import math

import numpy as np
import pytest

from quantcore.statistics import CscvPbo, cscv_pbo, effective_n_eigen


def _rounded_mean(values: np.ndarray) -> float:
    """Mean in percent rounded to one decimal: a coarse metric with many ties."""
    return float(np.round(values.mean() * 100.0, 1))


# --- cscv_pbo: values pinned from edgelab ---------------------------------------------------------


def _gauss(seed: int, t: int, n: int, mean: float = 0.0005) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 0.01, (t, n))


def _constant_column(seed: int, t: int) -> np.ndarray:
    x = np.random.default_rng(seed).normal(0.0, 0.01, (t, 4))
    x[:, 1] = 0.001
    return x


def _integer_returns(seed: int, t: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(-2, 3, (t, 6)).astype(float) * 0.01


# (input, kwargs, rng seed or None, (pbo, n_combinations, n_rows_dropped, subsampled))
EDGELAB_PBO_CASES = [
    (_gauss(24, 163, 12), {"n_splits": 16}, None, (0.5963480963480964, 12870, 3, False)),
    (_gauss(14, 80, 5), {"n_splits": 8}, None, (0.5142857142857142, 70, 0, False)),
    (_gauss(12, 43, 12), {"n_splits": 4}, None, (0.5, 6, 3, False)),
    (_gauss(1, 20, 2), {"n_splits": 2}, None, (1.0, 2, 0, False)),
    (
        _integer_returns(101, 49),
        {"n_splits": 8, "evaluate": _rounded_mean},
        None,
        (0.5285714285714286, 70, 1, False),
    ),
    (
        _gauss(401, 97, 7, mean=0.0003),
        {"n_splits": 12, "max_combinations": 37},
        901,
        (0.32432432432432434, 37, 1, True),
    ),
    (
        _gauss(400, 128, 7, mean=0.0003),
        {"n_splits": 16, "max_combinations": 500},
        900,
        (0.056, 500, 0, True),
    ),
]


@pytest.mark.parametrize(("x", "kwargs", "rng_seed", "expected"), EDGELAB_PBO_CASES)
def test_cscv_pbo_matches_edgelab_values(
    x: np.ndarray, kwargs: dict, rng_seed: int | None, expected: tuple
) -> None:
    rng = None if rng_seed is None else np.random.default_rng(rng_seed)
    result = cscv_pbo(x, rng=rng, **kwargs)
    assert isinstance(result, CscvPbo)
    assert (result.pbo, result.n_combinations, result.n_rows_dropped, result.subsampled) == expected
    assert result.n_splits == kwargs["n_splits"]


# --- cscv_pbo: small cases worked by hand ---------------------------------------------------------


def test_cscv_pbo_reversal_is_fully_overfit() -> None:
    # Trial 0 wins the first half and loses the second; trial 1 the opposite. With S = 2 each
    # in-sample winner ranks last out-of-sample: omega = 1/3, logit < 0, so PBO = 1.
    x = np.array([[0.02, -0.01], [0.03, -0.02], [-0.01, 0.02], [-0.02, 0.03]])
    assert cscv_pbo(x, n_splits=2).pbo == 1.0


def test_cscv_pbo_persistent_winner_is_not_overfit() -> None:
    # Trial 0 beats trial 1 in both halves: rank 2 of 2, omega = 2/3, logit > 0, so PBO = 0.
    x = np.array([[0.03, -0.01], [0.02, 0.00], [0.04, -0.02], [0.01, 0.01]])
    result = cscv_pbo(x, n_splits=2)
    assert result.pbo == 0.0
    assert result.n_combinations == 2


def test_cscv_pbo_tie_is_a_median_rank_and_counts_as_overfit() -> None:
    # Identical trials tie out-of-sample: rank 1.5 of 2, omega = 1/2, logit = 0 <= 0.
    column = np.array([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
    x = np.column_stack([column, column])
    assert cscv_pbo(x, n_splits=2).pbo == 1.0


def test_cscv_pbo_evaluates_every_combination_in_order() -> None:
    seen: list[float] = []

    def first_value(values: np.ndarray) -> float:
        seen.append(float(values[0]))
        return float(values.sum())

    x = np.arange(12.0).reshape(6, 2)
    result = cscv_pbo(x, n_splits=6, evaluate=first_value)
    assert result.n_combinations == math.comb(6, 3)
    # Each combination evaluates 2 trials on the train half, then 2 on the test half.
    assert len(seen) == 4 * math.comb(6, 3)
    # The first combination trains on blocks (0, 1, 2): rows 0-2, then tests on rows 3-5.
    assert seen[:4] == [0.0, 1.0, 6.0, 7.0]


def test_cscv_pbo_drops_trailing_rows() -> None:
    x = _gauss(5, 23, 3)
    result = cscv_pbo(x, n_splits=4)
    trimmed = cscv_pbo(x[:20], n_splits=4)
    assert result.n_rows_dropped == 3
    assert trimmed.n_rows_dropped == 0
    assert result.pbo == trimmed.pbo


def test_cscv_pbo_subsample_is_deterministic_under_the_generator() -> None:
    x = _gauss(7, 160, 6)
    first = cscv_pbo(x, max_combinations=200, rng=np.random.default_rng(3))
    second = cscv_pbo(x, max_combinations=200, rng=np.random.default_rng(3))
    assert first == second
    assert first.subsampled and first.n_combinations == 200


def test_cscv_pbo_does_not_subsample_at_or_above_the_total() -> None:
    x = _gauss(8, 40, 4)
    full = cscv_pbo(x, n_splits=6)
    capped = cscv_pbo(x, n_splits=6, max_combinations=20)
    assert capped == full
    assert not capped.subsampled and capped.n_combinations == 20


def test_cscv_pbo_does_not_depend_on_memory_layout() -> None:
    x = _gauss(9, 96, 8)
    assert cscv_pbo(np.ascontiguousarray(x), n_splits=8) == cscv_pbo(
        np.asfortranarray(x), n_splits=8
    )


@pytest.mark.parametrize(
    ("x", "kwargs", "match"),
    [
        (np.zeros((20, 3)), {"n_splits": 3}, "even"),
        (np.zeros((20, 3)), {"n_splits": 0}, "even"),
        (np.zeros(20), {"n_splits": 2}, "2-D"),
        (np.zeros((20, 1)), {"n_splits": 2}, "2 columns"),
        (np.zeros((7, 3)), {"n_splits": 8}, "n_splits rows"),
        (np.full((20, 3), np.nan), {"n_splits": 2}, "finite"),
        (np.full((20, 3), np.inf), {"n_splits": 2}, "finite"),
        (np.zeros((20, 3)), {"n_splits": 4, "max_combinations": 2}, "rng"),
    ],
)
def test_cscv_pbo_invalid_inputs_raise(x: np.ndarray, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cscv_pbo(x, **kwargs)


# --- effective_n_eigen ----------------------------------------------------------------------------


def _near_identical(seed: int, eps: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, (200, 1))
    return base + eps * rng.normal(0.0, 0.01, (200, 6))


EDGELAB_EFFECTIVE_N_CASES = [
    (_gauss(703, 30, 25, mean=0.0), 13.084703878060116),
    (_gauss(702, 500, 10, mean=0.0), 9.77986974110435),
    (_gauss(700, 50, 3, mean=0.0), 2.87274504599837),
    (_near_identical(801, 1e-6), 1.0000000000015312),
]


@pytest.mark.parametrize(("x", "expected"), EDGELAB_EFFECTIVE_N_CASES)
def test_effective_n_eigen_matches_edgelab_values(x: np.ndarray, expected: float) -> None:
    assert effective_n_eigen(x) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("value", [2.0**-9, 0.001, 0.002])
def test_cscv_pbo_scores_any_constant_column_zero(value: float) -> None:
    # A constant column scores a Sharpe ratio of 0 whatever its value: 0.001 and 0.002 have an
    # inexact mean, so their computed standard deviation is about 1e-19, not 0.
    x = _constant_column(301, 41)
    x[:, 1] = value
    exact = x.copy()
    exact[:, 1] = 2.0**-9
    assert cscv_pbo(x, n_splits=4) == cscv_pbo(exact, n_splits=4)
    assert cscv_pbo(x, n_splits=4).pbo == pytest.approx(1.0 / 3.0, rel=1e-15)


@pytest.mark.parametrize("value", [0.5, 0.001, 0.002])
def test_effective_n_eigen_any_constant_column_adds_nothing(value: float) -> None:
    x = np.random.default_rng(830).normal(0.0, 0.01, (80, 6))
    x[:, 2] = value
    # The constant column adds nothing: the result equals that of the other five columns.
    assert effective_n_eigen(x) == pytest.approx(
        effective_n_eigen(np.delete(x, 2, axis=1)), rel=1e-13
    )
    assert effective_n_eigen(x) == pytest.approx(4.847776643565792, rel=1e-12)


def test_effective_n_eigen_exactly_constant_column_adds_nothing() -> None:
    x = np.random.default_rng(830).normal(0.0, 0.01, (80, 6))
    x[:, 2] = 0.5
    assert effective_n_eigen(x) == pytest.approx(4.847776643565792, rel=1e-12)


def test_effective_n_eigen_orthogonal_trials_count_fully() -> None:
    hadamard = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float)
    x = np.kron(hadamard, np.ones((3, 1)))[:, 1:]
    assert effective_n_eigen(x) == pytest.approx(3.0, rel=1e-14)


def test_effective_n_eigen_identical_trials_count_once() -> None:
    column = np.random.default_rng(810).normal(0.0, 0.01, (100, 1))
    assert effective_n_eigen(np.repeat(column, 5, axis=1)) == pytest.approx(1.0, rel=1e-14)


def test_effective_n_eigen_all_constant_is_one() -> None:
    # Every correlation is undefined and counts as 0, so the matrix is 0 and the result is 1.
    assert effective_n_eigen(np.ones((10, 4))) == 1.0


def test_effective_n_eigen_two_trials() -> None:
    # Two trials with correlation rho have eigenvalues 1 +- rho: N_eff = 4 / (2 + 2 rho^2).
    x = np.random.default_rng(12).normal(size=(400, 2))
    rho = float(np.corrcoef(x, rowvar=False)[0, 1])
    assert effective_n_eigen(x) == pytest.approx(2.0 / (1.0 + rho**2), rel=1e-13)


@pytest.mark.parametrize(
    ("x", "match"),
    [
        (np.zeros(10), "2-D"),
        (np.zeros((10, 2, 2)), "2-D"),
        (np.zeros((10, 1)), "2 columns"),
        (np.zeros((1, 3)), "2 rows"),
        (np.array([[0.1, np.nan], [0.2, 0.3], [0.0, 0.1]]), "finite"),
        (np.array([[0.1, -np.inf], [0.2, 0.3], [0.0, 0.1]]), "finite"),
    ],
)
def test_effective_n_eigen_invalid_inputs_raise(x: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        effective_n_eigen(x)
