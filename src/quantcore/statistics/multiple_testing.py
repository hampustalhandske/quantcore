"""Multiple-testing diagnostics for a family of backtested trials.

`effective_n_eigen` estimates how many independent trials a `T x N` family of return series
amounts to, as the participation ratio `(sum lambda)^2 / sum lambda^2` of the eigenvalues of
the trials' correlation matrix: `N` for uncorrelated trials, `1` for identical ones.

`cscv_pbo` is the probability of backtest overfitting by combinatorially symmetric
cross-validation. The rows are cut into `S` equal blocks; for every way of choosing `S/2` blocks
as the in-sample half, the trial that scores best in-sample is ranked among all trials on the
out-of-sample half. Its relative mid-rank `omega = rank / (N + 1)` (ties count half a rank)
gives the logit `lambda = ln(omega / (1 - omega))`, and `PBO = P(lambda <= 0)`: how often the
in-sample winner is at or below the median out-of-sample.

References:
    Bailey, D.H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio Management,
    40(5), 94-107 (the effective number of trials; the participation-ratio form is not verified
    against the text). Bailey, D.H., Borwein, J.M., Lopez de Prado, M. and Zhu, Q.J. (2016), "The
    Probability of Backtest Overfitting." Journal of Computational Finance, 20(4), 39-69. See
    docs/REFERENCES.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

_OMEGA_CLIP = 1e-12


def _column_major(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """`returns` as a column-major float64 array.

    The last bit of `np.corrcoef` and of per-column reductions depends on memory layout; a fixed
    column-major layout makes the results independent of the caller's layout and equal to those
    computed from a pandas DataFrame's values, which are column-major.
    """
    return np.asfortranarray(returns, dtype=np.float64)


def _check_returns_matrix(
    returns: npt.NDArray[np.float64], min_rows: int, min_rows_name: str
) -> None:
    """Raise ValueError unless `returns` is finite, 2-D, with 2+ columns and `min_rows`+ rows."""
    if returns.ndim != 2:
        raise ValueError(f"returns must be a 2-D array (T x N), got {returns.ndim} dimensions")
    t_obs, n_trials = returns.shape
    if n_trials < 2:
        raise ValueError("returns must have at least 2 columns (trials)")
    if t_obs < min_rows:
        raise ValueError(f"returns must have at least {min_rows_name} rows")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")


def effective_n_eigen(returns: npt.NDArray[np.float64]) -> float:
    """Effective number of independent trials: eigenvalue participation ratio of the correlation.

    `N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2` over the eigenvalues of the `N x N` Pearson
    correlation matrix of the columns, with negative rounding-level eigenvalues set to 0, and
    1.0 when every eigenvalue is 0. A column whose computed standard deviation is exactly 0 has
    NaN correlations, which count as 0 (its diagonal entry included), so it adds nothing to
    `N_eff`. A constant column whose mean is not exact in floating point (0.002 repeated, say)
    keeps a spread near 1e-19 after centring; its correlations with the other columns are then
    near 0 and its diagonal is 1, so it counts as one uncorrelated trial.

    Args:
        returns: `T x N` finite returns, one column per trial, at least 2 rows and 2 columns.

    Returns:
        The effective number of trials, in `[1, N]`.

    Raises:
        ValueError: `returns` is not 2-D, has fewer than 2 columns or rows, or is not finite.
    """
    _check_returns_matrix(returns, 2, "2")
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(_column_major(returns), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    sum_lambda = float(eigenvalues.sum())
    sum_lambda_sq = float(np.sum(eigenvalues**2))
    return sum_lambda**2 / sum_lambda_sq if sum_lambda_sq > 0.0 else 1.0


def _per_period_sharpe(values: npt.NDArray[np.float64]) -> float:
    """Per-period Sharpe ratio `mean / std(ddof=1)`; 0.0 when the standard deviation is 0."""
    sigma = float(values.std(ddof=1))
    return float(values.mean()) / sigma if sigma != 0.0 else 0.0


@dataclass(frozen=True)
class CscvPbo:
    """Result of `cscv_pbo`.

    Attributes:
        pbo: Share of evaluated train/test combinations whose in-sample-best trial has an
            out-of-sample logit `<= 0`, in `[0, 1]`.
        n_splits: Number of row blocks `S`.
        n_combinations: Number of combinations evaluated: `C(S, S/2)`, or `max_combinations`
            when subsampled.
        n_rows_dropped: Trailing rows left out because `T` is not a multiple of `S`.
        subsampled: Whether a random subset of the combinations was evaluated.
    """

    pbo: float
    n_splits: int
    n_combinations: int
    n_rows_dropped: int
    subsampled: bool


def cscv_pbo(
    returns: npt.NDArray[np.float64],
    *,
    n_splits: int = 16,
    evaluate: Callable[[npt.NDArray[np.float64]], float] = _per_period_sharpe,
    max_combinations: int | None = None,
    rng: np.random.Generator | None = None,
) -> CscvPbo:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The first `S * floor(T / S)` rows are cut into `S` consecutive blocks of equal size. Each
    combination of `S/2` blocks, in `itertools.combinations` order, is the in-sample half and
    its complement the out-of-sample half; both halves keep the blocks in time order. The trial
    with the highest in-sample `evaluate` (the first on a tie) is ranked out-of-sample as
    `1 + #(others below) + 0.5 * #(others equal)`; `omega = rank / (N + 1)`, clipped to
    `[1e-12, 1 - 1e-12]`, and the logit is `ln(omega / (1 - omega))`.

    Args:
        returns: `T x N` finite per-period returns, one column per trial, `N >= 2`, `T >= S`.
        n_splits: Number of row blocks `S`; even and at least 2. `C(S, S/2)` combinations are
            evaluated (12,870 at `S = 16`).
        evaluate: Score of one trial's 1-D returns over one half; higher is better. Defaults to
            the per-period Sharpe ratio `mean / std(ddof=1)`, 0.0 when the computed standard
            deviation is exactly 0. A constant column whose mean is not exact in floating point
            has a computed standard deviation near 1e-19 instead, and a Sharpe ratio near 1e16
            in magnitude.
        max_combinations: When given and smaller than `C(S, S/2)`, evaluate only this many
            combinations, drawn without replacement by `rng.choice` and kept in their original
            order. None evaluates every combination.
        rng: Generator for the subsample; required when subsampling happens.

    Returns:
        `CscvPbo` with the PBO, `S`, the number of combinations evaluated, the number of rows
        dropped and whether the combinations were subsampled.

    Raises:
        ValueError: `n_splits` is not even and at least 2; `returns` is not 2-D, has fewer than
            2 columns or fewer than `n_splits` rows, or is not finite; or subsampling is needed
            and `rng` is None.
    """
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be even and at least 2")
    _check_returns_matrix(returns, n_splits, "n_splits")
    t_obs, n_trials = returns.shape
    block_size = t_obs // n_splits
    used_rows = block_size * n_splits

    values = _column_major(returns)[:used_rows, :]
    blocks = [values[i * block_size : (i + 1) * block_size, :] for i in range(n_splits)]

    train_sets = list(combinations(range(n_splits), n_splits // 2))
    total_combinations = len(train_sets)
    subsampled = False
    if max_combinations is not None and total_combinations > max_combinations:
        subsampled = True
        if rng is None:
            raise ValueError("rng is required when subsampling combinations")
        chosen = rng.choice(total_combinations, size=max_combinations, replace=False)
        train_sets = [train_sets[i] for i in sorted(chosen.tolist())]

    all_blocks = frozenset(range(n_splits))
    logits = np.empty(len(train_sets), dtype=np.float64)
    for k, train_block_idx in enumerate(train_sets):
        train_set = frozenset(train_block_idx)
        test_set = all_blocks - train_set
        train_rows = np.concatenate([blocks[b] for b in sorted(train_set)], axis=0)
        test_rows = np.concatenate([blocks[b] for b in sorted(test_set)], axis=0)

        is_metrics = np.array(
            [evaluate(train_rows[:, j]) for j in range(n_trials)], dtype=np.float64
        )
        oos_metrics = np.array(
            [evaluate(test_rows[:, j]) for j in range(n_trials)], dtype=np.float64
        )

        best_trial = int(np.argmax(is_metrics))
        best_oos = oos_metrics[best_trial]
        others = np.delete(oos_metrics, best_trial)
        oos_rank = 1 + int(np.sum(others < best_oos)) + 0.5 * int(np.sum(others == best_oos))
        omega = oos_rank / (n_trials + 1)
        omega = min(max(omega, _OMEGA_CLIP), 1.0 - _OMEGA_CLIP)
        logits[k] = math.log(omega / (1.0 - omega))

    return CscvPbo(
        pbo=float(np.mean(logits <= 0.0)),
        n_splits=n_splits,
        n_combinations=len(train_sets),
        n_rows_dropped=t_obs - used_rows,
        subsampled=subsampled,
    )
