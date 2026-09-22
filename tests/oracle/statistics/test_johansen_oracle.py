"""Oracle tests for B2: `johansen_trace_test` critical values vs statsmodels.

Confirmed defect B2 (see QUANTCORE_FIX_PROMPT.md): the table quantcore used
was the no-deterministic-term case, shifted by one index, so a trace
statistic needed to exceed ~12.32 to reject at 95% where the correct
threshold (for the unrestricted-constant model actually fitted) is 3.84 —
the test almost never rejected. The fix rebuilds the tables from MacKinnon,
Haug & Michelis (1999) for all three deterministic cases and 90/95/99%, adds
the maximum-eigenvalue statistic, and returns eigenvalues.

Oracle: `statsmodels.tsa.vector_ar.vecm.coint_johansen`, `det_order=0`
(statsmodels' "constant term", confirmed here to be the same unrestricted-
constant model quantcore fits — both include `np.ones(n_obs)` as an
unrestricted regressor in the short-run VAR-in-differences, not restricted
to lie in the cointegration space).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from quantcore.statistics.cointegration import (
    johansen_max_eigenvalue_test,
    johansen_trace_test,
)

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _random_walks(n: int, k: int, seed: int, correlation: float = 0.0) -> np.ndarray:
    """k independent (or correlated) random walks, shape (n, k)."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, size=(n, k))
    if correlation:
        innovations[:, 1:] += correlation * innovations[:, [0]]
    return np.cumsum(innovations, axis=0)


class TestCriticalValuesMatchStatsmodelsTable:
    """Same table (MacKinnon, Haug & Michelis 1999), same det_order='constant'."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_trace_critical_values(self, k: int) -> None:
        data = _random_walks(300, k, seed=RNG_SEED)
        _, crit_values, _ = johansen_trace_test(data, n_lags=1, deterministic="c")
        sm_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        np.testing.assert_allclose(crit_values, sm_result.cvt[:, 1], atol=1e-3)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_max_eigenvalue_critical_values(self, k: int) -> None:
        data = _random_walks(300, k, seed=RNG_SEED)
        _, crit_values, _, _ = johansen_max_eigenvalue_test(data, n_lags=1, deterministic="c")
        sm_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        np.testing.assert_allclose(crit_values, sm_result.cvm[:, 1], atol=1e-3)


class TestStatisticsMatchStatsmodels:
    """The trace/max-eig statistics and eigenvalues themselves, independent
    of the critical-value table."""

    def test_trace_statistic_matches(self) -> None:
        data = _random_walks(300, 3, seed=RNG_SEED, correlation=0.3)
        trace_stats, _, _ = johansen_trace_test(data, n_lags=1)
        sm_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        np.testing.assert_allclose(trace_stats, sm_result.lr1, rtol=1e-4)

    def test_max_eig_statistic_matches(self) -> None:
        data = _random_walks(300, 3, seed=RNG_SEED, correlation=0.3)
        max_eig_stats, _, _, _ = johansen_max_eigenvalue_test(data, n_lags=1)
        sm_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        np.testing.assert_allclose(max_eig_stats, sm_result.lr2, rtol=1e-4)

    def test_eigenvalues_match(self) -> None:
        data = _random_walks(300, 3, seed=RNG_SEED, correlation=0.3)
        _, _, eigvals, _ = johansen_max_eigenvalue_test(data, n_lags=1)
        sm_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        sm_eig = np.real(sm_result.eig)
        np.testing.assert_allclose(np.sort(eigvals), np.sort(sm_eig), rtol=1e-4)


class TestBriefNumbersReproduced:
    def test_unrestricted_constant_95pct_values(self) -> None:
        # The exact figures the finding was written against.
        data = _random_walks(50, 6, seed=RNG_SEED)
        _, crit_values, _ = johansen_trace_test(data, n_lags=1, deterministic="c")
        expected = np.array([95.7542, 69.8189, 47.8545, 29.7961, 15.4943, 3.8415])
        np.testing.assert_allclose(crit_values, expected)

    def test_old_table_would_have_required_stat_above_twelve_not_3_84(self) -> None:
        # Direct demonstration of the old bug's practical effect: the wrong
        # (shifted, no-deterministic-term) table's k-r=1 value was 12.3212;
        # the correct unrestricted-constant value is 3.8415. A trace
        # statistic between those two thresholds is a real rejection at 95%
        # that the old table would have missed entirely.
        data = _random_walks(50, 6, seed=RNG_SEED)
        _, crit_values, _ = johansen_trace_test(data, n_lags=1, deterministic="c")
        assert crit_values[-1] == pytest.approx(3.8415)
        assert crit_values[-1] < 12.3212


class TestMonteCarloSize:
    """Under independent random walks (rank 0 by construction), the k-r=k
    trace test's rejection rate at 5% must be close to 5%. The old
    (index-shifted) table's threshold was ~3.2x too high, so its rejection
    rate under this null was far below 5% (it almost never rejected — the
    finding's stated failure mode).

    n_obs=1000 (loosened from a smaller default): the asymptotic Johansen
    critical values carry real finite-sample size distortion that only
    vanishes as n grows — confirmed by running this exact Monte Carlo
    through `statsmodels.tsa.vector_ar.vecm.coint_johansen` directly, which
    over-rejects by the same amount at n_obs=200 (~16% vs. the 5% nominal)
    and converges to ~5% by n_obs=1000. That is a documented property of the
    asymptotic test itself, not a quantcore defect, so the test uses a
    sample size where it doesn't confound the size check with the known
    finite-sample bias.
    """

    def test_trace_test_rank_zero_rejection_rate_near_nominal_five_percent(self) -> None:
        n_trials = 200
        n_obs = 1000
        k = 3
        alpha = 0.05
        rng = np.random.default_rng(RNG_SEED + 200)
        rejections = 0
        for _ in range(n_trials):
            data = np.cumsum(rng.normal(0.0, 1.0, size=(n_obs, k)), axis=0)
            trace_stats, crit_values, _ = johansen_trace_test(data, n_lags=1)
            # H0: rank <= 0, i.e. no cointegration at all — the full-k-r test.
            if trace_stats[0] > crit_values[0]:
                rejections += 1
        observed_rate = rejections / n_trials

        se = np.sqrt(alpha * (1 - alpha) / n_trials)
        margin = 2.81 * se  # z for a two-sided 99.5% interval
        assert abs(observed_rate - alpha) < margin, (
            f"observed rejection rate {observed_rate:.3f} is not within "
            f"{margin:.3f} of the nominal {alpha}; expected ~5% under a true null "
            f"of no cointegration"
        )
