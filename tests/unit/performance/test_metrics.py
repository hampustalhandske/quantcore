"""Tests for performance metrics: Sharpe/Sortino/Calmar/Information ratios, drawdown, VaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quantcore.performance.metrics import (
    _cagr,
    _drawdown_series,
    calmar_ratio,
    component_var,
    hit_rate,
    information_ratio,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


class TestSharpeRatio:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            sharpe_ratio(np.array([]))

    def test_matches_hand_computed_formula(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, 0.0])
        rf = 0.001
        periods = 12
        mean = returns.mean()
        std = returns.std(ddof=1)
        expected = (mean - rf) / std * np.sqrt(periods)
        actual = sharpe_ratio(returns, risk_free_rate=rf, periods_per_year=periods)
        assert actual == pytest.approx(expected, rel=1e-9)


class TestSortinoRatio:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            sortino_ratio(np.array([]))

    def test_no_negative_excess_returns_is_inf(self) -> None:
        returns = np.array([0.01, 0.02, 0.0, 0.03])
        actual = sortino_ratio(returns, risk_free_rate=0.0)
        assert np.isinf(actual)

    def test_matches_hand_computed_formula(self) -> None:
        returns = np.array([0.02, -0.01, 0.01, -0.03])
        rf = 0.0
        periods = 12
        excess = returns - rf
        downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
        expected = (returns.mean() - rf) / downside * np.sqrt(periods)
        actual = sortino_ratio(returns, risk_free_rate=rf, periods_per_year=periods)
        assert actual == pytest.approx(expected, rel=1e-9)


class TestDrawdownSeries:
    def test_max_matches_maximum_drawdown(self) -> None:
        returns = np.array([0.05, -0.10, 0.02, -0.03, 0.08, -0.15, 0.01])
        assert _drawdown_series(returns).max() == pytest.approx(maximum_drawdown(returns))

    def test_zero_at_new_highs(self) -> None:
        returns = np.array([0.05, 0.03, 0.02])  # monotonically increasing
        dd = _drawdown_series(returns)
        assert np.all(dd == 0.0)

    def test_peak_is_seeded_with_starting_capital(self) -> None:
        returns = np.array([-0.10, -0.10, 0.50])
        # cumulative: 0.90, 0.81, 1.215; peak (seeded with 1.0): 1.0, 1.0, 1.215
        np.testing.assert_allclose(_drawdown_series(returns), [0.10, 0.19, 0.0])

    def test_matches_hand_computed_path(self) -> None:
        returns = np.array([0.10, -0.05, -0.05])
        # cumulative: 1.10, 1.045, 0.99275
        # peak:       1.10, 1.10,  1.10
        # dd:         0.0,  0.05,  0.0975 (approximately)
        dd = _drawdown_series(returns)
        cumulative = np.cumprod(1.0 + returns)
        expected = (np.maximum.accumulate(cumulative) - cumulative) / np.maximum.accumulate(
            cumulative
        )
        np.testing.assert_allclose(dd, expected)


class TestCalmarRatio:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            calmar_ratio(np.array([]))

    def test_zero_drawdown_is_inf(self) -> None:
        returns = np.full(10, 0.01)
        actual = calmar_ratio(returns)
        assert np.isinf(actual)

    def test_default_uses_cagr(self) -> None:
        returns = np.array([0.05, -0.10, 0.02, 0.03])
        periods = 12
        mdd = maximum_drawdown(returns)
        expected = _cagr(returns, periods) / abs(mdd)
        actual = calmar_ratio(returns, periods_per_year=periods)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_arithmetic_return_method_matches_old_formula(self) -> None:
        # Regression test for the prior default, still available explicitly.
        returns = np.array([0.05, -0.10, 0.02, 0.03])
        periods = 12
        old_annualized_return = returns.mean() * periods
        mdd = maximum_drawdown(returns)
        expected = old_annualized_return / abs(mdd)
        actual = calmar_ratio(returns, periods_per_year=periods, return_method="arithmetic")
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_cagr_and_arithmetic_methods_differ_for_volatile_series(self) -> None:
        returns = np.array([0.20, -0.20, 0.20, -0.20, 0.20, -0.15])
        cagr_calmar = calmar_ratio(returns, return_method="cagr")
        arithmetic_calmar = calmar_ratio(returns, return_method="arithmetic")
        assert cagr_calmar != pytest.approx(arithmetic_calmar)


class TestInformationRatio:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            information_ratio(np.array([]), np.array([]))
        with pytest.raises(ValueError):
            information_ratio(np.array([0.01, 0.02]), np.array([0.01]))

    def test_matches_hand_computed_formula(self) -> None:
        returns = np.array([0.02, 0.01, -0.01, 0.03])
        benchmark = np.array([0.01, 0.015, -0.02, 0.02])
        periods = 12
        diff = returns - benchmark
        expected = diff.mean() / diff.std(ddof=1) * np.sqrt(periods)
        actual = information_ratio(returns, benchmark, periods_per_year=periods)
        assert actual == pytest.approx(expected, rel=1e-9)


class TestMaximumDrawdown:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            maximum_drawdown(np.array([]))

    def test_monotonically_increasing_returns_is_zero(self) -> None:
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        actual = maximum_drawdown(returns)
        assert actual == pytest.approx(0.0, abs=1e-12)

    def test_known_drawdown_value(self) -> None:
        returns = np.array([0.10, -0.20, 0.05])
        cum = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cum)
        expected = np.max((peak - cum) / peak)
        actual = maximum_drawdown(returns)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_first_period_loss_is_a_drawdown_from_starting_capital(self) -> None:
        # The wealth path starts at 1.0 before the first return; a loss on
        # day one is a drawdown from that starting capital, not a new high.
        assert maximum_drawdown(np.array([-0.5])) == pytest.approx(0.5)

    def test_losses_before_first_new_high_measured_from_starting_capital(self) -> None:
        # 1 - 0.9 * 0.9 = 0.19, not the 0.10 measured from the already
        # reduced first value.
        assert maximum_drawdown(np.array([-0.1, -0.1, 0.5])) == pytest.approx(0.19)


class TestHitRate:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            hit_rate(np.array([]))

    def test_invalid_zero_policy_raises(self) -> None:
        with pytest.raises(ValueError):
            hit_rate(np.array([0.01, -0.01]), zero_policy="bogus")  # type: ignore[arg-type]

    def test_known_fraction(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
        actual = hit_rate(returns)
        assert actual == pytest.approx(2.0 / 5.0, abs=1e-12)

    def test_default_matches_loss_policy(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
        assert hit_rate(returns) == pytest.approx(hit_rate(returns, zero_policy="loss"))

    def test_exclude_policy_drops_zero_periods(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0, 0.0])
        actual = hit_rate(returns, zero_policy="exclude")
        assert actual == pytest.approx(2.0 / 4.0, abs=1e-12)

    def test_exclude_policy_all_zero_is_nan(self) -> None:
        returns = np.zeros(5)
        assert np.isnan(hit_rate(returns, zero_policy="exclude"))

    def test_win_policy_counts_zero_as_a_win(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
        actual = hit_rate(returns, zero_policy="win")
        assert actual == pytest.approx(3.0 / 5.0, abs=1e-12)


class TestProfitFactor:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            profit_factor(np.array([]))

    def test_no_losses_is_inf(self) -> None:
        returns = np.array([0.0, 0.01, 0.02])
        actual = profit_factor(returns)
        assert np.isinf(actual)

    def test_known_ratio(self) -> None:
        returns = np.array([0.02, -0.01, 0.03, -0.02])
        expected = (0.02 + 0.03) / abs(-0.01 - 0.02)
        actual = profit_factor(returns)
        assert actual == pytest.approx(expected, rel=1e-9)


class TestComponentVar:
    def test_invalid_inputs(self) -> None:
        weights = np.array([0.5, 0.5])
        cov = np.eye(2)
        with pytest.raises(ValueError):
            component_var(np.array([0.5, 0.3, 0.2]), cov, 0.99)
        with pytest.raises(ValueError):
            component_var(weights, cov, 0.0)
        with pytest.raises(ValueError):
            component_var(weights, cov, 1.0)

    def test_components_sum_to_portfolio_var(self) -> None:
        weights = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        confidence_level = 0.99
        z_alpha = norm.ppf(confidence_level)
        portfolio_var = np.sqrt(weights @ cov @ weights) * z_alpha

        components = component_var(weights, cov, confidence_level)
        assert components.shape == (2,)
        assert components.sum() == pytest.approx(portfolio_var, abs=1e-8)

    def test_all_zero_weights_returns_zero_vector_without_warning(self) -> None:
        weights = np.zeros(3)
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.02]])
        with np.errstate(invalid="raise", divide="raise"):
            components = component_var(weights, cov, 0.99)
        assert components.shape == (3,)
        assert np.array_equal(components, np.zeros(3))
