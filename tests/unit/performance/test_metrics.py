"""Tests for performance metrics: Sharpe/Sortino/Calmar/Information ratios, drawdown, VaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quantcore.performance.metrics import (
    calmar_ratio,
    component_var,
    hit_rate,
    information_ratio,
    maximum_drawdown,
    profit_factor,
    rolling_sharpe,
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


class TestCalmarRatio:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            calmar_ratio(np.array([]))

    def test_zero_drawdown_is_inf(self) -> None:
        returns = np.full(10, 0.01)
        actual = calmar_ratio(returns)
        assert np.isinf(actual)

    def test_matches_hand_computed_formula(self) -> None:
        returns = np.array([0.05, -0.10, 0.02, 0.03])
        periods = 12
        annualized_return = returns.mean() * periods
        mdd = maximum_drawdown(returns)
        expected = annualized_return / abs(mdd)
        actual = calmar_ratio(returns, periods_per_year=periods)
        assert actual == pytest.approx(expected, rel=1e-9)


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


class TestRollingSharpe:
    def test_invalid_inputs(self) -> None:
        returns = np.array([0.01, 0.02, 0.03, 0.01])
        with pytest.raises(ValueError):
            rolling_sharpe(returns, window=0)
        with pytest.raises(ValueError):
            rolling_sharpe(returns, window=len(returns) + 1)

    def test_output_shape_and_leading_nans(self) -> None:
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.015])
        window = 3
        result = rolling_sharpe(returns, window=window)
        assert result.shape == (5,)
        assert np.all(np.isnan(result[: window - 1]))
        assert not np.any(np.isnan(result[window - 1 :]))


class TestHitRate:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            hit_rate(np.array([]))

    def test_known_fraction(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
        actual = hit_rate(returns)
        assert actual == pytest.approx(2.0 / 5.0, abs=1e-12)


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
