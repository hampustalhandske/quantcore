"""Tests for the drawdown, payoff and tail ratios in `quantcore.performance.ratios`."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from quantcore.performance import (
    common_sense_ratio,
    cpc_index,
    gain_to_pain_ratio,
    outlier_loss_ratio,
    outlier_win_ratio,
    profit_factor,
    risk_of_ruin,
    serenity_index,
    ulcer_index,
    ulcer_performance_index,
)

Ratio = Callable[[npt.NDArray[np.float64]], float]

ALL_RATIOS: tuple[Ratio, ...] = (
    ulcer_index,
    ulcer_performance_index,
    serenity_index,
    gain_to_pain_ratio,
    common_sense_ratio,
    cpc_index,
    outlier_win_ratio,
    outlier_loss_ratio,
    risk_of_ruin,
)
# The three ratios built on the Ulcer Index divide by `n - 1`.
NEEDS_TWO_OBSERVATIONS: tuple[Ratio, ...] = (ulcer_index, ulcer_performance_index, serenity_index)
ACCEPTS_ONE_OBSERVATION: tuple[Ratio, ...] = (
    gain_to_pain_ratio,
    common_sense_ratio,
    cpc_index,
    outlier_win_ratio,
    outlier_loss_ratio,
    risk_of_ruin,
)

# Wealth 1.1, 0.99 against peaks 1.1, 1.1: drawdowns 0, 0.1.
UP_THEN_DOWN = np.array([0.1, -0.1])

# Wealth 1.2, 0.6, 1.5, 1.125, 1.2375, 1.60875 against peaks 1.2, 1.2, 1.5, 1.5, 1.5, 1.60875:
# drawdowns 0, 0.5, 0, 0.25, 0.175, 0, whose squares sum to 0.343125. Gains sum to 2.1 over four
# periods, losses to 0.75 over two. Sorted returns -0.5, -0.25, 0.1, 0.2, 0.3, 1.5 give the
# linearly interpolated quantiles Q(0.95) = 1.2, Q(0.05) = -0.4375, Q(0.99) = 1.44 and
# Q(0.01) = -0.4875.
MIXED = np.array([0.2, -0.5, 1.5, -0.25, 0.1, 0.3])

ALL_POSITIVE = np.array([0.01, 0.02, 0.03, 0.04])
ALL_NEGATIVE = np.array([-0.01, -0.02, -0.03, -0.04])
ALL_ZERO = np.zeros(6)
ONE_LOSS_THEN_FLAT = np.concatenate(([-0.05], np.zeros(39)))


class TestInputContract:
    @pytest.mark.parametrize("ratio", ALL_RATIOS)
    def test_empty_returns_raise(self, ratio: Ratio) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ratio(np.array([]))

    @pytest.mark.parametrize("ratio", ALL_RATIOS)
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_returns_raise(self, ratio: Ratio, bad_value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            ratio(np.array([0.01, bad_value, -0.02]))

    @pytest.mark.parametrize("ratio", ALL_RATIOS)
    def test_two_dimensional_returns_raise(self, ratio: Ratio) -> None:
        with pytest.raises(ValueError, match="1-D"):
            ratio(np.array([[0.01, -0.02], [0.03, -0.01]]))

    @pytest.mark.parametrize("ratio", NEEDS_TWO_OBSERVATIONS)
    def test_single_observation_raises_where_the_divisor_is_n_minus_one(self, ratio: Ratio) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            ratio(np.array([-0.01]))

    @pytest.mark.parametrize("ratio", ACCEPTS_ONE_OBSERVATION)
    def test_single_observation_is_accepted_elsewhere(self, ratio: Ratio) -> None:
        assert isinstance(ratio(np.array([-0.01])), float)

    @pytest.mark.parametrize("ratio", ALL_RATIOS)
    def test_result_is_a_python_float(self, ratio: Ratio) -> None:
        assert type(ratio(MIXED)) is float

    @pytest.mark.parametrize("ratio", ALL_RATIOS)
    def test_input_is_not_modified(self, ratio: Ratio) -> None:
        returns = MIXED.copy()
        ratio(returns)
        np.testing.assert_array_equal(returns, MIXED)


class TestUlcerIndex:
    def test_known_values(self) -> None:
        assert ulcer_index(UP_THEN_DOWN) == pytest.approx(0.1, rel=1e-12)
        assert ulcer_index(MIXED) == pytest.approx(math.sqrt(0.343125 / 5.0), rel=1e-12)

    def test_divides_by_n_minus_one_not_n(self) -> None:
        drawdowns = np.array([0.0, 0.5, 0.0, 0.25, 0.175, 0.0])
        root_mean_square = math.sqrt(float(np.mean(drawdowns**2)))
        assert ulcer_index(MIXED) == pytest.approx(
            root_mean_square * math.sqrt(6.0 / 5.0), rel=1e-12
        )

    def test_is_a_fraction_not_a_percentage(self) -> None:
        assert ulcer_index(np.array([-0.5, 0.0])) == pytest.approx(math.sqrt(0.5), rel=1e-12)

    def test_first_period_loss_is_measured_from_starting_capital(self) -> None:
        assert ulcer_index(np.array([-0.1, 0.0])) == pytest.approx(math.sqrt(2.0 * 0.01), rel=1e-12)

    def test_never_under_water_is_zero(self) -> None:
        assert ulcer_index(ALL_POSITIVE) == 0.0
        assert ulcer_index(ALL_ZERO) == 0.0

    def test_total_loss_keeps_the_drawdown_at_one(self) -> None:
        assert ulcer_index(np.array([0.05, -1.0, 0.02, 0.03])) == pytest.approx(1.0, rel=1e-12)

    def test_return_below_minus_one_gives_drawdowns_above_one(self) -> None:
        # Wealth 1.05, -0.525, -0.5355, -0.551565 against a peak of 1.05.
        drawdowns = np.array([0.0, 1.5, 1.51, 1.5253])
        expected = math.sqrt(float(np.sum(drawdowns**2)) / 3.0)
        assert ulcer_index(np.array([0.05, -1.5, 0.02, 0.03])) == pytest.approx(expected, rel=1e-12)


class TestUlcerPerformanceIndex:
    def test_known_values(self) -> None:
        assert ulcer_performance_index(UP_THEN_DOWN) == pytest.approx(-0.1, rel=1e-12)
        assert ulcer_performance_index(MIXED) == pytest.approx(
            0.60875 / math.sqrt(0.343125 / 5.0), rel=1e-12
        )

    def test_numerator_is_the_total_compounded_return_not_annualised(self) -> None:
        returns = np.array([0.1, -0.2, 0.3])
        total_return = 1.1 * 0.8 * 1.3 - 1.0
        assert ulcer_performance_index(returns) == pytest.approx(
            total_return / ulcer_index(returns), rel=1e-12
        )

    def test_never_under_water_is_inf(self) -> None:
        assert ulcer_performance_index(ALL_POSITIVE) == math.inf
        assert ulcer_performance_index(ALL_ZERO) == math.inf

    def test_total_loss(self) -> None:
        assert ulcer_performance_index(np.array([0.05, -1.0, 0.02, 0.03])) == pytest.approx(
            -1.0, rel=1e-12
        )


class TestSerenityIndex:
    def test_known_value(self) -> None:
        # Drawdowns 0, 0.5, 0, 0.25, 0.175, 0 have mean 0.154167 and sample standard deviation
        # 0.200260, so the Gaussian 95% threshold is 0.154167 + 1.644854 * 0.200260 = 0.483566.
        # Only the 0.5 drawdown lies beyond it, so the conditional drawdown at risk is 0.5. The
        # returns have sample variance 0.47975, the Ulcer Index is sqrt(0.068625), and the returns
        # sum to 1.35: 1.35 / (sqrt(0.068625) * 0.5 / sqrt(0.47975)).
        expected = 2.7 * math.sqrt(0.47975 / 0.068625)
        assert serenity_index(MIXED) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(7.1388799232, rel=1e-10)

    def test_threshold_is_used_when_no_drawdown_lies_beyond_it(self) -> None:
        # Drawdowns 0, 0.1: mean 0.05, sample standard deviation 0.1 / sqrt(2). The threshold
        # 0.05 + 1.644854 * 0.070711 = 0.166309 exceeds both, so it stands in for the tail mean.
        returns = np.array([0.1, -0.2, 0.1])
        wealth = np.cumprod(1.0 + returns)
        drawdowns = (np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth)
        threshold = float(np.mean(drawdowns)) + 1.6448536269514722 * float(
            np.std(drawdowns, ddof=1)
        )
        assert threshold > float(np.max(drawdowns))
        pitfall = threshold / float(np.std(returns, ddof=1))
        expected = float(np.sum(returns)) / (ulcer_index(returns) * pitfall)
        assert serenity_index(returns) == pytest.approx(expected, rel=1e-9)

    def test_zero_sum_of_returns_is_zero(self) -> None:
        assert serenity_index(UP_THEN_DOWN) == 0.0

    def test_numerator_is_the_sum_of_returns_not_the_compounded_return(self) -> None:
        # Sum 0.0 but compounded return -0.25: the index is 0, not negative.
        assert serenity_index(np.array([0.5, -0.5])) == 0.0

    def test_never_under_water_is_nan(self) -> None:
        assert math.isnan(serenity_index(ALL_POSITIVE))
        assert math.isnan(serenity_index(ALL_ZERO))

    def test_constant_returns_are_nan(self) -> None:
        assert math.isnan(serenity_index(np.full(30, -0.01)))
        assert math.isnan(serenity_index(np.full(252, 0.07)))

    def test_constant_drawdown_is_nan(self) -> None:
        assert math.isnan(serenity_index(ONE_LOSS_THEN_FLAT))

    def test_zero_denominator_from_an_overflowing_volatility_is_nan(self) -> None:
        # The squared deviations of these returns exceed the float64 range, so the volatility is
        # infinite, the pitfall is zero, and so is its product with the Ulcer Index.
        with pytest.warns(RuntimeWarning, match="overflow"):
            value = serenity_index(np.array([-2.3e-16, 1e300]))
        assert math.isnan(value)

    def test_all_negative_is_finite_and_negative(self) -> None:
        value = serenity_index(ALL_NEGATIVE)
        assert math.isfinite(value)
        assert value < 0.0


class TestGainToPainRatio:
    def test_known_values(self) -> None:
        assert gain_to_pain_ratio(UP_THEN_DOWN) == 0.0
        assert gain_to_pain_ratio(MIXED) == pytest.approx(1.35 / 0.75, rel=1e-12)

    def test_equals_profit_factor_minus_one(self) -> None:
        returns = np.random.default_rng(7).normal(0.001, 0.02, 500)
        assert gain_to_pain_ratio(returns) == pytest.approx(profit_factor(returns) - 1.0, rel=1e-12)

    def test_no_losses_is_inf(self) -> None:
        assert gain_to_pain_ratio(ALL_POSITIVE) == math.inf
        assert gain_to_pain_ratio(ALL_ZERO) == math.inf

    def test_no_gains_is_minus_one(self) -> None:
        assert gain_to_pain_ratio(ALL_NEGATIVE) == pytest.approx(-1.0, rel=1e-12)


class TestCommonSenseRatio:
    def test_known_values(self) -> None:
        assert common_sense_ratio(UP_THEN_DOWN) == pytest.approx(1.0, rel=1e-12)
        assert common_sense_ratio(MIXED) == pytest.approx(2.8 * 1.2 / 0.4375, rel=1e-12)

    def test_no_losses_with_positive_lower_tail_is_inf(self) -> None:
        assert common_sense_ratio(ALL_POSITIVE) == math.inf

    def test_no_gains_is_zero(self) -> None:
        assert common_sense_ratio(ALL_NEGATIVE) == 0.0

    def test_zero_lower_tail_quantile_is_nan(self) -> None:
        assert math.isnan(common_sense_ratio(ALL_ZERO))
        assert math.isnan(common_sense_ratio(ONE_LOSS_THEN_FLAT))
        assert math.isnan(common_sense_ratio(np.concatenate(([0.05], np.zeros(39)))))


class TestCpcIndex:
    def test_known_values(self) -> None:
        assert cpc_index(UP_THEN_DOWN) == pytest.approx(0.5, rel=1e-12)
        assert cpc_index(MIXED) == pytest.approx(2.8 * (4.0 / 6.0) * 1.4, rel=1e-12)

    def test_win_rate_ignores_zero_return_periods(self) -> None:
        with_flat_periods = np.array([0.2, 0.0, -0.5, 1.5, 0.0, -0.25, 0.1, 0.3, 0.0])
        assert cpc_index(with_flat_periods) == pytest.approx(cpc_index(MIXED), rel=1e-12)

    def test_no_wins_or_no_losses_is_nan(self) -> None:
        assert math.isnan(cpc_index(ALL_POSITIVE))
        assert math.isnan(cpc_index(ALL_NEGATIVE))
        assert math.isnan(cpc_index(ALL_ZERO))


class TestOutlierWinRatio:
    def test_known_values(self) -> None:
        assert outlier_win_ratio(UP_THEN_DOWN) == pytest.approx(0.98, rel=1e-12)
        assert outlier_win_ratio(MIXED) == pytest.approx(1.44 / 0.525, rel=1e-12)

    def test_zero_returns_count_in_the_denominator_mean(self) -> None:
        # Sorted 0, 0, 0.1, 0.3: Q(0.99) = 0.1 + 0.97 * 0.2 = 0.294; mean of the four is 0.1.
        assert outlier_win_ratio(np.array([0.1, 0.0, 0.3, 0.0])) == pytest.approx(2.94, rel=1e-12)

    def test_no_non_negative_return_is_nan(self) -> None:
        assert math.isnan(outlier_win_ratio(ALL_NEGATIVE))

    def test_zero_mean_of_non_negative_returns_is_nan(self) -> None:
        assert math.isnan(outlier_win_ratio(ALL_ZERO))
        assert math.isnan(outlier_win_ratio(np.array([0.0, -0.01, 0.0])))


class TestOutlierLossRatio:
    def test_known_values(self) -> None:
        assert outlier_loss_ratio(UP_THEN_DOWN) == pytest.approx(0.98, rel=1e-12)
        assert outlier_loss_ratio(MIXED) == pytest.approx(-0.4875 / -0.375, rel=1e-12)

    def test_no_losses_is_nan(self) -> None:
        assert math.isnan(outlier_loss_ratio(ALL_POSITIVE))
        assert math.isnan(outlier_loss_ratio(ALL_ZERO))

    def test_non_negative_lower_quantile_is_not_positive(self) -> None:
        # One loss among 39 flat periods: Q(0.01) = -0.05 * (1 - 0.39) = -0.0305 over a mean loss
        # of -0.05.
        assert outlier_loss_ratio(ONE_LOSS_THEN_FLAT) == pytest.approx(0.61, rel=1e-12)
        # With 200 flat periods the 1% quantile is zero, and so is the ratio.
        assert outlier_loss_ratio(np.concatenate(([-0.05], np.zeros(200)))) == 0.0


class TestRiskOfRuin:
    def test_known_values(self) -> None:
        assert risk_of_ruin(UP_THEN_DOWN) == pytest.approx(1.0 / 9.0, rel=1e-12)
        assert risk_of_ruin(MIXED) == pytest.approx(0.2**6, rel=1e-12)

    def test_exponent_counts_zero_periods_but_the_win_rate_does_not(self) -> None:
        # Win rate 1/2 from the two non-zero periods; exponent 4 from all four.
        assert risk_of_ruin(np.array([0.1, 0.0, -0.1, 0.0])) == pytest.approx(
            (0.5 / 1.5) ** 4, rel=1e-12
        )

    def test_no_winning_period_is_one(self) -> None:
        assert risk_of_ruin(ALL_NEGATIVE) == 1.0
        assert risk_of_ruin(ALL_ZERO) == 1.0

    def test_every_non_zero_period_winning_is_zero(self) -> None:
        assert risk_of_ruin(ALL_POSITIVE) == 0.0
        assert risk_of_ruin(np.array([0.1, 0.0, 0.2])) == 0.0

    def test_stays_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(11)
        for n in (1, 2, 17, 400):
            value = risk_of_ruin(rng.normal(0.0, 0.01, n))
            assert 0.0 <= value <= 1.0
