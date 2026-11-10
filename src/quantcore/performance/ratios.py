"""Drawdown, payoff and tail ratios of a periodic return series.

Every function takes periodic simple returns r_1..r_n and returns one float.
None of them annualises, subtracts a risk-free rate, or resamples: the
returns are used at the frequency they arrive in. Each computes the
definition used by the QuantStats library (`quantstats.stats`), which in
several places is not the one in the statistic's original source; every
such difference is stated in the function's docstring.

Notation:

    W_t = prod_{s<=t} (1 + r_s)             wealth, starting capital 1
    P_t = max(1, W_1, ..., W_t)             running peak, starting capital included
    D_t = (P_t - W_t) / P_t                 drawdown, a positive fraction
                                            (`_drawdown_series`)
    G   = sum of the positive returns
    L   = |sum of the negative returns|
    Q_p = p-quantile of the returns, interpolated linearly between the two
          order statistics around position p * (n - 1)

    Ulcer Index:
      UI = sqrt(sum_t D_t^2 / (n - 1))

    Ulcer Performance Index:
      UPI = (W_n - 1) / UI

    Serenity index:
      S = sum_t r_t / (UI * pitfall),  pitfall = CDaR / std(r, ddof=1)
      CDaR = mean of the drawdowns above mean(D) + z * std(D, ddof=1), or
      that threshold itself when no drawdown is above it, with
      z = -Phi^-1(1 - 0.95)

    Gain-to-pain ratio:
      GPR = sum_t r_t / L

    Common sense ratio:
      CSR = (G / L) * |Q_0.95 / Q_(1 - 0.95)|

    CPC index:
      CPC = (G / L) * win rate * payoff ratio
      win rate = #{r_t > 0} / #{r_t != 0}
      payoff ratio = mean(r_t : r_t > 0) / |mean(r_t : r_t < 0)|

    Outlier win and loss ratios:
      Q_0.99 / mean(r_t : r_t >= 0)  and  Q_0.01 / mean(r_t : r_t < 0)

    Risk of ruin:
      ((1 - win rate) / (1 + win rate)) ** n

A ratio that is undefined or infinite for its input (a zero risk
denominator, an empty category such as no loss or no gain, a constant
series) raises `quantcore.UndefinedResultError`, a `ValueError`; each
function lists its cases. No function returns `inf` or `nan`: a result that
overflows float64 when the returns are compounded or summed also raises
`UndefinedResultError`. Returns below -1 are accepted: wealth turns negative
and the drawdowns exceed 1.

References:
    Martin, P.G. and McCann, B.B. (1989), *The Investor's Guide to Fidelity
    Funds: Winning Strategies for Mutual Fund Investors*, Wiley. (Ulcer
    Index, Ulcer Performance Index.)

    Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005), "Drawdown Measure in
    Portfolio Optimization," *International Journal of Theoretical and
    Applied Finance*, 8(1), 13-58. (Conditional drawdown at risk.)

    Schwager, J.D. (2012), *Hedge Fund Market Wizards*, Wiley, Appendix A.
    (Gain-to-pain ratio.)

    Feller, W. (1968), *An Introduction to Probability Theory and Its
    Applications*, vol. 1, 3rd ed., Wiley, ch. XIV. (Gambler's ruin.)

    The Serenity index, the common sense ratio, the CPC index and the
    fixed-stake risk-of-ruin formula are cited by attribution in their
    functions' docstrings. The outlier ratios and the tail ratio have no
    published source. See docs/REFERENCES.md.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from quantcore.errors import UndefinedResultError, _finite_result
from quantcore.performance.metrics import _drawdown_series

_TAIL_QUANTILE = 0.95
_OUTLIER_WIN_QUANTILE = 0.99
_OUTLIER_LOSS_QUANTILE = 0.01
_DRAWDOWN_AT_RISK_CONFIDENCE = 0.95

# Number of standard deviations above the mean drawdown at which the Gaussian
# drawdown-at-risk threshold sits. Written as the quantile of the complement
# `1 - confidence`, which in float64 is not the same number as 0.05.
_DRAWDOWN_AT_RISK_Z = -float(norm.ppf(1.0 - _DRAWDOWN_AT_RISK_CONFIDENCE))


def _validate_returns(returns: npt.NDArray[np.float64], min_observations: int = 1) -> None:
    if returns.ndim != 1:
        raise ValueError("returns must be a 1-D array")
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if returns.size < min_observations:
        raise ValueError("returns must contain at least two observations")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values")


def ulcer_index(returns: npt.NDArray[np.float64]) -> float:
    """Ulcer Index: root of the sum of squared drawdowns over `n - 1`.

    UI = sqrt(sum_t D_t^2 / (n - 1)), with D_t the drawdown from the running
    peak of the wealth path (see `_drawdown_series`). The drawdowns are
    fractions, so a path that sits 50% below its peak throughout has an
    index near 0.5, not 50.

    The divisor is `n - 1`, the QuantStats convention. Martin's definition
    is the root of the *mean* squared drawdown (divisor `n`) with drawdowns
    quoted in percent, so this value is sqrt(n / (n - 1)) times his index
    expressed as a fraction: 2.6% higher at n = 20, 0.2% at n = 252.

    Args:
        returns: Periodic simple returns, 1-D, at least two observations.

    Returns:
        The index, never negative; 0.0 when the path is never below its
        running peak.

    Raises:
        ValueError: `returns` is not 1-D, has fewer than two observations,
            or contains NaN or infinity.
        UndefinedResultError: Compounding the returns overflows float64.

    References:
        Martin, P.G. and McCann, B.B. (1989), *The Investor's Guide to
        Fidelity Funds*, Wiley.
    """
    _validate_returns(returns, min_observations=2)
    return _finite_result(_ulcer_index(_drawdown_series(returns)), "Ulcer Index")


def _ulcer_index(drawdowns: npt.NDArray[np.float64]) -> float:
    return math.sqrt(float(np.sum(drawdowns**2)) / (drawdowns.size - 1))


def ulcer_performance_index(returns: npt.NDArray[np.float64]) -> float:
    """Ulcer Performance Index: total compounded return over the Ulcer Index.

    UPI = (prod_t (1 + r_t) - 1) / UI, with UI from `ulcer_index`.

    The numerator is the compounded return over the whole sample: it is not
    annualised and no risk-free return is subtracted (the QuantStats
    convention with a zero risk-free rate). Martin's ratio divides the
    return in excess of the risk-free return by the Ulcer Index, with
    returns quoted per year. Because the numerator here grows with the
    length of the sample, values from samples of different lengths are not
    comparable.

    Args:
        returns: Periodic simple returns, 1-D, at least two observations.

    Returns:
        The ratio; negative when the sample ends below its starting
        capital.

    Raises:
        ValueError: `returns` is not 1-D, has fewer than two observations,
            or contains NaN or infinity.
        UndefinedResultError: The Ulcer Index is 0, i.e. the path is never
            below its running peak (an all-zero series included); or the
            result overflows float64.

    References:
        Martin, P.G. and McCann, B.B. (1989), *The Investor's Guide to
        Fidelity Funds*, Wiley.
    """
    _validate_returns(returns, min_observations=2)
    return _finite_result(_ulcer_performance_index(returns), "Ulcer Performance Index")


def _ulcer_performance_index(returns: npt.NDArray[np.float64]) -> float:
    ulcer = _ulcer_index(_drawdown_series(returns))
    if ulcer == 0.0:
        raise UndefinedResultError(
            "the path is never below its running peak, so the Ulcer Index is 0; "
            "the Ulcer Performance Index is undefined"
        )
    return (float(np.prod(1.0 + returns)) - 1.0) / ulcer


def serenity_index(returns: npt.NDArray[np.float64]) -> float:
    """Serenity index: sum of returns over the Ulcer Index times a drawdown-tail penalty.

    S = sum_t r_t / (UI * pitfall), where UI is `ulcer_index` and

        pitfall = CDaR / std(r, ddof=1)

    CDaR, the conditional drawdown at risk, is the mean of the drawdowns D_t
    lying strictly above the threshold

        q = mean(D) + z * std(D, ddof=1),  z = -Phi^-1(1 - 0.95) = 1.645

    the 95% quantile of a normal distribution fitted to the drawdown
    series. When no drawdown lies above q, CDaR is q itself.

    This is the QuantStats definition. Its numerator is the plain sum of the
    returns (not compounded, not annualised, no risk-free rate subtracted),
    its volatility is per period, and its Ulcer Index divides by `n - 1`.
    Its cut-off q comes from a normal distribution fitted to the drawdowns,
    whereas Chekhlov, Uryasev and Zabarankin define the conditional
    drawdown at risk on the empirical drawdown distribution, as the mean of
    its worst `1 - alpha` share.

    Because the tail is selected by a strict comparison with q, the result
    is discontinuous in the returns: a drawdown that crosses q moves into
    or out of the tail mean.

    Args:
        returns: Periodic simple returns, 1-D, at least two observations.

    Returns:
        The index; negative when the returns sum to less than zero.

    Raises:
        ValueError: `returns` is not 1-D, has fewer than two observations,
            or contains NaN or infinity.
        UndefinedResultError: The returns have zero range (zero volatility);
            the drawdown series has zero range (the path is never below its
            peak, or drops once and stays flat, leaving no spread to fit);
            the product UI * pitfall evaluates to zero (the volatility
            overflows float64); or the result is not finite.

    References:
        The Serenity ratio is attributed to a white paper by the asset
        manager KeyQuant, which QuantStats names as its source
        (`KeyQuant_WhitePaper_APT_Part1.pdf`). Chekhlov, A., Uryasev, S. and
        Zabarankin, M. (2005), "Drawdown Measure in Portfolio Optimization,"
        *International Journal of Theoretical and Applied Finance*, 8(1),
        13-58, for the conditional drawdown at risk.
    """
    _validate_returns(returns, min_observations=2)
    return _finite_result(_serenity_index(returns), "Serenity index")


def _serenity_index(returns: npt.NDArray[np.float64]) -> float:
    # Constancy is tested on the range, not on the standard deviation: the
    # mean of a constant array is often not exactly that constant, which
    # leaves a standard deviation of rounding residue instead of zero.
    if np.ptp(returns) == 0.0:
        raise UndefinedResultError("returns have zero range; the Serenity index is undefined")
    drawdowns = _drawdown_series(returns)
    if np.ptp(drawdowns) == 0.0:
        raise UndefinedResultError(
            "the drawdown series has zero range; the Serenity index is undefined"
        )
    threshold = float(np.mean(drawdowns)) + _DRAWDOWN_AT_RISK_Z * float(np.std(drawdowns, ddof=1))
    tail = drawdowns[drawdowns > threshold]
    conditional_drawdown = float(np.mean(tail)) if tail.size > 0 else threshold
    pitfall = conditional_drawdown / float(np.std(returns, ddof=1))
    denominator = _ulcer_index(drawdowns) * pitfall
    if denominator == 0.0:
        raise UndefinedResultError(
            "the Ulcer Index times the pitfall is 0 (the volatility overflows float64); "
            "the Serenity index is undefined"
        )
    return float(np.sum(returns)) / denominator


def gain_to_pain_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Gain-to-pain ratio: sum of all returns over the absolute sum of the losses.

    GPR = sum_t r_t / |sum_{r_t < 0} r_t|, which equals `profit_factor - 1`.

    Schwager defines the ratio on monthly returns. No resampling is done
    here: the returns are used at the frequency they are given in, so a
    caller wanting Schwager's figure passes monthly returns.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        The ratio, at least -1.0 (reached when there is no gain).

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.
        UndefinedResultError: There is no negative return (an all-zero
            series included), or the result overflows float64.

    References:
        Schwager, J.D. (2012), *Hedge Fund Market Wizards*, Wiley,
        Appendix A.
    """
    _validate_returns(returns)
    return _finite_result(_gain_to_pain_ratio(returns), "gain-to-pain ratio")


def _gain_to_pain_ratio(returns: npt.NDArray[np.float64]) -> float:
    pain = abs(float(np.sum(returns[returns < 0.0])))
    if pain == 0.0:
        raise UndefinedResultError(
            "there is no negative return; the gain-to-pain ratio is undefined"
        )
    return float(np.sum(returns)) / pain


def _win_rate(returns: npt.NDArray[np.float64]) -> float:
    """Winning share of the non-zero periods; 0.0 when every period is zero."""
    nonzero = returns[returns != 0.0]
    if nonzero.size == 0:
        return 0.0
    return float(np.mean(nonzero > 0.0))


def _profit_factor(returns: npt.NDArray[np.float64]) -> float:
    """G / L, the sum of the gains over the absolute sum of the losses; L must not be 0."""
    gains = float(np.sum(returns[returns > 0.0]))
    losses = abs(float(np.sum(returns[returns < 0.0])))
    return gains / losses


def _tail_ratio(returns: npt.NDArray[np.float64]) -> float:
    """|Q_0.95 / Q_(1 - 0.95)|; the lower quantile must not be 0."""
    lower, upper = np.quantile(returns, (1.0 - _TAIL_QUANTILE, _TAIL_QUANTILE))
    return abs(float(upper) / float(lower))


def common_sense_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Common sense ratio: profit factor times tail ratio.

    CSR = (G / L) * |Q_0.95 / Q_(1 - 0.95)|: `profit_factor` multiplied by
    the ratio of the 95% quantile of the returns to their 5% quantile, in
    absolute value. Quantiles interpolate linearly between order
    statistics.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        The ratio, never negative; 0.0 when there is no positive return.

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.
        UndefinedResultError: The 5% quantile is exactly 0; there is no
            negative return (the profit factor is undefined); or the result
            overflows float64.

    References:
        Attributed to Laurent Bernut (Bernut, L. (2021), *Algorithmic Short
        Selling with Python*, Packt). The tail ratio is the convention of
        the pyfolio and empyrical libraries and has no published source.
    """
    _validate_returns(returns)
    return _finite_result(_common_sense_ratio(returns), "common sense ratio")


def _common_sense_ratio(returns: npt.NDArray[np.float64]) -> float:
    if float(np.quantile(returns, 1.0 - _TAIL_QUANTILE)) == 0.0:
        raise UndefinedResultError(
            "the 5% quantile of the returns is 0; the common sense ratio is undefined"
        )
    if not np.any(returns < 0.0):
        raise UndefinedResultError(
            "there is no negative return, so the profit factor is undefined; "
            "the common sense ratio is undefined"
        )
    return _profit_factor(returns) * _tail_ratio(returns)


def cpc_index(returns: npt.NDArray[np.float64]) -> float:
    """CPC index: profit factor times win rate times payoff ratio.

    CPC = (G / L) * w * (mean win / |mean loss|), where G / L is
    `profit_factor`, w = #{r_t > 0} / #{r_t != 0} is the winning share of
    the non-zero periods (`hit_rate` with `zero_policy="exclude"`), and the
    payoff ratio divides the mean positive return by the absolute mean
    negative return. Zero returns enter none of the three factors.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        The index, positive.

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.
        UndefinedResultError: There is no positive return or no negative
            return, since the payoff ratio is then undefined; or the result
            overflows float64.

    References:
        Attributed in trading literature to Sunny J. Harris. No primary
        text is cited: the product implemented is the one QuantStats
        computes under this name.
    """
    _validate_returns(returns)
    return _finite_result(_cpc_index(returns), "CPC index")


def _cpc_index(returns: npt.NDArray[np.float64]) -> float:
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    if wins.size == 0 or losses.size == 0:
        raise UndefinedResultError(
            "the CPC index needs at least one positive and one negative return"
        )
    payoff_ratio = float(np.mean(wins)) / abs(float(np.mean(losses)))
    return _profit_factor(returns) * _win_rate(returns) * payoff_ratio


def outlier_win_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Outlier win ratio: 99% quantile of the returns over the mean non-negative return.

    Q_0.99(r) / mean(r_t : r_t >= 0). The quantile is taken over all
    returns and interpolates linearly between order statistics; the mean
    includes zero returns, unlike the payoff ratio in `cpc_index`.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        The ratio; negative when the 99% quantile is itself negative.

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.
        UndefinedResultError: No return is non-negative, or the
            non-negative returns are all zero; or the result overflows
            float64.

    References:
        None: the statistic is defined by the QuantStats library and has no
        published source.
    """
    _validate_returns(returns)
    return _finite_result(_outlier_win_ratio(returns), "outlier win ratio")


def _outlier_win_ratio(returns: npt.NDArray[np.float64]) -> float:
    non_negative = returns[returns >= 0.0]
    if non_negative.size == 0:
        raise UndefinedResultError("no return is non-negative; the outlier win ratio is undefined")
    mean_non_negative = float(np.mean(non_negative))
    if mean_non_negative == 0.0:
        raise UndefinedResultError(
            "the non-negative returns are all 0; the outlier win ratio is undefined"
        )
    return float(np.quantile(returns, _OUTLIER_WIN_QUANTILE)) / mean_non_negative


def outlier_loss_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Outlier loss ratio: 1% quantile of the returns over the mean negative return.

    Q_0.01(r) / mean(r_t : r_t < 0). The quantile is taken over all returns
    and interpolates linearly between order statistics.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        The ratio; positive when the 1% quantile is negative, 0.0 or
        negative otherwise.

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.
        UndefinedResultError: There is no negative return, or the result
            overflows float64.

    References:
        None: the statistic is defined by the QuantStats library and has no
        published source.
    """
    _validate_returns(returns)
    return _finite_result(_outlier_loss_ratio(returns), "outlier loss ratio")


def _outlier_loss_ratio(returns: npt.NDArray[np.float64]) -> float:
    negative = returns[returns < 0.0]
    if negative.size == 0:
        raise UndefinedResultError(
            "there is no negative return; the outlier loss ratio is undefined"
        )
    return float(np.quantile(returns, _OUTLIER_LOSS_QUANTILE)) / float(np.mean(negative))


def risk_of_ruin(returns: npt.NDArray[np.float64]) -> float:
    """Risk of ruin as QuantStats defines it: ((1 - w) / (1 + w)) ** n.

    w = #{r_t > 0} / #{r_t != 0} is the winning share of the non-zero
    periods (0.0 when every period is zero) and n is the number of periods,
    zero returns included.

    This is not a probability of ruin under any model. The fixed-stake
    formula it resembles is ((1 - A) / (1 + A)) ** C, with A the trader's
    advantage P(win) - P(loss) and C the number of units of capital, which
    is the gambler's-ruin probability (q / p) ** C of an even-payoff game.
    Here the win rate stands in for the advantage and the number of periods
    for the units of capital, so the value shrinks toward 0 as the sample
    grows, whatever the returns.

    Args:
        returns: Periodic simple returns, 1-D.

    Returns:
        A value in [0, 1]: 1.0 when no period wins (an all-zero series
        included), 0.0 when every non-zero period wins.

    Raises:
        ValueError: `returns` is not 1-D, is empty, or contains NaN or
            infinity.

    References:
        Feller, W. (1968), *An Introduction to Probability Theory and Its
        Applications*, vol. 1, 3rd ed., Wiley, ch. XIV, for the gambler's
        ruin. The fixed-stake form in terms of the trader's advantage is
        attributed to Kaufman, P.J., *Trading Systems and Methods*, Wiley.
    """
    _validate_returns(returns)
    return _finite_result(_risk_of_ruin(returns), "risk of ruin")


def _risk_of_ruin(returns: npt.NDArray[np.float64]) -> float:
    win_rate = _win_rate(returns)
    return float(((1.0 - win_rate) / (1.0 + win_rate)) ** returns.size)
