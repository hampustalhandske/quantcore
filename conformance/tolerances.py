"""Written tolerance policy for the conformance suite, one entry per target.

Default policy: a closed-form function is judged at ``1e-10`` relative tolerance. Anything looser
must justify itself here with a concrete numerical reason, tight enough to still catch a real
definitional difference. Each tolerance is derived from the numerical analysis of the computation
it judges and carries that reasoning in its justification; a tolerance is not widened to make a
target pass.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from conformance.compare import Tolerance

# A case whose effective relative tolerance would need to exceed this to explain the observed
# error is `ill_conditioned` -- excluded from the pass/fail decision, not silently widened
# without limit.
ILL_CONDITIONED_RTOL_CAP = 1e-6


@dataclass(frozen=True)
class ConditionAwareTolerance(Tolerance):
    """A `Tolerance` whose effective relative tolerance widens with a per-case condition number.

    A fixed relative tolerance cannot separate a genuine definitional difference from
    floating-point cancellation on an ill-conditioned input: `sharpe_ratio` agrees with its
    oracle to ~1e-15 relative on
    ordinary inputs, but on a near-constant short series (Sharpe of order 1e8) the standard
    deviation is computed from differences of nearly equal numbers, and a handful of draws differ
    by up to ~2e-9 relative for that reason alone, not because quantcore's formula is wrong.

    `rtol_eff(*args, **kwargs) = max(rtol, K * eps * kappa(*args, **kwargs) ** p)`, evaluated on
    the exact positional/keyword arguments a conformance case passes to the function under test.
    A case whose `rtol_eff` exceeds `ILL_CONDITIONED_RTOL_CAP` is `ill_conditioned`
    (`conformance.harness.check()` excludes it from the decision but still records and counts it)
    rather than being judged against an unboundedly loose tolerance.

    Attributes:
        kappa: Condition number of one case's inputs, `None` for a plain (non-condition-aware)
            tolerance.
        K: Constant from the tolerance's own first-order error-bound derivation, stated in that
            tolerance's `justification`.
        p: Power `kappa` is raised to in the bound, likewise stated and justified per tolerance.
    """

    kappa: Callable[..., float] | None = None
    K: float = 0.0
    p: float = 0.0

    def rtol_eff(self, *args: object, **kwargs: object) -> float:
        """Effective relative tolerance for one case, before the ill-conditioning cap is checked."""
        if self.kappa is None:
            return self.rtol
        eps = float(np.finfo(np.float64).eps)
        # `float.__pow__` is typed to allow a `complex` result, so typeshed widens
        # `kappa(...) ** self.p` to `Any`; both operands are `float` here, so the result is too.
        bound = self.K * eps * float(self.kappa(*args, **kwargs) ** self.p)
        return max(self.rtol, bound)

    def is_ill_conditioned(self, *args: object, **kwargs: object) -> bool:
        """Whether this case's `rtol_eff` exceeds `ILL_CONDITIONED_RTOL_CAP`."""
        if self.kappa is None:
            return False
        return self.rtol_eff(*args, **kwargs) > ILL_CONDITIONED_RTOL_CAP


def _sharpe_kappa(
    returns: npt.NDArray[np.float64], risk_free_per_period: float, periods_per_year: int
) -> float:
    """Condition number of one `sharpe_ratio` case: `max|r - rf_p| / std(r - rf_p, ddof=1)`.

    The sample variance `std(x, ddof=1) ** 2` is a difference of nearly equal numbers whenever `x`
    is close to constant; its first-order relative error from floating-point cancellation scales
    with `(max|x| / std(x)) ** 2` (Higham, "Accuracy and Stability of Numerical Algorithms", 2002,
    ch. 1.6, error analysis of summation), independently of which summation order the two
    implementations use. `x = returns - risk_free_per_period` is the exact quantity `sharpe_ratio`
    takes the mean and standard deviation of, so this reuses it directly rather than
    approximating with `returns` alone.
    """
    excess = returns - risk_free_per_period
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return float("inf")
    return float(np.max(np.abs(excess))) / std


def _drawdowns(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Drawdown at every period, as a positive fraction of the running peak of the wealth path.

    Wealth is `cumprod(1 + returns)` and the peak is its running maximum including the starting
    capital of 1, the convention of every drawdown-based quantity this module holds a tolerance
    for.
    """
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    result: npt.NDArray[np.float64] = (peak - wealth) / peak
    return result


def _calmar_kappa(returns: npt.NDArray[np.float64], periods_per_year: int) -> float:
    """Condition number of one Calmar case: `1 / maximum drawdown`; `inf` with no drawdown.

    A drawdown is the difference of two wealth values that agree in their leading digits, so a
    rounding error of one part in 2**53 on either becomes a relative error of that size divided
    by the drawdown. `periods_per_year` scales the numerator only and does not enter.
    """
    maximum_drawdown = float(np.max(_drawdowns(returns)))
    if maximum_drawdown == 0.0:
        return float("inf")
    return 1.0 / maximum_drawdown


def _ulcer_kappa(returns: npt.NDArray[np.float64]) -> float:
    """Condition number of one Ulcer-Index-denominated case: `1 / Ulcer Index`; `inf` at zero.

    The Ulcer Index is `sqrt(sum(drawdown**2) / (n - 1))`. An absolute rounding error of fixed size
    on each drawdown moves it by an absolute amount of the same order whatever its own size, so
    the relative error of anything divided by it grows as its reciprocal.
    """
    drawdowns = _drawdowns(returns)
    ulcer_index = float(np.sqrt(np.sum(drawdowns**2) / (returns.size - 1)))
    if ulcer_index == 0.0:
        return float("inf")
    return 1.0 / ulcer_index


TOLERANCES: dict[str, Tolerance] = {
    "sharpe_ratio": ConditionAwareTolerance(
        rtol=1e-10,
        atol=1e-9,
        justification=(
            "closed-form ratio of a mean and a ddof=1 standard deviation; atol above the default "
            "1e-10 relative floor only to stay well-defined when the expected value itself is "
            "near zero (a near-zero mean excess return), not because the two implementations "
            "disagree at any observed scale (measured max relative error ~3e-13 on 2000 random "
            "draws up to n=500). Condition-aware: rtol_eff = max(rtol, K*eps*kappa**p) with "
            "kappa = max|r - rf_p| / std(r, ddof=1), p = 2 (the sample variance's cancellation "
            "error scales with the square of that ratio, see _sharpe_kappa's docstring). Both "
            "quantcore and the oracle compute std via a numerically stable two-pass (mean, then "
            "sum-of-squared-deviations) algorithm, not the naive one-pass E[x^2]-E[x]^2 formula "
            "the worst-case kappa**2 bound assumes, so the realised error constant is well below "
            "the O(1) a naive-algorithm bound would need; K = 0.01 reflects that -- still large "
            "enough that a genuinely pathological case (near-constant short series, kappa of "
            "order 1e15-1e16 from a near-zero sampled std) crosses the 1e-6 ill-conditioned cap by "
            "many orders of magnitude regardless, while the milder case of a merely tiny-scale "
            "return series with a comparatively large risk-free rate (kappa of order 1e5-1e6, "
            "measured on this suite's `tiny_scale` family) stays under the cap, matching that "
            "family's realised sub-cap numerical agreement; an ordinary draw's kappa of a few "
            "units leaves rtol_eff at the plain 1e-10 floor regardless of K."
        ),
        kappa=_sharpe_kappa,
        K=0.01,
        p=2.0,
    ),
    "maximum_drawdown": Tolerance(
        rtol=1e-10,
        atol=1e-9,
        justification=(
            "closed-form cumprod/cummax reduction; atol accommodates a true drawdown of exactly "
            "0 (a never-declining path), where relative error is undefined and the comparison "
            "must fall back to the absolute term."
        ),
    ),
    "newey_west_cov": Tolerance(
        rtol=1e-9,
        atol=1e-11,
        justification=(
            "closed-form Bartlett-kernel sandwich estimator; quantcore accumulates the kernel sum "
            "lag-by-lag while the statsmodels-based reference (bread * S_hac_simple * bread) sums "
            "in a different order, so floating-point rounding differs at the last 2-3 significant "
            "digits for n up to 5000 and L up to a few dozen lags (measured max relative error "
            "~6e-14 on 500 random regressions up to n=300); rtol stays one order tighter than that "
            "margin so a genuinely wrong kernel or scaling constant is still caught."
        ),
    ),
    "square_root_market_impact": Tolerance(
        rtol=1e-12,
        atol=0.0,
        justification="closed form, three multiplications.",
    ),
    "short_borrow_cost": Tolerance(
        rtol=1e-12,
        atol=0.0,
        justification="closed form, three multiplications.",
    ),
    "margin_financing_cost": Tolerance(
        rtol=1e-12,
        atol=0.0,
        justification="closed form, three multiplications.",
    ),
    "sortino_ratio": Tolerance(
        rtol=1e-10,
        atol=1e-9,
        justification=(
            "closed-form ratio of a mean and a full-sample downside deviation; atol above the "
            "1e-10 relative floor for the same reason as sharpe_ratio: a near-zero numerator."
        ),
    ),
    "calmar_ratio": ConditionAwareTolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "ratio of a CAGR (a power/root, not a pure sum-of-products) to a max-drawdown "
            "reduction; one order looser than the pure closed-form default to absorb the CAGR "
            "power's floating-point rounding, still tight enough to catch a definitional "
            "mismatch (a wrong return_method disagrees at the percent level, not the 1e-9 level). "
            "Condition-aware: rtol_eff = max(rtol, K*eps*kappa**p) with kappa = 1/MDD (the "
            "maximum drawdown of the case's own returns), K = 2, p = 1, because a fixed relative "
            "tolerance cannot hold as MDD -> 0: on returns of order 1e-8 (this suite's "
            "`tiny_scale` family, MDD of order 1e-8) the two sides differ by up to 1.5e-8. "
            "Derivation, to first order in the unit roundoff u = eps/2. Both sides build the same "
            "wealth path W = cumprod(1 + r) with the same operations in the same order, and the "
            "same numerator (empyrical's annual_return and quantcore's cagr raise the same "
            "product to the same power: bit-identical on 6132 inputs), so they can differ only in "
            "what each does after W. quantcore computes (P - W)/P with P the running peak: the "
            "subtraction is exact for a drawdown under 50% (Sterbenz), the division rounds once, "
            "relative error u. empyrical first multiplies the path by its starting value 100, "
            "which rounds W and P independently (relative error u each), then computes "
            "(W' - P')/P': the two scaling errors do not cancel in the difference, an absolute "
            "error up to 2u(1 - MDD) on a quantity of size MDD, plus u from the division. The two "
            "maximum drawdowns therefore differ by at most eps*(1 - MDD)/MDD + eps = eps/MDD "
            "relative, so p = 1; dividing the common numerator by each adds u per side, giving "
            "eps*(1/MDD + 1) <= 2*eps/MDD since MDD <= 1, so K = 2. Checked on 6132 inputs "
            "outside this suite (volatility 1e-2 to 1e-12 and the ten return families): largest "
            "difference 0.64 of eps*(1/MDD + 1). The bound is on the difference between the two "
            "implementations, which is what this tolerance judges; it is not a bound on either "
            "one's distance from the exact Calmar ratio of the same float64 returns. Against "
            "rational arithmetic, on the 11 cases of this suite's ci run that a fixed rtol of 1e-9 "
            "rejects, both maximum drawdowns are within eps/MDD of the exact one (quantcore at "
            "most 0.39, empyrical 0.59 of it), but the CAGR numerator both sides share is itself "
            "E**(p/n) - 1 with E - 1 as small as 1e-10 and is off by up to 1.3e-6 relative, "
            "identically on both sides, which no comparison between them can see."
        ),
        kappa=_calmar_kappa,
        K=2.0,
        p=1.0,
    ),
    "calmar_ratio_arithmetic": ConditionAwareTolerance(
        rtol=1e-10,
        atol=1e-9,
        justification=(
            "closed-form arithmetic mean over a max-drawdown reduction. Condition-aware with the "
            "same kappa = 1/MDD, K = 2, p = 1 as calmar_ratio and for the same reason (see its "
            "derivation): the numerator mean(r)*periods_per_year is the same on both sides, the "
            "two maximum drawdowns differ by at most eps/MDD relative because empyrical scales "
            "the wealth path by 100 before subtracting the peak, and the final division adds u "
            "per side (largest difference on the same 6132 inputs: 0.88 of eps*(1/MDD + 1)). "
            "Against rational arithmetic, on the 10 cases of this suite's ci run that a "
            "fixed rtol of 1e-10 rejects, both sides are within eps/MDD of the exact ratio "
            "(quantcore at most 0.58, the oracle 0.97 of it; quantcore the closer in 5 of 10)."
        ),
        kappa=_calmar_kappa,
        K=2.0,
        p=1.0,
    ),
    "information_ratio": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "closed-form ratio of a mean and a ddof=1 standard deviation of the active-return "
            "series, compared after annualising empyrical's per-period excess_sharpe by "
            "sqrt(periods_per_year); one order looser than the pure closed-form default to "
            "absorb that extra multiplication's rounding."
        ),
    ),
    "hit_rate": Tolerance(
        rtol=1e-10,
        atol=1e-10,
        justification="closed-form ratio of two counts.",
    ),
    "profit_factor": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification="closed-form ratio of two sums; one order looser to absorb summation order.",
    ),
    "ulcer_index": Tolerance(
        rtol=1e-10,
        atol=1e-15,
        justification=(
            "sqrt(sum(D**2) / (n - 1)) over the drawdowns D of the same wealth path "
            "W = cumprod(1 + r) on both sides. rtol is the closed-form default. atol is the "
            "absolute error the oracle's drawdown formula leaves on the index, which does not "
            "shrink with the index: quantstats computes each drawdown as W/P - 1, rounding the "
            "quotient by up to u/2 = eps/4 in absolute terms whatever the size of the drawdown, "
            "where quantcore's (P - W)/P subtracts exactly and rounds only in relative terms. "
            "Perturbing every non-zero drawdown by at most d moves sum(D**2) by 2*d*sum(D) and the "
            "index by d*sum(D)/sqrt((n - 1)*sum(D**2)) <= d*sqrt(n/(n - 1)) <= 1.42*d "
            "(Cauchy-Schwarz): 7.9e-17 for d = eps/4. Drawdowns above 50%, where the oracle's "
            "1 + (W - 1) and its W/P - 1 each round once more, and wealth beyond 2**53 raise d "
            "to at most 5.25*u, 8.2e-16 on the index. atol = 1e-15 covers that; on returns of "
            "order 1e-8 (index of order 1e-8) it is 1e-7 relative, three orders below the "
            "1/(2n) >= 1e-4 by which an n-for-(n - 1) divisor would differ. Checked on 3446 "
            "inputs outside this suite: largest absolute difference 1.1e-16, and 5.3e-17 where "
            "the index is below 1e-3."
        ),
    ),
    "ulcer_performance_index": ConditionAwareTolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "(prod(1 + r) - 1) / UI. The numerator is the same product on both sides; the Ulcer "
            "Index differs by at most 1.42*eps/4 = 0.354*eps in absolute terms because quantstats "
            "rounds each drawdown W/P - 1 by up to eps/4 where quantcore's (P - W)/P subtracts "
            "exactly (derivation under ulcer_index). Relative to the index that is 0.354*eps/UI, "
            "unbounded as UI -> 0, so the tolerance is condition-aware: rtol_eff = max(rtol, "
            "K*eps*kappa**p) with kappa = 1/UI, p = 1, K = 0.5 (0.354 rounded up). The bound only "
            "exceeds the 1e-10 floor for UI < 1.1e-6, where every drawdown is far below 50% and "
            "d = eps/4 applies. Checked on 2994 inputs outside this suite: largest difference "
            "0.67 of 0.354*eps/UI. atol only keeps the comparison defined when the compounded "
            "return, and so the ratio, is zero."
        ),
        kappa=_ulcer_kappa,
        K=0.5,
        p=1.0,
    ),
    "serenity_index": ConditionAwareTolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "sum(r) / (UI * CDaR / std(r, ddof=1)), CDaR the mean of the drawdowns beyond the "
            "threshold q = mean(D) + 1.645*std(D, ddof=1), or q itself when none is. Same cause as "
            "ulcer_performance_index: quantstats' drawdowns carry an absolute rounding error up "
            "to d = eps/4 each. It moves UI by at most 0.354*eps/UI relative. It moves a tail "
            "mean by at most d, and CDaR >= q >= UI/sqrt(2) (because UI**2 = std(D)**2 + "
            "n/(n - 1)*mean(D)**2 <= 2*q**2), so by 0.354*eps/UI relative. When CDaR is q, q moves "
            "by at most (1 + 1.645*sqrt(2))*d, 1.18*eps/UI relative. Worst case 1.53*eps/UI: "
            "kappa = 1/UI, p = 1, K = 2. Checked on 2989 inputs outside this suite: largest "
            "difference 0.31 of 1.53*eps/UI. The bound assumes both sides select the same tail: a "
            "drawdown within rounding distance of q that falls on different sides of it changes "
            "the tail mean by an amount no tolerance covers, a discontinuity of the definition "
            "itself. atol only keeps the comparison defined when the returns sum to zero."
        ),
        kappa=_ulcer_kappa,
        K=2.0,
        p=1.0,
    ),
    "gain_to_pain": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: sum(r) over |sum of the negative r|, the same NumPy reductions on both "
            "sides. atol covers a ratio near zero, where the numerator cancels and a relative "
            "comparison is meaningless: the ratio's absolute error is a few eps times "
            "(1 + profit factor)."
        ),
    ),
    "common_sense_ratio": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: profit factor (two sums of same-signed terms) times |Q(0.95)/Q(0.05)|, "
            "each quantile a linear interpolation between two order statistics by the same NumPy "
            "routine on both sides. atol covers the value 0 a series with no gain gives."
        ),
    ),
    "cpc_index": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: profit factor times a ratio of two counts times a ratio of two means of "
            "same-signed terms; nothing cancels."
        ),
    ),
    "outlier_win_ratio": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: one linearly interpolated quantile over one mean of non-negative terms; "
            "atol covers a 99% quantile at or near zero."
        ),
    ),
    "outlier_loss_ratio": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: one linearly interpolated quantile over one mean of negative terms; "
            "atol covers a 1% quantile at or near zero."
        ),
    ),
    "risk_of_ruin": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed form: a ratio of two integer counts raised to an integer power. The value "
            "lies in [0, 1] and underflows toward 0 as the series lengthens, where only an "
            "absolute comparison is meaningful; hence atol."
        ),
    ),
    "value_at_risk": Tolerance(
        rtol=1e-10,
        atol=1e-10,
        justification="closed form: a mean, a ddof=1 standard deviation, and a normal quantile.",
    ),
    "conditional_value_at_risk": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "closed form, with an extra normal-pdf evaluation and division by (1 - alpha); one "
            "order looser than the pure closed-form default for that additional operation."
        ),
    ),
    "value_at_risk_cornish_fisher": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification=(
            "closed form (a quartic polynomial in z plus a normal quantile), but with the same "
            "order-of-summation freedom as ols; two orders looser than the default to absorb "
            "that, still tight enough to catch a wrong coefficient in the expansion."
        ),
    ),
    "ols_coefficients": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification=(
            "quantcore solves via np.linalg.lstsq, statsmodels via a pseudo-inverse/Moore-Penrose "
            "path; both are correct least-squares solutions but differ in the last few "
            "significant digits on ill-conditioned regressors (measured max relative error "
            "~4e-10 on 500 random regressions up to n=300), one order looser than the closed-form "
            "default to absorb that, still far tighter than a definitional mismatch would produce."
        ),
    ),
    "ols_r_squared": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification="same lstsq-vs-pseudo-inverse rounding margin as ols_coefficients.",
    ),
    "newey_west_optimal_lags": Tolerance(
        rtol=1e-10,
        atol=1e-9,
        justification="closed-form floor(4*(n/100)^(2/9)); atol covers the lags=0 case exactly.",
    ),
    "jarque_bera": Tolerance(
        rtol=1e-9,
        atol=1e-12,
        justification=(
            "closed form in the second, third and fourth central moments, then exp(-JB/2). Both "
            "sides centre the deviations twice, so the moments are good to a few n*eps whatever "
            "the ratio of mean to spread. The statistic is less accurate than its moments when "
            "the sample is nearly symmetric, because skewness enters squared: its relative error "
            "is about eps*n/|S|, 1e-11 at n=5000 and |S|=1e-4, which is why rtol is one order "
            "above the closed-form default. The p-value's relative error is half the statistic's "
            "absolute error, below 1e-11 wherever p exceeds atol (JB < 55). atol covers a "
            "statistic near zero and makes any two p-values below 1e-12 equal."
        ),
    ),
    "ewma_variance": Tolerance(
        rtol=1e-10,
        atol=1e-10,
        justification=(
            "hand recursion using the identical closed-form formula and seeding quantcore "
            "documents (sigma_0^2 = returns[0]**2); measured max abs/rel error exactly 0.0 on "
            "500 random draws up to n=2000, since both sides perform the same arithmetic in the "
            "same order."
        ),
    ),
    "garch_11_variance": Tolerance(
        rtol=1e-12,
        atol=1e-14,
        justification=(
            "the same recursion evaluated by two implementations from the same start (arch's "
            "backcast of the residuals), compared from the first step; measured max abs error "
            "9e-19 and max rel error 8e-16 on 400 draws of 1 to 2000 points, so rtol is a "
            "little over three orders above the measured error, covering accumulation over "
            "the longest series."
        ),
    ),
    "egarch_11_variance": Tolerance(
        rtol=1e-11,
        atol=1e-12,
        justification=(
            "as garch_11_variance, but for the log-variance recursion (an extra log/exp round "
            "trip per step vs. plain variance); measured max abs error 9e-16 and max rel error "
            "1.1e-12 outside the near-constant family, so rtol and atol are about one order "
            "above."
        ),
    ),
    "gjr_garch_11_variance": Tolerance(
        rtol=1e-12,
        atol=1e-14,
        justification=(
            "as garch_11_variance; measured max abs error 9e-19 and max rel error 9e-16 on 400 "
            "draws."
        ),
    ),
    "hmm_fit_transition_matrix": Tolerance(
        rtol=0.05,
        atol=0.05,
        justification=(
            "two independent EM fits (quantcore's own Baum-Welch; hmmlearn's, seeded from the "
            "same quantile-bucket initialisation, conformance.targets.regime."
            "_quantile_bucket_init) of the same non-convex likelihood; measured relative "
            "difference ~1.1% on a 400-observation well-separated 2-state series -- hmm_fit_log_"
            "likelihood is the tighter, primary criterion (see below)."
        ),
    ),
    "hmm_fit_means": Tolerance(
        rtol=0.05,
        atol=0.05,
        justification="as hmm_fit_transition_matrix; measured relative difference ~0.2%.",
    ),
    "hmm_fit_variances": Tolerance(
        rtol=0.05,
        atol=0.05,
        justification="as hmm_fit_transition_matrix; measured relative difference ~0.14%.",
    ),
    "hmm_fit_initial_probs": Tolerance(
        rtol=0.05,
        atol=0.05,
        justification=(
            "as hmm_fit_transition_matrix; atol dominates here since initial_probs is a "
            "probability, near 0 or 1 on well-separated regimes, where a tiny absolute gap "
            "(measured <1e-3) is a huge relative one."
        ),
    ),
    "hmm_fit_log_likelihood": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification=(
            "hmmlearn.score, evaluated on quantcore's own fitted parameters, cross-validates the "
            "forward-algorithm arithmetic on identical numbers (not a second fit) -- measured "
            "exact agreement (0.0 difference) on every case tried."
        ),
    ),
    "hmm_decode": Tolerance(
        rtol=0.0,
        atol=1e-6,
        justification=(
            "hmmlearn's Viterbi decode, given quantcore's own parameters directly (no independent "
            "fit, so no label-permutation ambiguity); measured exact agreement (100%) on "
            "well-separated regimes. atol admits only exact-integer-valued disagreement (none "
            "observed), not a fractional tolerance -- a genuine one-state disagreement must still "
            "be caught."
        ),
    ),
    "hmm_predict_proba": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        justification=(
            "hmmlearn's forward-backward smoother, given quantcore's own parameters directly; "
            "measured max abs difference ~2e-21 (machine precision) since both implement the "
            "identical algorithm on identical numbers."
        ),
    ),
    "hmm_filtered_proba": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        justification=(
            "hand forward-algorithm implementation (conformance.targets.regime."
            "_forward_filtered_proba), independent of quantcore's own (private) forward pass; "
            "measured max abs difference ~3e-20 (machine precision)."
        ),
    ),
    "dynamic_hedge_ratio": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        justification=(
            "hand-written scalar Kalman regression (one state, no intercept, H_t = x_t) started "
            "from the same prior as quantcore, beta ~ N(initial_beta, 1) with initial_beta = 0 "
            "unless given, and compared on every element from the first. Both sides do the same "
            "predict/update arithmetic, so the measured difference is exactly 0.0; an "
            "implementation with different arithmetic (filterpy's matrix filter with the "
            "Joseph-form covariance update) differs from quantcore by at most 1e-13 relative on "
            "400 series of 2 to 800 points, the rounding of the innovation y_t - x_t * beta. The "
            "tolerance sits between that and the smallest difference of definition it has to "
            "catch: a prior mean of y[0] / x[0] instead of 0 moves the first value by "
            "R / (x_0^2 * (1 + Q) + R) relative, 2e-5 to 2e-4 for x near 5 and R = 1e-3, and the "
            "second by 8e-6 to 5e-5."
        ),
    ),
    "black_scholes_merton_price": Tolerance(
        rtol=1e-9,
        atol=1e-7,
        justification=(
            "closed-form BSM price; quantcore and the reference N()/erf-based formula agree to "
            "~1e-13 absolute on ordinary inputs, but a deep OTM/ITM price near zero is the "
            "difference of two near-equal exp()*N() terms, so relative error grows while the "
            "absolute error stays at machine precision; atol dominates there (measured max abs "
            "error 1.14e-13, max rel error 1.48e-3 on 800 conformance draws across the "
            "atm/deep-itm/deep-otm/dividend/rate/vol families in test_tier2_pricing_extra.py; "
            "every case stayed within this tolerance -- all quantcore)."
        ),
    ),
    "bsm_delta": Tolerance(
        rtol=1e-8,
        atol=1e-9,
        justification=(
            "closed-form BSM delta = exp(-qT)*N(d1) (or -1 for a put); same cancellation as "
            "black_scholes_merton_price on deep OTM/ITM delta near 0: measured max abs error "
            "1.11e-16, max rel error 5.17e-6 on the same 800-case sweep, atol dominates and "
            "every case stayed within this tolerance."
        ),
    ),
    "bsm_vega": Tolerance(
        rtol=1e-10,
        atol=1e-12,
        justification=(
            "closed-form BSM vega = S*exp(-qT)*phi(d1)*sqrt(T), a single product with no "
            "cancellation; measured max relative error ~7.5e-16 against quantcore.pricing.bs_vega "
            "on the same sweep, at the closed-form default."
        ),
    ),
    "bs_gamma": Tolerance(
        rtol=1e-10,
        atol=1e-15,
        justification=(
            "closed-form BSM gamma = exp(-qT)*phi(d1)/(S*sigma*sqrt(T)); no cancellation (all "
            "factors are strictly positive products), so quantcore and the reference N()/erf-"
            "based formula agree near machine precision everywhere, including deep OTM/ITM: "
            "measured max abs error 8.1e-20, max rel error 9.0e-16 on 2000 draws across the "
            "atm/deep-itm/deep-otm/dividend/rate/vol families in "
            "conformance/test_tier2_pricing_extra.py's bsm_case generator."
        ),
    ),
    "bs_theta": Tolerance(
        rtol=1e-8,
        atol=1e-10,
        justification=(
            "closed-form BSM theta (a sum/difference of several exp()*N()/phi() terms, divided "
            "by 365); a deep OTM/short-expiry theta near zero is a difference of near-equal "
            "terms, so relative error grows while the absolute error stays at machine precision -- "
            "atol dominates there. Measured max abs error 5.6e-17 across the full 3000-case sweep "
            "(any magnitude), max rel error 5.1e-13 restricted to cases with |expected| > 1e-6 "
            "(the near-zero cases' relative error is cancellation noise, not a formula "
            "difference); rtol one order looser than the closed-form default to leave headroom "
            "above that measured 5.1e-13, atol one order above the measured 5.6e-17 max abs error."
        ),
    ),
    "bs_rho": Tolerance(
        rtol=1e-6,
        atol=1e-11,
        justification=(
            "closed-form BSM rho = K*T*exp(-rT)*N(d2) (or the put mirror); a short-expiry deep "
            "OTM rho is close to zero (little time for the rate to matter), so relative error "
            "grows there from ordinary floating-point rounding in exp()/N(), not a formula "
            "difference -- atol dominates. Measured max abs error 9.1e-13 across the full "
            "3000-case sweep (any magnitude), max rel error 1.1e-8 restricted to cases with "
            "|expected| > 1e-6; rtol two orders looser than the closed-form default and atol two "
            "orders above the measured max abs error, both comfortably above the measured values "
            "while still far tighter than a definitional mismatch (a wrong sign or a missing "
            "factor of K*T) would produce."
        ),
    ),
    "option_roll_schedule": Tolerance(
        rtol=0.0,
        atol=0.0,
        justification=(
            "returns a date, not a float: `conformance.targets.pricing_extra."
            "_DateComparator` requires the two dates to be exactly identical (there is no "
            "meaningful notion of 'close' for a discrete choice of listed expiry); rtol/atol are "
            "unused by that comparator and kept at 0.0 so they can never be mistaken for a real "
            "tolerance."
        ),
    ),
    "protective_put_sizing": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification=(
            "sizing arithmetic (ceil/floor of a notional-to-shares-to-contracts conversion, plus "
            "a put price from the closed-form BSM formula) around a single BSM put evaluation; "
            "one order looser than the closed-form default for that put price's own rounding "
            "(see black_scholes_merton_price's tolerance), propagated through the sizing "
            "division/multiplication chain. `contracts` (recorded as the tuple's first, integer-"
            "valued component) is compared at this same tolerance but is exact in practice: it is "
            "a ceil()/floor() of a quantity that agrees between quantcore and the oracle to "
            "float64 precision, so an off-by-one contract is still caught (it fails at rtol=1e-8 "
            "for any contracts count above single digits)."
        ),
    ),
    # --- conformance.targets.time_series -----------------------------------------------------
    "factor_loadings_loadings": Tolerance(
        rtol=1e-8,
        atol=1e-10,
        justification=(
            "closed-form OLS coefficients; measured exact agreement (near machine precision) "
            "against statsmodels.OLS on the same design matrix, one order looser than the pure "
            "closed-form default to absorb lstsq-vs-pinv rounding, as ols_coefficients above."
        ),
    ),
    "factor_loadings_standard_errors": Tolerance(
        rtol=1e-7,
        atol=1e-10,
        justification=(
            "Newey-West HAC standard errors; inherits newey_west_cov's own summation-order "
            "rounding margin (an extra sqrt of a diagonal element on top), one order looser than "
            "newey_west_cov's own tolerance to absorb that additional operation."
        ),
    ),
    "factor_loadings_t_stats": Tolerance(
        rtol=1e-7,
        atol=1e-8,
        justification="ratio of two already-tight quantities (loadings, standard_errors); atol "
        "guards a near-zero loading's t-stat from an unstable relative comparison.",
    ),
    "factor_loadings_p_values": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        justification=(
            "two-sided normal survival function of the t-stat; one order looser than t_stats to "
            "absorb norm.sf's own rounding, still tight enough to catch a wrong tail convention."
        ),
    ),
    "factor_loadings_r_squared": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification="same lstsq-vs-pinv rounding margin as ols_r_squared.",
    ),
    "factor_loadings_residuals": Tolerance(
        rtol=1e-8,
        atol=1e-8,
        justification="y - X @ coefficients; inherits factor_loadings_loadings' own margin.",
    ),
    "fama_macbeth_mean_risk_premia": Tolerance(
        rtol=1e-8,
        atol=1e-10,
        justification=(
            "mean of T per-period OLS-on-betas slopes; measured exact agreement against "
            "linearmodels.FamaMacBeth(cov_type='unadjusted') in the conformance test, one order "
            "looser than the closed-form default for the same lstsq rounding as ols_coefficients."
        ),
    ),
    "fama_macbeth_standard_errors": Tolerance(
        rtol=1e-7,
        atol=1e-10,
        justification=(
            "std(lambda_t, ddof=1)/sqrt(T); measured exact agreement against linearmodels."
            "FamaMacBeth(cov_type='unadjusted') in the conformance test -- quantcore's default "
            "convention (n_lags_nw=None) is the plain Fama & MacBeth (1973) time-series standard "
            "error, the same one linearmodels' 'unadjusted' estimator computes; one order looser "
            "than the closed-form default for the extra sqrt/divide."
        ),
    ),
    "arima_fit": Tolerance(
        rtol=1e-6,
        atol=1e-3,
        justification=(
            "iterative estimator: quantcore's L-BFGS-B fit and statsmodels' own fit maximize the "
            "same exact Gaussian likelihood and stop at slightly different points near its "
            "maximum, so raw parameters are never compared -- this tolerance governs only the "
            "`log_likelihood` quantity, both solutions evaluated under statsmodels' own `loglike` "
            "(never re-fit). atol=1e-3 nats is the oracle's accuracy, not quantcore's: "
            "quantcore's stopping rule leaves it within about 1e-9 nats of the maximum, while "
            "statsmodels' fit ends up to several 1e-4 nats below it (it stops some ARMA fits on "
            "its iteration limit) and its `loglike` carries about 1e-8 nats of evaluation noise, "
            "so a narrower band would count the oracle's own shortfall as quantcore wins. "
            "rtol=1e-6 scales the band with the log-likelihood's magnitude (about 1e-3 nats more "
            "at 800 observations). `params` is reported, never compared -- see "
            "`DecisiveFloatArrayComparator`."
        ),
    ),
    "arima_residuals": Tolerance(
        rtol=1e-9,
        atol=1e-10,
        justification=(
            "conditional (sum-of-squares) residuals against scipy.signal.lfilter run as the "
            "inverse ARMA filter phi(B)/theta(B) from the same initial conditions (lfiltic): the "
            "same linear recursion evaluated in transposed direct form II instead of quantcore's "
            "loop over the difference equation, so the two differ only by rounding. Each step's "
            "rounding error is of order eps times the terms summed and is carried forward by the "
            "1/theta(B) recursion, whose gain is at most 1/(1 - rho) for largest MA root modulus "
            "rho (about 3 for the generated families, rho <= 0.69), which puts the difference "
            "near 1e-14 on these O(10) series. rtol=1e-9 is one order looser than the "
            "closed-form default to leave room for that accumulation over up to 1000 steps; "
            "atol=1e-10 covers residuals that are themselves near zero."
        ),
    ),
    "arima_forecast": Tolerance(
        rtol=1e-9,
        atol=1e-10,
        justification=(
            "exact conditional-mean forecast against statsmodels' SARIMAX filtered at the same "
            "coefficients with exact diffuse initialisation and its steady-state shortcut "
            "disabled (`tolerance=0.0`), so both sides run the same Kalman recursion to "
            "rounding; with the shortcut left on, statsmodels' own forecast is off by 1e-10 to "
            "1e-9 and could not anchor this tolerance. The two filters arrange the arithmetic "
            "differently (quantcore uses the shift structure of the transition matrix and "
            "switches to the steady-state gain once the filtered covariance is below 1e-14 "
            "relative), and a perturbation of the gain decays at the MA root rate, which puts "
            "the difference near 1e-14 to 1e-13 on these series. rtol=1e-9 is one order looser "
            "than the closed-form default for the accumulated rounding of up to 1000 filter "
            "steps; atol=1e-10 covers forecasts that decay to zero (an MA(q) forecast is exactly "
            "0 beyond q steps), where a relative error is undefined."
        ),
    ),
    "select_arima_order": Tolerance(
        rtol=0.0,
        atol=0.0,
        justification=(
            "order selection is inherently ambiguous between correlated models, so agreement "
            "rate is reported, not asserted -- this quantity's comparator "
            "(edgelab.backend.function.ReportOnlyScalarComparator) never fails a case and "
            "build_report.build_resolution_map records it as unverifiable without consulting "
            "this tolerance for a pass/fail decision at all, so rtol=atol=0 is the honest value "
            "rather than a wide number chosen to never trip: no code path judges 'close' against "
            "it any more."
        ),
    ),
    "ou_half_life": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "closed-form AR(1)-OLS slope through -ln(2)/ln(1+a); the oracle anchors the slope "
            "itself to statsmodels.OLS (an independent library, for oracle independence), one "
            "order looser than the pure closed-form default for the extra log evaluation and "
            "statsmodels' own QR-based solve."
        ),
    ),
    "spread_zscore": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "closed-form rolling mean/ddof=1-std z-score; the oracle anchors the rolling "
            "window itself to pandas.Series.rolling(window) (an independent library, for "
            "oracle independence), one order looser than the pure closed-form default for the "
            "rolling-window division."
        ),
    ),
    # --- portfolio family (conformance.targets.portfolio) ---------------------------------------
    "ledoit_wolf_shrinkage": Tolerance(
        rtol=1e-6,
        atol=1e-6,
        justification=(
            "matrix quantity: sklearn.covariance.LedoitWolf(assume_centered=False) matches "
            "quantcore's population (1/T) normalisation and sample-mean centering exactly (both "
            "documented in quantcore.portfolio.covariance's module docstring); measured max "
            "abs error ~1.2e-15 on a 200x5 standard-normal panel, well inside the closed-form "
            "default. shrinkage quantity: recovered from quantcore's returned matrix by projecting "
            "(cov - sample) onto (target - sample) in the Frobenius inner product (see "
            "_recovered_shrinkage's docstring); this projection is itself an extra division by a "
            "sum of squared matrix entries, so rtol/atol are loosened four orders from the "
            "closed-form default to absorb that division's amplification of the matrix "
            "comparison's own ~1e-15 noise on a small-k panel, while still catching a shrinkage "
            "intensity that is wrong at any practically meaningful scale; measured shrinkage "
            "agreement ~1e-13 on the same panel."
        ),
    ),
    # The four optimiser functions below are iterative estimators without a unique optimum
    # (see conformance.targets.portfolio's module docstring), so they are compared on their
    # ATTAINED OBJECTIVE via classify_iterative_estimator's sign-test rule. This SAME Tolerance
    # governs both each case's informational `within_tolerance` flag (shown per-case in the
    # report) AND the status decision itself: `build_report` passes this rtol/atol straight
    # into `classify_iterative_estimator`, so a case is a win/tie/loss margin of
    # `atol + rtol*|ref|` -- there is no second, harness-only decision tolerance.
    "min_variance_weights": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        justification=(
            "convex QP with a unique optimum; quantcore's SLSQP and the oracle's cvxpy/CLARABEL "
            "interior-point solver are two different numerical procedures for the identical "
            "program (same objective, same sum-to-one/long-only constraints, read directly from "
            "quantcore.portfolio.optimization._min_variance_weights), so their attained objectives "
            "should coincide far tighter than a generic iterative-estimator gap; measured relative "
            "objective gap ~1e-8 to ~1e-10 on 300 well-conditioned random covariances up to k=10."
        ),
    ),
    "unconstrained_mean_variance_weights": Tolerance(
        rtol=1e-9,
        atol=1e-10,
        justification=(
            "closed form on both sides (quantcore and the reference both solve Sigma @ w = mu via "
            "np.linalg.solve, no iterative optimiser at all), one order looser than the pure "
            "closed-form default only for the extra quadratic-form evaluation building the "
            "objective; measured relative gap consistent with 0 (both sides perform the same "
            "direct linear solve)."
        ),
    ),
    "risk_parity_weights": Tolerance(
        rtol=1e-3,
        atol=1e-3,
        justification=(
            "compared directly on WEIGHTS, not an objective (see conformance.targets.portfolio's "
            "module docstring): Spinu "
            "(2013)'s reformulation is strictly convex, so the risk-parity portfolio is unique and "
            "two correct solvers must agree on the weight vector itself. Generous relative to the "
            "closed-form default because quantcore's SLSQP least-squares-of-contribution-"
            "differences formulation and the oracle's independent log-barrier L-BFGS-B solver "
            "(conformance.targets.portfolio._spinu_risk_parity_weights) are very different "
            "numerical paths to the same point; measured agreement ~1e-7 on well-behaved cases in "
            "an empirical 200-case sweep (seed 100, k in 2..10, well_conditioned family), while "
            "quantcore's SLSQP failed to reach an equal-risk-contribution solution (or raised "
            "RuntimeError on non-convergence) on 13/200 cases, with weight deviations up to "
            "0.39. This tolerance is tight "
            "enough that both outcomes are correctly classified: convergent cases pass, divergent "
            "ones fail by orders of magnitude."
        ),
    ),
    "implied_equilibrium_returns": Tolerance(
        rtol=1e-10,
        atol=1e-10,
        justification=(
            "closed-form Pi = risk_aversion * Sigma @ w_mkt, hand-derived from He & Litterman "
            "(1999) eq. 4 -- no library implements this single matrix-vector product, so the "
            "reference is anchored by citation plus the two independent cross-checks in "
            "conformance/test_tier2_portfolio.py (zero-confidence view recovers the prior; "
            "infinite-confidence view recovers the view) that a wrong transcription would break."
        ),
    ),
    "black_litterman": Tolerance(
        rtol=1e-9,
        atol=1e-9,
        justification=(
            "closed-form Black-Litterman posterior via two np.linalg.solve calls against a (v, v) "
            "system, hand-derived from He & Litterman (1999) eq. 8 -- no library implements this "
            "exact inverse-free form, so the reference is anchored by citation plus two "
            "independent cross-checks in conformance/test_tier2_portfolio.py (a zero-confidence "
            "view recovers the prior; an infinite-confidence view recovers the view) "
            "that a wrong transcription would break; one order looser than the pure closed-form "
            "default to absorb the two linear solves' rounding, still far tighter than a "
            "definitional mismatch would produce; measured exact agreement (0.0 difference) "
            "against an independently-coded hand implementation of the same "
            "algebraically-equivalent inverse-free form quantcore's own module docstring derives."
        ),
    ),
}
