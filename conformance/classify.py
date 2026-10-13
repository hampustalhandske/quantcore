"""`classify()`: pure function turning recorded conformance cases into a conformance status.

No file I/O, no process state, no randomness: given the same `CaseResult`s and flags, `classify()`
always returns the same status. That makes it directly unit-testable without running the (slow,
hypothesis-driven) conformance suite itself, and it is the single place the verdict rule
("verified" only when quantcore was within tolerance of its oracle on every recorded case) is
decided.
"""

import dataclasses
import math
import statistics
from collections.abc import Sequence
from typing import Literal

from scipy.stats import binomtest

type ConformanceStatus = Literal["verified", "non-standard", "defect", "unverifiable"]


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """One differential-testing case: quantcore's and the oracle's outcomes on one input, compared.

    `harness.check()` builds and records these; `conformance/conftest.py` serialises them to
    `results/raw.json`; `build_report.py` reads them back to call `classify()`.

    One input can yield several `CaseResult`s: a function returning several named quantities
    (e.g. `engle_granger_test`'s `statistic` and `p_value`) is recorded as one `CaseResult` per
    quantity, all sharing the same `function`, `family`, and `length_bucket`.

    Attributes:
        function: Name of the `ConformanceTarget` under test.
        family: Which generator family produced this case's input (e.g. `"student_t3"`).
        length_bucket: Which declared length bucket (`"short"`, `"medium"`, `"long"`) the input's
            length falls into.
        abs_err: Maximum absolute error of quantcore against the reference for this quantity.
        rel_err: Maximum relative error of quantcore against the reference for this quantity.
        within_tolerance: Whether quantcore's result was within the function's tolerance of the
            reference's for this quantity; `False` whenever quantcore raised or returned a
            non-finite value.
        quantcore_raised: `"{ExcType}: {message}"` if the quantcore callable raised or returned a
            non-finite value, else `None`.
        reference_raised: `"{ExcType}: {message}"` if the reference callable raised, else `None`
            (in practice always `None`: `harness.check()` fails the run immediately if the
            reference callable raises, rather than recording a case for it).
        quantity: Which named quantity this case measures; `"value"` for a function with a single
            recorded result.
        ill_conditioned: Whether this case's effective tolerance (`conformance.tolerances.
            ConditionAwareTolerance.rtol_eff`) exceeded the ill-conditioning cap, so it is excluded
            from the pass/fail decision but still counted and reported. Always `False` for a
            function whose tolerance is not condition-aware.
        oracle: Which oracle this case compares against. `""` (the default) means the function's
            *primary* oracle — the ordinary quantcore-vs-reference comparison `classify()` bases
            the status on. Any other value (e.g. `"quantstats"`) tags a *second-oracle
            cross-check* case: one that compares that second oracle's own output against
            `reference` (the standard definition), never against quantcore. Such a case must
            never influence `classify()`'s decision — callers filter `cases` to `oracle == ""`
            before passing them in — and for such a case `quantcore_raised` is repurposed to hold
            the *second oracle's* raised/non-finite description (there is no quantcore involved),
            while `reference_raised` stays `None` as usual.
    """

    function: str
    family: str
    length_bucket: str
    abs_err: float
    rel_err: float
    within_tolerance: bool
    quantcore_raised: str | None
    reference_raised: str | None
    quantity: str = "value"
    ill_conditioned: bool = False
    oracle: str = ""


def classify(
    cases: Sequence[CaseResult],
    *,
    has_reference: bool,
    alternative_matches: bool | None = None,
) -> ConformanceStatus:
    """Turn recorded differential-testing cases into a conformance status.

    Args:
        cases: Every `CaseResult` `harness.check()` recorded for one target, already
            filtered to `oracle == ""` (the primary quantcore-vs-reference comparison) by the
            caller — a second-oracle cross-check case (e.g. `oracle="quantstats"`) must never
            reach this function, since it compares a different pair of values entirely. Must be
            non-empty when ``has_reference`` is `True` (a real conformance run always produces at
            least one case for a target with a reference callable); ignored (may be empty) when
            ``has_reference`` is `False`, since `harness.check()` never records a case for a
            target with no reference to compare against.
        has_reference: Whether this target has an independent reference/oracle callable at all.
        alternative_matches: Whether the function's conformance module supplies a documented,
            cited `alternative_definition` that reproduces quantcore's output within tolerance on
            every case where the standard comparison failed. Computed upstream by the harness (it
            has the raw values to test the alternative definition against); `None` means no
            alternative definition was supplied. Ignored unless the standard comparison fails.

    Returns:
        The status: `unverifiable` when there is no reference to compare against; `verified` when
        quantcore never raised and every case was within tolerance; otherwise `non-standard` if
        ``alternative_matches`` is true, else `defect`. A function recording several quantities
        (e.g. a statistic and a p-value, each its own `CaseResult` with a distinct `quantity`) is
        `verified` only if every quantity in every case was within tolerance: this falls out of
        the rule applying to `cases` as a whole, without needing to group by `quantity` itself.

    Raises:
        ValueError: ``has_reference`` is `True` but ``cases`` is empty.
    """
    if not has_reference:
        return "unverifiable"

    if not cases:
        raise ValueError("classify() called with no recorded cases but has_reference=True")

    quantcore_ever_raised = any(bool(case.quantcore_raised) for case in cases)
    all_within_tolerance = not quantcore_ever_raised and all(
        case.within_tolerance for case in cases
    )

    if all_within_tolerance:
        return "verified"

    if alternative_matches:
        return "non-standard"

    return "defect"


# --- iterative estimators without a unique optimum ---------------------------------------------

type ObjectiveOutcome = Literal["win", "tie", "loss"]

# Two attained log-likelihoods within this absolute gap are practically indistinguishable fits of
# the same data, so the win/tie/loss breakdown reports them as a tie rather than a razor-thin win
# or loss; this is a reporting granularity, distinct from the caller-supplied `rtol`/`atol` margin
# below (which decides quantcore's status, not how a human-readable count is bucketed).
TIE_ABS_GAP = 1e-3

# The one-sided sign test's significance level ("at the 5% level").
SIGN_TEST_ALPHA = 0.05

# "non-standard" requires the median gap on well-specified cases to be within 1e-3 of zero;
# "defect" requires it strictly below this, in nats (the attained log-likelihoods' own units)
# rather than relative -- a plain constant, unlike the per-case rtol/atol win/tie/loss margin
# above.
NON_STANDARD_MEDIAN_GAP_THRESHOLD = 1e-3

# "defect" if quantcore fails to converge on more than 5% of well-specified cases where the
# oracle converges.
NON_CONVERGENCE_FRACTION_CAP = 0.05


@dataclasses.dataclass(frozen=True)
class ObjectiveCase:
    """One iterative-estimator fit case: quantcore's and the oracle's attained objective.

    Each side's objective is evaluated at that side's own solution. Both objectives are on a
    scale where a larger value is the better solution: every win/tie/loss rule in this module
    counts `quantcore_objective > reference_objective` as quantcore doing better. A log-likelihood
    is on that scale already; an objective that is minimised (a portfolio variance, a sum of
    squares) must be negated by whoever builds the case.

    Built by `conformance.harness.check()`'s recorded `family` and the two attained log-likelihood
    values it logs for the `log_likelihood` quantity of a fitter like `fit_garch_11`: two correct
    local optimizers can legitimately land on different local optima, so parameters are never
    compared directly; only the attained objective, under one shared likelihood, is.

    Attributes:
        family: Generator family the input came from, e.g. `"student_t3"`.
        well_specified: Whether `family` matches the fitter's own assumed distribution (e.g.
            Gaussian innovations for a Gaussian-likelihood GARCH fitter). A misspecified family
            (e.g. Student-t(3) data under a Gaussian likelihood) is expected to produce
            boundary/local-optimum disagreements that are not, on their own, evidence of a
            quantcore defect.
        quantcore_objective: quantcore's attained objective at its own solution, larger meaning
            better, or `-inf` if quantcore raised (it did not even produce a solution to
            evaluate).
        reference_objective: The oracle's attained objective at its own solution, on the same
            larger-is-better scale and under the identical shared objective function.
    """

    family: str
    well_specified: bool
    quantcore_objective: float
    reference_objective: float


def objective_outcome(case: ObjectiveCase, *, tie_abs_gap: float = TIE_ABS_GAP) -> ObjectiveOutcome:
    """Win/tie/loss of quantcore's attained objective against the oracle's, for one case.

    A reporting-granularity classification (see `TIE_ABS_GAP`), independent of
    `classify_iterative_estimator`'s decision rule.
    """
    gap = case.quantcore_objective - case.reference_objective
    if abs(gap) < tie_abs_gap:
        return "tie"
    return "win" if gap > 0.0 else "loss"


def tally_objective_outcomes(
    cases: Sequence[ObjectiveCase],
) -> dict[str, dict[ObjectiveOutcome, int]]:
    """Win/tie/loss counts per generator family, for a fitter's report and finding.

    Pure and I/O-free.

    Args:
        cases: Every recorded `ObjectiveCase` for one fitter.

    Returns:
        Mapping from family name to `{"win": n, "tie": n, "loss": n}`; a family with zero recorded
        cases is absent.
    """
    tallies: dict[str, dict[ObjectiveOutcome, int]] = {}
    for case in cases:
        family_tally = tallies.setdefault(case.family, {"win": 0, "tie": 0, "loss": 0})
        family_tally[objective_outcome(case)] += 1
    return tallies


def _well_specified_outcome(case: ObjectiveCase, *, rtol: float, atol: float) -> ObjectiveOutcome:
    """Win/tie/loss of one well-specified case for the decision (not the report) tie margin.

    Distinct from `objective_outcome` (which uses `TIE_ABS_GAP`, a reporting granularity): this
    uses the function's own declared `Tolerance`. `classify_iterative_estimator`'s tie band and
    `Tolerance` for the same function are ONE notion of "differs": a case is a loss only if
    `|gap| > atol + rtol*|ref|` under the declared tolerance; there is no second decision
    tolerance.
    """
    margin = atol + rtol * abs(case.reference_objective)
    gap = case.quantcore_objective - case.reference_objective
    if abs(gap) <= margin:
        return "tie"
    return "win" if gap > 0.0 else "loss"


@dataclasses.dataclass(frozen=True)
class IterativeEstimatorDecision:
    """The full outcome of `classify_iterative_estimator`: the status and the evidence behind it.

    Attributes:
        status: The resolved `ConformanceStatus`.
        wins: Well-specified cases where quantcore's attained objective beat the oracle's by more
            than the decision tolerance (the caller-supplied ``rtol``/``atol``, the function's own
            declared `Tolerance`).
        ties: Well-specified cases within the decision tolerance of the oracle -- excluded from
            the sign test.
        losses: Well-specified cases where quantcore's attained objective was worse than the
            oracle's by more than the decision tolerance.
        p_value: The one-sided binomial sign test's p-value (`H0: p_loss = 0.5`, testing `losses`
            against `wins + losses`), or `1.0` if there were no decided (non-tied) cases at all.
        median_gap: The median of `quantcore_objective - reference_objective` across every
            well-specified case (wins, ties, and losses alike).
        non_convergence_fraction: Fraction of well-specified cases, among those where the oracle
            converged, where quantcore did not (`quantcore_objective` non-finite).
    """

    status: ConformanceStatus
    wins: int
    ties: int
    losses: int
    p_value: float
    median_gap: float
    non_convergence_fraction: float


def classify_iterative_estimator(
    cases: Sequence[ObjectiveCase],
    *,
    rtol: float,
    atol: float,
    alpha: float = SIGN_TEST_ALPHA,
    non_standard_median_gap_threshold: float = NON_STANDARD_MEDIAN_GAP_THRESHOLD,
    non_convergence_fraction_cap: float = NON_CONVERGENCE_FRACTION_CAP,
) -> IterativeEstimatorDecision:
    """Classify an iterative estimator (a GARCH-family fitter, HMM EM, any non-convex MLE).

    Classifies an iterative estimator without a unique optimum: the comparison is on the
    attained objective (not raw parameters, which two correct optimizers may legitimately
    disagree on), decided by a one-sided binomial sign test plus a median-gap threshold on
    well-specified cases only. Misspecified-family cases (e.g. Student-t(3) data under a Gaussian
    likelihood) are never passed to the sign test or the median -- they are reported
    (`tally_objective_outcomes`) but never decide the status.

    ``rtol``/``atol`` are the SAME `Tolerance` declared for this function in
    `conformance/tolerances.py` -- there is no second, harness-only decision tolerance. A case is
    a win/loss only if `|gap| > atol + rtol*|ref|`; otherwise it is a tie excluded from the sign
    test.

    Args:
        cases: Every recorded `ObjectiveCase` for one fitter, across every declared family.
        rtol: Relative margin (the function's declared `Tolerance.rtol`) below which a
            well-specified case's gap counts as a win/loss rather than a tie excluded from the
            sign test.
        atol: Absolute margin (the function's declared `Tolerance.atol`), added to the relative
            term above to form the same `atol + rtol*|ref|` closeness rule used everywhere else.
        alpha: The sign test's significance level.
        non_standard_median_gap_threshold: How far below zero the median gap may fall and still
            count as "the same likelihood ridge" (`non-standard`) rather than `defect`.
        non_convergence_fraction_cap: The maximum tolerated fraction of well-specified,
            oracle-converged cases where quantcore fails to converge before that alone is a
            `defect`.

    Returns:
        An `IterativeEstimatorDecision` with the resolved status plus every count,
        the sign test's p-value, and the median gap:

        - `verified`: the sign test does not reject `H0: p_loss = 0.5` in quantcore's
          disfavour (`p_value > alpha`) AND the median gap is `>= -rtol`.
        - `defect`: quantcore fails to converge on more than
          ``non_convergence_fraction_cap`` of well-specified, oracle-converged cases; or the sign
          test rejects `H0` AND the median gap is below `-non_standard_median_gap_threshold`.
        - `non-standard`: everything else (the sign test rejects `H0`, or `verified`'s
          median-gap condition alone fails, but the median gap stays within
          ``non_standard_median_gap_threshold`` of zero) -- "different optimiser", finding
          written.

    Raises:
        ValueError: ``cases`` is empty, or contains no well-specified case.
    """
    if not cases:
        raise ValueError("classify_iterative_estimator() called with no recorded cases")
    well_specified_cases = [c for c in cases if c.well_specified]
    if not well_specified_cases:
        raise ValueError("classify_iterative_estimator() called with no well-specified cases")

    wins = ties = losses = 0
    for case in well_specified_cases:
        outcome = _well_specified_outcome(case, rtol=rtol, atol=atol)
        if outcome == "win":
            wins += 1
        elif outcome == "tie":
            ties += 1
        else:
            losses += 1

    decided = wins + losses
    p_value = binomtest(losses, decided, 0.5, alternative="greater").pvalue if decided else 1.0
    median_gap = statistics.median(
        case.quantcore_objective - case.reference_objective for case in well_specified_cases
    )

    oracle_converged = [c for c in well_specified_cases if math.isfinite(c.reference_objective)]
    non_converged = [c for c in oracle_converged if not math.isfinite(c.quantcore_objective)]
    non_convergence_fraction = (
        len(non_converged) / len(oracle_converged) if oracle_converged else 0.0
    )

    sign_test_rejects = p_value <= alpha

    if non_convergence_fraction > non_convergence_fraction_cap:
        status: ConformanceStatus = "defect"
    elif not sign_test_rejects and median_gap >= -rtol:
        status = "verified"
    elif sign_test_rejects and median_gap < -non_standard_median_gap_threshold:
        status = "defect"
    else:
        status = "non-standard"

    return IterativeEstimatorDecision(
        status=status,
        wins=wins,
        ties=ties,
        losses=losses,
        p_value=p_value,
        median_gap=median_gap,
        non_convergence_fraction=non_convergence_fraction,
    )
