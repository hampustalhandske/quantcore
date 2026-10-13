"""Tests for conformance.classify: the pure recorded-cases -> status classifiers.

`classify` takes a sequence of `CaseResult` plus the facts its rule needs that are not part of
any single case (whether a reference exists at all, whether an alternative definition
reproduced quantcore's output) and returns a `ConformanceStatus`.
`classify_iterative_estimator` does the same for a fitter judged on its attained objective.
"""

import conformance.classify as classify_module
import pytest
from conformance.classify import (
    CaseResult,
    ObjectiveCase,
    classify,
    classify_iterative_estimator,
    objective_outcome,
    tally_objective_outcomes,
)


def _case(
    within_tolerance: bool = True,
    quantcore_raised: bool = False,
    *,
    reference_raised: bool = False,
    abs_err: float = 0.0,
    rel_err: float = 0.0,
    function: str = "sharpe_ratio",
    family: str = "normal",
    length_bucket: str = "medium",
    quantity: str = "value",
) -> CaseResult:
    return CaseResult(
        function=function,
        family=family,
        length_bucket=length_bucket,
        abs_err=abs_err,
        rel_err=rel_err,
        within_tolerance=within_tolerance,
        quantcore_raised=quantcore_raised,
        reference_raised=reference_raised,
        quantity=quantity,
    )


def test_classify_verified_when_all_cases_within_tolerance_and_no_quantcore_exception() -> None:
    """All cases within tolerance with no quantcore exception classifies as verified."""
    cases = [_case() for _ in range(50)]
    assert classify(cases, has_reference=True) == "verified"


def test_classify_prevents_verified_when_quantcore_raises_even_if_all_within_tolerance() -> None:
    """A quantcore exception prevents verified even if every recorded case is within tolerance."""
    cases = [*(_case() for _ in range(20)), _case(quantcore_raised=True)]
    assert classify(cases, has_reference=True) != "verified"


def test_classify_one_bad_case_among_thousands_flips_verified() -> None:
    """A single out-of-tolerance case among thousands is enough to lose verified status."""
    good_cases = [_case() for _ in range(5000)]
    assert classify(good_cases, has_reference=True) == "verified"

    bad_cases = [*good_cases[:-1], _case(within_tolerance=False, abs_err=1.0, rel_err=1.0)]
    assert classify(bad_cases, has_reference=True) != "verified"


def test_classify_non_standard_when_alternative_definition_matches() -> None:
    """Cases failing standard comparison but matching alternative_definition are non-standard."""
    cases = [*(_case() for _ in range(10)), _case(within_tolerance=False)]
    assert classify(cases, has_reference=True, alternative_matches=True) == "non-standard"


def test_classify_raises_on_empty_cases_when_a_reference_exists() -> None:
    """A target with a reference but no recorded case cannot be classified at all."""
    with pytest.raises(ValueError, match="no recorded cases"):
        classify([], has_reference=True)


def test_classify_defect_when_no_alternative_definition_matches() -> None:
    """Cases failing standard comparison with no matching alternative_definition classify defect."""
    cases = [*(_case() for _ in range(10)), _case(within_tolerance=False)]
    assert classify(cases, has_reference=True, alternative_matches=False) == "defect"


def test_classify_requires_every_quantity_within_tolerance() -> None:
    """A function recording several quantities is verified only if every quantity passes.

    `engle_granger_test` records one case per input as two `CaseResult`s, `statistic` and
    `p_value`. If the statistic always agrees with the oracle but the p_value never does, the
    function as a whole must not classify as verified.
    """
    statistic_cases = [_case(quantity="statistic") for _ in range(20)]
    passing_p_value_cases = [_case(quantity="p_value") for _ in range(20)]
    assert classify(statistic_cases + passing_p_value_cases, has_reference=True) == "verified"

    failing_p_value_cases = [
        _case(quantity="p_value", within_tolerance=False, abs_err=1.0, rel_err=1.0)
        for _ in range(20)
    ]
    assert classify(statistic_cases + failing_p_value_cases, has_reference=True) != "verified"


def test_classify_unverifiable_without_reference() -> None:
    """With no reference the status is unverifiable, whatever cases are (or are not) passed."""
    assert classify([], has_reference=False) == "unverifiable"
    assert classify([_case(within_tolerance=False)], has_reference=False) == "unverifiable"


# --- iterative estimators without a unique optimum -----------------------------------------------


def _obj(
    *,
    well_specified: bool = True,
    quantcore_objective: float = 100.0,
    reference_objective: float = 100.0,
    family: str = "normal",
) -> ObjectiveCase:
    return ObjectiveCase(
        family=family,
        well_specified=well_specified,
        quantcore_objective=quantcore_objective,
        reference_objective=reference_objective,
    )


def test_objective_outcome_win_tie_loss() -> None:
    """win/tie/loss by the |gap| < 1e-3 reporting granularity."""
    assert objective_outcome(_obj(quantcore_objective=101.0, reference_objective=100.0)) == "win"
    assert objective_outcome(_obj(quantcore_objective=99.0, reference_objective=100.0)) == "loss"
    assert objective_outcome(_obj(quantcore_objective=100.0001, reference_objective=100.0)) == "tie"


def test_tally_objective_outcomes_groups_by_family() -> None:
    cases = [
        _obj(family="normal", quantcore_objective=101.0, reference_objective=100.0),
        _obj(family="normal", quantcore_objective=100.0, reference_objective=100.0),
        _obj(family="student_t3", quantcore_objective=90.0, reference_objective=100.0),
    ]
    tallies = tally_objective_outcomes(cases)
    assert tallies["normal"] == {"win": 1, "tie": 1, "loss": 0}
    assert tallies["student_t3"] == {"win": 0, "tie": 0, "loss": 1}


def _win(*, gap: float = 1.0, reference_objective: float = 100.0) -> ObjectiveCase:
    return _obj(
        well_specified=True,
        reference_objective=reference_objective,
        quantcore_objective=reference_objective + gap,
    )


def _tie(*, reference_objective: float = 100.0) -> ObjectiveCase:
    # Half the decision margin (rtol * reference_objective): a tie, excluded from the sign test.
    return _obj(
        well_specified=True,
        reference_objective=reference_objective,
        quantcore_objective=reference_objective + 5e-5,
    )


def _loss(*, gap: float = -1.0, reference_objective: float = 100.0) -> ObjectiveCase:
    return _obj(
        well_specified=True,
        reference_objective=reference_objective,
        quantcore_objective=reference_objective + gap,
    )


def test_classify_iterative_estimator_verified_57_20_2() -> None:
    """57 wins / 20 ties / 2 losses: sign test does not reject, median gap is positive."""
    cases = [
        *(_win() for _ in range(57)),
        *(_tie() for _ in range(20)),
        *(_loss() for _ in range(2)),
    ]
    decision = classify_iterative_estimator(cases, rtol=1e-6, atol=0.0)
    assert decision.wins == 57
    assert decision.ties == 20
    assert decision.losses == 2
    assert decision.p_value > 0.05
    assert decision.status == "verified"


def test_classify_iterative_estimator_non_standard_10_0_30_small_median_gap() -> None:
    """10 wins / 0 ties / 30 losses, median gap -1e-4 (within 1e-3 of zero) -> non-standard."""
    cases = [
        *(_win(gap=5.0, reference_objective=1.0) for _ in range(10)),
        *(_loss(gap=-1e-4, reference_objective=1.0) for _ in range(30)),
    ]
    decision = classify_iterative_estimator(cases, rtol=1e-6, atol=0.0)
    assert decision.wins == 10
    assert decision.ties == 0
    assert decision.losses == 30
    assert decision.p_value <= 0.05
    assert decision.median_gap == pytest.approx(-1e-4)
    assert decision.status == "non-standard"


def test_classify_iterative_estimator_defect_10_0_30_large_median_gap() -> None:
    """Same 10/0/30 split, median gap -0.05 (below the -1e-3 non-standard threshold) -> defect."""
    cases = [
        *(_win(gap=5.0, reference_objective=1.0) for _ in range(10)),
        *(_loss(gap=-0.05, reference_objective=1.0) for _ in range(30)),
    ]
    decision = classify_iterative_estimator(cases, rtol=1e-6, atol=0.0)
    assert decision.p_value <= 0.05
    assert decision.median_gap == pytest.approx(-0.05)
    assert decision.status == "defect"


def test_classify_iterative_estimator_defect_on_non_convergence_despite_favourable_sign_test() -> (
    None
):
    """6% non-convergence forces defect even though the sign test alone would not reject.

    90 wins, 4 ties, 6 losses (the non-converged cases): the sign test alone would not reject
    `H0`, and the median gap is positive, but the non-convergence gate overrides both.
    """
    cases = [
        *(_win() for _ in range(90)),
        *(_tie() for _ in range(4)),
        *(
            _obj(well_specified=True, quantcore_objective=float("-inf"), reference_objective=100.0)
            for _ in range(6)
        ),
    ]
    decision = classify_iterative_estimator(cases, rtol=1e-6, atol=0.0)
    assert decision.non_convergence_fraction == pytest.approx(0.06)
    assert decision.p_value > 0.05
    assert decision.median_gap > 0.0
    assert decision.status == "defect"


def test_classify_iterative_estimator_misspecified_losses_never_change_status() -> None:
    """Misspecified-family losses, however bad, never enter the sign test, median, or status."""
    well_specified_cases = [
        *(_win() for _ in range(57)),
        *(_tie() for _ in range(20)),
        *(_loss() for _ in range(2)),
    ]
    baseline = classify_iterative_estimator(well_specified_cases, rtol=1e-6, atol=0.0)
    assert baseline.status == "verified"

    misspecified_disaster = [
        _obj(well_specified=False, quantcore_objective=float("-inf"), reference_objective=100.0)
        for _ in range(500)
    ]
    with_misspecified = classify_iterative_estimator(
        well_specified_cases + misspecified_disaster, rtol=1e-6, atol=0.0
    )

    assert with_misspecified.status == baseline.status
    assert with_misspecified.wins == baseline.wins
    assert with_misspecified.ties == baseline.ties
    assert with_misspecified.losses == baseline.losses
    assert with_misspecified.p_value == baseline.p_value
    assert with_misspecified.median_gap == baseline.median_gap


def test_classify_iterative_estimator_has_no_module_level_decision_rtol_constant() -> None:
    """There is no second `DECISION_RTOL`: no module-level constant by that name exists."""
    assert not hasattr(classify_module, "DECISION_RTOL")


def test_classify_iterative_estimator_margin_uses_atol_plus_rtol_times_ref() -> None:
    """The win/tie/loss margin is `atol + rtol*|ref|`, matching `Tolerance.is_close` exactly.

    With `reference_objective` near zero, a pure-rtol margin would be ~0, so `atol` alone must be
    what turns a tiny gap into a tie.
    """
    tie_case = _obj(well_specified=True, reference_objective=1e-9, quantcore_objective=1e-9 + 5e-9)
    decision = classify_iterative_estimator([tie_case], rtol=1e-6, atol=1e-8)
    assert decision.ties == 1
    assert decision.wins == 0
    assert decision.losses == 0


def test_classify_iterative_estimator_atol_does_not_mask_a_real_loss() -> None:
    """A gap larger than `atol + rtol*|ref|` is still a loss, not swallowed by a loose atol."""
    loss_case = _obj(well_specified=True, reference_objective=1e-9, quantcore_objective=1e-9 - 5e-7)
    decision = classify_iterative_estimator([loss_case], rtol=1e-6, atol=1e-8)
    assert decision.losses == 1
    assert decision.ties == 0


def test_classify_iterative_estimator_raises_on_empty_cases() -> None:
    with pytest.raises(ValueError, match="no recorded cases"):
        classify_iterative_estimator([], rtol=1e-6, atol=0.0)


def test_classify_iterative_estimator_raises_when_no_well_specified_case() -> None:
    with pytest.raises(ValueError, match="no well-specified cases"):
        classify_iterative_estimator([_obj(well_specified=False)], rtol=1e-6, atol=0.0)
