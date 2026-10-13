"""Tests for conformance.build_report: the builder of the conformance map, report and findings.

Everything here runs on synthetic `CaseResult`/`ReproDetail` records, toy targets and temporary
directories, so no test needs the (slow, hypothesis-driven) conformance suite to have run. The
few tests that read the real registry check only that the builder's own tables (coverage sources,
repro configurations, regeneration recipes, iterative estimators) agree with the registered
targets and with the generators they describe.
"""

import contextlib
import dataclasses
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# The builder imports every tier test module, and through them every library of the `oracle`
# extra; skip this module without them.
pytest.importorskip("hypothesis")
pytest.importorskip("pandas")
pytest.importorskip("statsmodels")
pytest.importorskip("arch")
pytest.importorskip("sklearn")
pytest.importorskip("linearmodels")
pytest.importorskip("empyrical")
pytest.importorskip("quantstats")
pytest.importorskip("QuantLib")
pytest.importorskip("vollib")
pytest.importorskip("hmmlearn")
pytest.importorskip("pykalman")
pytest.importorskip("cvxpy")

import conformance.build_report as build_report_module
from conformance import test_tier1_statistics_extra, test_tier2_filtering
from conformance.build_report import (
    COVERAGE_SOURCES,
    DECLARED_COVERAGE,
    GENERATED_FILE_COMMENT,
    ITERATIVE_ESTIMATORS,
    REGEN_RECIPES,
    REPRO_CONFIGS,
    ConformanceMap,
    DeclaredCoverage,
    IterativeEstimator,
    QuantityErrors,
    ReproCallConfig,
    ReproRunResult,
    TargetRecord,
    ToleranceRecord,
    _assert_no_extreme_magnitude,
    _total_numbers,
    aggregate_errors,
    build_conformance_map,
    configuration_errors,
    conformance_map_from_json,
    conformance_map_to_json,
    decision_diff,
    declared_coverage_from,
    evidence_for_quantity,
    explore_mismatches,
    headline_case_tolerance_violation,
    ill_conditioned_counts,
    ill_conditioned_gate_failures,
    incomplete_targets,
    iterative_estimator_decision,
    matched_pairs,
    missing_coverage,
    missing_coverage_by_target,
    missing_finding_sources,
    objective_cases_for,
    oracle_cross_check_rows,
    oracle_disagreement_examples,
    oracle_extra_distributions,
    oracle_versions,
    oriented_objective,
    parse_repro_stdout,
    parse_repro_summary,
    per_function_counts,
    reconstruct_args,
    remove_stale_findings,
    render_conformance_md,
    render_finding,
    render_minimal_repro,
    repro_still_mismatches,
    round_significant,
    run_repro_subprocess,
    select_shortest_failing_detail,
    unreachable_declared_buckets,
    unused_finding_sources,
    verified_functions_from_map,
    verify_repro,
)
from conformance.classify import CaseResult
from conformance.compare import ReportOnlyScalarComparator, ScalarComparator, Tolerance, registry
from conformance.findings import FindingSource
from conformance.generators import (
    COINTEGRATION_FAMILIES,
    RETURN_FAMILIES,
    cointegration_pair,
    returns_series,
)
from conformance.harness import ReproDetail, _repr_inputs
from conformance.tolerances import TOLERANCES
from hypothesis import given, settings


def _detail(
    *,
    abs_err: float | None = 1e-6,
    rel_err: float | None = 1e-6,
    expected_abs: float = 1.0,
    quantity: str = "value",
    quantcore_value: str | None = "1.0",
    reference_value: str = "1.0",
) -> ReproDetail:
    return ReproDetail(
        function="sharpe_ratio",
        family="normal",
        quantity=quantity,
        seed=0,
        abs_err=abs_err,
        rel_err=rel_err,
        expected_abs=expected_abs,
        quantcore_value=quantcore_value,
        reference_value=reference_value,
        inputs_repr="(...)",
    )


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (0.0, 3, 0.0),
        (1234.5, 3, 1230.0),
        (0.000123456, 3, 0.000123),
        (9.999e-9, 2, 1.0e-8),
        (-5.0, 3, -5.0),
    ],
    ids=["zero", "large-magnitude", "small-magnitude", "rounds-up-across-exponent", "negative"],
)
def test_round_significant_rounds_to_given_digits(value, digits, expected) -> None:
    """round_significant keeps only the given number of significant decimal digits."""
    assert round_significant(value, digits) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_round_significant_passes_through_non_finite(value: float) -> None:
    """A non-finite value has no meaningful significant digit and is returned unchanged."""
    result = round_significant(value)
    assert math.isnan(result) if math.isnan(value) else result == value


def test_aggregate_errors_excludes_near_zero_expected_from_max_rel_error() -> None:
    """A case whose oracle value is ~0 contributes to max_abs_error but not to max_rel_error."""
    near_zero_case = _detail(abs_err=1e-4, rel_err=1e296, expected_abs=0.0)
    normal_case = _detail(abs_err=1e-9, rel_err=2e-9, expected_abs=1.0)
    max_abs, max_rel = aggregate_errors([near_zero_case, normal_case])
    assert max_abs == pytest.approx(1e-4)
    assert max_rel == pytest.approx(2e-9)


def test_aggregate_errors_returns_none_when_no_case_qualifies() -> None:
    """Both aggregates are None when there is nothing to aggregate over."""
    assert aggregate_errors([]) == (None, None)

    quantcore_raised_case = _detail(abs_err=None, rel_err=None, expected_abs=1.0)
    assert aggregate_errors([quantcore_raised_case]) == (None, None)

    all_near_zero = [_detail(abs_err=1e-4, rel_err=1e296, expected_abs=0.0)]
    max_abs, max_rel = aggregate_errors(all_near_zero)
    assert max_abs == pytest.approx(1e-4)
    assert max_rel is None


def test_aggregate_errors_rounds_to_three_significant_digits() -> None:
    """The returned magnitudes are rounded to 3 significant digits, not raw floats."""
    case = _detail(abs_err=1.234567e-8, rel_err=9.87654e-11, expected_abs=1.0)
    max_abs, max_rel = aggregate_errors([case])
    assert max_abs == pytest.approx(1.23e-8)
    assert max_rel == pytest.approx(9.88e-11)


def test_assert_no_extreme_magnitude_passes_on_clean_content(tmp_path) -> None:
    """Content with no e+296 substring raises nothing."""
    _assert_no_extreme_magnitude("max rel err: 1.23e-08", tmp_path / "clean.md")


def test_assert_no_extreme_magnitude_fails_on_leaked_magnitude(tmp_path) -> None:
    """Content containing a leaked ~1e+296 magnitude fails loudly, naming the offending path."""
    path = tmp_path / "leaked.md"
    with pytest.raises(AssertionError, match=r"leaked\.md"):
        _assert_no_extreme_magnitude("max rel err: 4.80e+296", path)


def _case(
    *,
    function: str = "sharpe_ratio",
    family: str = "normal",
    length_bucket: str = "short",
    within_tolerance: bool = True,
) -> CaseResult:
    return CaseResult(
        function=function,
        family=family,
        length_bucket=length_bucket,
        abs_err=1e-9,
        rel_err=1e-9,
        within_tolerance=within_tolerance,
        quantcore_raised=None,
        reference_raised=None,
    )


def test_per_function_counts_tallies_cases_per_family_and_bucket() -> None:
    """Each recorded case increments its target's family count and bucket count by one."""
    cases = [
        _case(function="sharpe_ratio", family="normal", length_bucket="short"),
        _case(function="sharpe_ratio", family="normal", length_bucket="medium"),
        _case(function="sharpe_ratio", family="ar1", length_bucket="short"),
        _case(function="maximum_drawdown", family="normal", length_bucket="long"),
    ]
    counts = per_function_counts(cases)
    sharpe_families, sharpe_buckets = counts["sharpe_ratio"]
    assert sharpe_families == {"normal": 2, "ar1": 1}
    assert sharpe_buckets == {"short": 2, "medium": 1}
    drawdown_families, drawdown_buckets = counts["maximum_drawdown"]
    assert drawdown_families == {"normal": 1}
    assert drawdown_buckets == {"long": 1}


def test_per_function_counts_empty_for_no_cases() -> None:
    """A target with no recorded cases at all is simply absent from the result."""
    assert per_function_counts([]) == {}


_DECLARED = {
    "sharpe_ratio": DeclaredCoverage(families=("normal", "ar1"), buckets=("short", "medium"))
}


def test_missing_coverage_empty_when_every_declared_cell_has_a_case() -> None:
    """No missing-coverage lines when every declared family and bucket has at least one case."""
    counts = {"sharpe_ratio": ({"normal": 3, "ar1": 1}, {"short": 2, "medium": 2})}
    assert missing_coverage(_DECLARED, counts) == []


@pytest.mark.parametrize(
    ("counts", "expected_substring"),
    [
        ({"sharpe_ratio": ({"ar1": 1}, {"short": 1, "medium": 1})}, "family 'normal'"),
        ({"sharpe_ratio": ({"normal": 1, "ar1": 1}, {"short": 1})}, "length bucket 'medium'"),
        ({}, "no recorded cases at all"),
    ],
    ids=["empty-family", "empty-bucket", "target-missing-entirely"],
)
def test_missing_coverage_reports_the_empty_cell(counts, expected_substring: str) -> None:
    """A declared family or bucket with zero recorded cases produces a line naming it."""
    missing = missing_coverage(_DECLARED, counts)
    assert any("sharpe_ratio" in line and expected_substring in line for line in missing)


def test_headline_case_tolerance_violation_none_when_genuinely_outside_tolerance() -> None:
    """A case correctly flagged out-of-tolerance, with a real gap, passes the recheck silently."""
    tolerance = Tolerance(rtol=1e-6, atol=1e-8, justification="test")
    case = _case(within_tolerance=False)
    detail = _detail(quantcore_value="1.001", reference_value="1.0")
    assert headline_case_tolerance_violation(case, detail, tolerance) is None


def test_headline_case_tolerance_violation_catches_within_tolerance_flag() -> None:
    """`within_tolerance=True` alone is refused (the headline-case gate)."""
    tolerance = Tolerance(rtol=1e-6, atol=1e-8, justification="test")
    case = _case(within_tolerance=True)
    detail = _detail(quantcore_value="1.0", reference_value="1.0")
    violation = headline_case_tolerance_violation(case, detail, tolerance)
    assert violation is not None
    assert "within_tolerance=True" in violation


def test_headline_case_tolerance_violation_recomputes_from_raw_values() -> None:
    """Catches a stale/inconsistent flag.

    Recorded within_tolerance=False but the raw numbers themselves are actually within the
    declared tolerance (an 8e-8 gap inside a 4.7e-7 tolerance).
    """
    tolerance = Tolerance(rtol=1e-6, atol=1e-8, justification="test")
    case = _case(within_tolerance=False)
    detail = _detail(quantcore_value="0.46000008", reference_value="0.46")
    violation = headline_case_tolerance_violation(case, detail, tolerance)
    assert violation is not None
    assert "within the declared tolerance on recheck" in violation


def test_headline_case_tolerance_violation_none_when_quantcore_raised() -> None:
    """Quantcore raising outright has nothing numeric to recheck -- an unambiguous defect."""
    tolerance = Tolerance(rtol=1e-6, atol=1e-8, justification="test")
    case = _case(within_tolerance=False)
    detail = _detail(quantcore_value=None, reference_value="1.0")
    assert headline_case_tolerance_violation(case, detail, tolerance) is None


def _ill_case(*, function: str = "sharpe_ratio", ill: bool = False, oracle: str = "") -> CaseResult:
    return CaseResult(
        function=function,
        family="near_constant",
        length_bucket="short",
        abs_err=0.0,
        rel_err=0.0,
        within_tolerance=True,
        quantcore_raised=None,
        reference_raised=None,
        ill_conditioned=ill,
        oracle=oracle,
    )


def test_ill_conditioned_counts_tallies_per_function() -> None:
    """`(ill_conditioned_count, total_count)` per target, over the primary comparison only."""
    cases = [
        _ill_case(ill=True),
        _ill_case(ill=False),
        _ill_case(ill=False),
        _ill_case(function="maximum_drawdown", ill=False),
        # A second-oracle cross-check case never counts toward either target's tally.
        _ill_case(ill=True, oracle="quantstats"),
    ]
    counts = ill_conditioned_counts(cases)
    assert counts["sharpe_ratio"] == (1, 3)
    assert counts["maximum_drawdown"] == (0, 1)


def test_ill_conditioned_gate_failures_below_threshold() -> None:
    """A target whose well-conditioned fraction falls below the minimum is reported."""
    counts = {"sharpe_ratio": (15, 100), "maximum_drawdown": (0, 50)}
    failures = ill_conditioned_gate_failures(counts, min_well_conditioned_fraction=0.9)
    assert len(failures) == 1
    assert "sharpe_ratio" in failures[0]
    assert "15/100" in failures[0]


def test_ill_conditioned_gate_failures_empty_when_every_target_meets_the_threshold() -> None:
    counts = {"sharpe_ratio": (5, 100), "maximum_drawdown": (0, 50)}
    assert ill_conditioned_gate_failures(counts, min_well_conditioned_fraction=0.9) == []


def test_ill_conditioned_gate_failures_ignores_a_target_with_zero_cases() -> None:
    assert ill_conditioned_gate_failures({"f": (0, 0)}, min_well_conditioned_fraction=0.9) == []


def test_explore_mismatches_excludes_ill_conditioned_cases() -> None:
    """An `ill_conditioned` case is excluded from the decision, so it is never a mismatch."""
    cases = [
        CaseResult(
            function="sharpe_ratio",
            family="near_constant",
            length_bucket="short",
            abs_err=2e-9,
            rel_err=2e-9,
            within_tolerance=False,
            quantcore_raised=None,
            reference_raised=None,
            ill_conditioned=True,
        )
    ]
    details = [_rd(function="sharpe_ratio", family="near_constant")]
    mismatches = explore_mismatches(cases, details, verified_functions={"sharpe_ratio"})
    assert mismatches == []


def _map(*, targets: dict[str, TargetRecord] | None = None) -> ConformanceMap:
    return ConformanceMap(
        schema_version=1,
        quantcore_version="1.0.0",
        oracle_versions={"statsmodels": "0.14.0"},
        hypothesis_profile="ci",
        targets=targets if targets is not None else {"sharpe_ratio": _entry()},
    )


def _entry(
    *,
    status: str = "verified",
    rtol: float = 1e-10,
    atol: float = 1e-9,
    justification: str = "closed-form",
    finding: str | None = None,
    cases: int = 200,
    max_abs_error: float | None = 1e-12,
    max_rel_error: float | None = 1e-12,
    oracle: str = "empyrical-reloaded",
    by_quantity: dict[str, QuantityErrors] | None = None,
    reason: str | None = None,
) -> TargetRecord:
    return TargetRecord(
        status=status,
        oracle=oracle,
        cases=cases,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        by_quantity=by_quantity if by_quantity is not None else {},
        tolerance=ToleranceRecord(rtol=rtol, atol=atol, justification=justification),
        reason=reason if reason is not None else ("" if status == "verified" else "reason"),
        finding=finding,
    )


# ---------------------------------------------------------------------------
# build_conformance_map on toy targets
# ---------------------------------------------------------------------------


def _fake_target(*, quantcore=None, reference=None, oracle: str = "test oracle", compare=None):
    """A minimal stand-in with the attributes `build_report` reads off a `ConformanceTarget`.

    Not a real `ConformanceTarget`: constructing one registers it in the real, process-global
    registry (`conformance.compare.registry()`), which would leak a throwaway test-only target
    into every other test that walks the registry for the rest of the session. These tests pass
    a dict of these as `targets`, or monkeypatch `build_report_module.registry` to return one.
    """
    return SimpleNamespace(
        quantcore=quantcore,
        reference=reference,
        oracle=oracle,
        compare=compare if compare is not None else ScalarComparator(),
    )


def _noop(*args, **kwargs):
    return 0.0


def test_build_conformance_map_reads_the_registry_when_no_targets_are_given(monkeypatch) -> None:
    """With `targets=None` every registered target is classified."""
    fake_registry = {"select_hmm_n_states": _fake_target(quantcore=_noop)}
    monkeypatch.setattr(build_report_module, "registry", lambda: fake_registry)

    conformance_map = build_conformance_map(cases=[], details=[], hypothesis_profile="ci")

    assert set(conformance_map.targets) == {"select_hmm_n_states"}


def test_build_conformance_map_target_without_an_oracle_is_unverifiable() -> None:
    """A target with no `reference` is `unverifiable`, with a stated reason and no tolerance.

    Nothing was ever compared, so there is no case count, no error figure and no tolerance to
    record; the reason says why the status is not `verified`.
    """
    targets = {"select_hmm_n_states": _fake_target(quantcore=_noop)}

    conformance_map = build_conformance_map([], [], "ci", targets)

    entry = conformance_map.targets["select_hmm_n_states"]
    assert entry.status == "unverifiable"
    assert entry.reason == build_report_module.NO_ORACLE_REASON
    assert entry.cases == 0
    assert entry.max_abs_error is None
    assert entry.max_rel_error is None
    assert entry.by_quantity == {}
    assert entry.tolerance is None
    assert entry.finding is None


def test_build_conformance_map_rejects_a_target_without_a_quantcore_callable() -> None:
    """A target with nothing to judge is a configuration error naming every such target."""
    targets = {
        "ulcer_index": _fake_target(reference=_noop),
        "kpss": _fake_target(reference=_noop),
        "sharpe_ratio": _fake_target(quantcore=_noop, reference=_noop),
    }
    with pytest.raises(ValueError, match=r"without a quantcore callable.*kpss, ulcer_index"):
        build_conformance_map([], [], "ci", targets)


def test_build_conformance_map_report_only_comparator_is_never_verified(monkeypatch) -> None:
    """A target whose sole comparator never fails `assert_close` cannot be classified `verified`.

    `select_arima_order`/`select_var_lag_order` register `ReportOnlyScalarComparator` as their
    *entire* comparator (no decisive quantity alongside it), so `harness.check()` records
    `within_tolerance=True` on every case regardless of actual agreement, and `classify()` would
    otherwise call them `verified` even when the recorded `max_rel_error` shows real
    disagreement. `build_conformance_map` must recognise such a target and record it as
    `unverifiable`, however clean the recorded cases look.
    """
    monkeypatch.setitem(
        build_report_module.TOLERANCES,
        "toy_order_selection",
        Tolerance(rtol=1e-10, atol=1e-10, justification="test"),
    )
    targets = {
        "toy_order_selection": _fake_target(
            quantcore=_noop, reference=_noop, compare=ReportOnlyScalarComparator()
        )
    }
    cases = [
        CaseResult(
            function="toy_order_selection",
            family="ar2",
            length_bucket="medium",
            abs_err=302.0,
            rel_err=302.0,
            within_tolerance=True,  # the report-only comparator never sets this to False
            quantcore_raised=None,
            reference_raised=None,
        )
    ]

    conformance_map = build_conformance_map(cases, [], "ci", targets)

    entry = conformance_map.targets["toy_order_selection"]
    assert entry.status == "unverifiable"
    assert entry.reason == build_report_module.REPORT_ONLY_REASON
    assert entry.tolerance == ToleranceRecord(rtol=1e-10, atol=1e-10, justification="test")


def test_build_conformance_map_target_with_an_oracle_and_zero_cases_fails_the_build() -> None:
    """A target with an oracle and no recorded case raises instead of receiving a status."""
    targets = {"sharpe_ratio": _fake_target(quantcore=_noop, reference=_noop)}

    with pytest.raises(ValueError, match="no recorded cases"):
        build_conformance_map([], [], "ci", targets)


def _toy_run(
    name: str, *, within: list[bool], ill: list[bool] | None = None
) -> tuple[list[CaseResult], list[ReproDetail]]:
    """One case and one matching detail per entry of ``within``, for a single-quantity target."""
    ill = ill if ill is not None else [False] * len(within)
    cases = [
        CaseResult(
            function=name,
            family="normal",
            length_bucket="short",
            abs_err=0.0 if ok else 0.5,
            rel_err=0.0 if ok else 0.5,
            within_tolerance=ok,
            quantcore_raised=None,
            reference_raised=None,
            ill_conditioned=flag,
        )
        for ok, flag in zip(within, ill, strict=True)
    ]
    details = [
        ReproDetail(
            function=name,
            family="normal",
            quantity="value",
            seed=index,
            abs_err=0.0 if ok else 0.5,
            rel_err=0.0 if ok else 0.5,
            expected_abs=1.0,
            quantcore_value="1.0" if ok else "1.5",
            reference_value="1.0",
            inputs_repr="(array([1.0, 2.0]),)",
        )
        for index, ok in enumerate(within)
    ]
    return cases, details


@pytest.fixture
def toy_tolerance(monkeypatch) -> Tolerance:
    """Declare a tolerance for the toy target `toy_ratio` for the duration of one test."""
    tolerance = Tolerance(rtol=1e-6, atol=1e-8, justification="toy tolerance")
    monkeypatch.setitem(build_report_module.TOLERANCES, "toy_ratio", tolerance)
    return tolerance


def test_build_conformance_map_verified_target_records_evidence_and_tolerance(
    toy_tolerance,
) -> None:
    """Every case within tolerance: `verified`, no reason, no finding, the declared tolerance."""
    targets = {"toy_ratio": _fake_target(quantcore=_noop, reference=_noop, oracle="hand formula")}
    cases, details = _toy_run("toy_ratio", within=[True, True, True])

    entry = build_conformance_map(cases, details, "ci", targets).targets["toy_ratio"]

    assert entry.status == "verified"
    assert entry.oracle == "hand formula"
    assert entry.cases == 3
    assert entry.by_quantity == {"value": QuantityErrors(3, 0.0, 0.0)}
    assert entry.tolerance == ToleranceRecord(1e-6, 1e-8, "toy tolerance")
    assert entry.reason == ""
    assert entry.finding is None


def test_build_conformance_map_defect_target_gets_a_numbered_finding_and_its_title(
    toy_tolerance, tmp_path, monkeypatch
) -> None:
    """A case outside tolerance: `defect`, a numbered finding, its title as the reason."""
    monkeypatch.setattr(build_report_module, "FINDINGS_SRC_DIR", tmp_path)
    (tmp_path / "toy_ratio.md").write_text(_FINDING_SOURCE_TEXT, encoding="utf-8")
    targets = {"toy_ratio": _fake_target(quantcore=_noop, reference=_noop)}
    cases, details = _toy_run("toy_ratio", within=[True, False])

    entry = build_conformance_map(cases, details, "ci", targets).targets["toy_ratio"]

    assert entry.status == "defect"
    assert entry.finding == "conformance/findings/001-toy_ratio.md"
    assert entry.reason == "toy_ratio is off by a half"
    assert entry.max_abs_error == pytest.approx(0.5)


def test_build_conformance_map_excludes_ill_conditioned_cases_from_the_decision(
    toy_tolerance,
) -> None:
    """A case outside tolerance does not cost `verified` when it is flagged ill-conditioned."""
    targets = {"toy_ratio": _fake_target(quantcore=_noop, reference=_noop)}
    cases, details = _toy_run("toy_ratio", within=[True, False], ill=[False, True])

    entry = build_conformance_map(cases, details, "ci", targets).targets["toy_ratio"]

    assert entry.status == "verified"
    assert entry.cases == 2


def test_build_conformance_map_ignores_second_oracle_cases(toy_tolerance) -> None:
    """A disagreeing second-oracle cross-check never changes a target's status or case count."""
    targets = {"toy_ratio": _fake_target(quantcore=_noop, reference=_noop)}
    cases, details = _toy_run("toy_ratio", within=[True])
    cross_cases, cross_details = _toy_run("toy_ratio", within=[False])
    cases += [dataclasses.replace(c, oracle="quantstats") for c in cross_cases]
    details += [dataclasses.replace(d, oracle="quantstats") for d in cross_details]

    entry = build_conformance_map(cases, details, "ci", targets).targets["toy_ratio"]

    assert entry.status == "verified"
    assert entry.cases == 1
    assert entry.max_abs_error == 0.0


@pytest.mark.parametrize(
    "regenerated_entry",
    [
        _entry(max_abs_error=9.99e-9, max_rel_error=9.99e-9),
        _entry(cases=5000),
        _entry(oracle="empyrical-reloaded (v2)"),
    ],
    ids=["error-magnitude", "case-count", "oracle-version"],
)
def test_decision_diff_ignores_noise(regenerated_entry: TargetRecord) -> None:
    """Differing only in error magnitude, case count, or oracle string is not a decision change."""
    committed = _map()
    regenerated = _map(targets={"sharpe_ratio": regenerated_entry})
    assert decision_diff(committed, regenerated) == []


def test_decision_diff_ignores_versions_and_profile() -> None:
    """Library versions, the quantcore version and the profile are not decisions either."""
    committed = _map()
    regenerated = dataclasses.replace(
        committed,
        quantcore_version="9.9.9",
        oracle_versions={"statsmodels": "9.9.9"},
        hypothesis_profile="full",
    )
    assert decision_diff(committed, regenerated) == []


@pytest.mark.parametrize(
    ("regenerated_entry", "expected_substring"),
    [
        (_entry(status="defect"), "sharpe_ratio.status"),
        (_entry(rtol=1e-6), "sharpe_ratio.tolerance"),
        (_entry(atol=1e-6), "sharpe_ratio.tolerance"),
        (_entry(justification="another reason"), "sharpe_ratio.tolerance"),
        (_entry(finding="conformance/findings/001-sharpe_ratio.md"), "sharpe_ratio.finding"),
    ],
    ids=["status-change", "rtol-change", "atol-change", "justification-change", "finding-change"],
)
def test_decision_diff_reports_changed_field(
    regenerated_entry: TargetRecord, expected_substring: str
) -> None:
    """A changed status, tolerance, or finding produces a diff line naming the field."""
    committed = _map()
    regenerated = _map(targets={"sharpe_ratio": regenerated_entry})
    diff = decision_diff(committed, regenerated)
    assert len(diff) == 1
    assert expected_substring in diff[0]


def test_decision_diff_reports_a_tolerance_that_appears_or_disappears() -> None:
    """A target going from no tolerance to a tolerance (or back) is a decision change."""
    without = dataclasses.replace(_entry(), tolerance=None)
    diff = decision_diff(_map(targets={"f": without}), _map(targets={"f": _entry()}))
    assert len(diff) == 1
    assert "f.tolerance" in diff[0]
    assert decision_diff(_map(targets={"f": without}), _map(targets={"f": without})) == []


def test_decision_diff_reports_added_and_removed_targets() -> None:
    """A target present in only one of the two maps produces a line naming it."""
    committed = _map(targets={"sharpe_ratio": _entry()})
    regenerated = _map(targets={"sharpe_ratio": _entry(), "maximum_drawdown": _entry()})
    assert decision_diff(committed, regenerated) == [
        "maximum_drawdown: added in the regenerated map"
    ]
    assert decision_diff(committed, _map(targets={})) == [
        "sharpe_ratio: removed from the regenerated map"
    ]


def test_missing_finding_sources_reports_only_defect_or_non_standard_without_a_source(
    tmp_path, monkeypatch
) -> None:
    """A defect/non-standard target with no findings_src file is named; others are not."""
    monkeypatch.setattr(build_report_module, "FINDINGS_SRC_DIR", tmp_path)
    (tmp_path / "has_source.md").write_text("present", encoding="utf-8")
    targets = {
        "has_source": _entry(status="defect"),
        "missing_source": _entry(status="non-standard"),
        "verified_fn": _entry(status="verified"),
        "unverifiable_fn": _entry(status="unverifiable"),
    }
    missing = missing_finding_sources(targets)
    assert any("missing_source" in line for line in missing)
    assert not any("has_source" in line for line in missing)
    assert not any("verified_fn" in line for line in missing)
    assert not any("unverifiable_fn" in line for line in missing)


def test_unused_finding_sources_lists_sources_of_targets_without_a_finding(
    tmp_path, monkeypatch
) -> None:
    """A source for a `verified` target is listed; one in use, or for an absent target, is not."""
    monkeypatch.setattr(build_report_module, "FINDINGS_SRC_DIR", tmp_path)
    for name in ("still_defect", "became_verified", "left_out_of_the_map"):
        (tmp_path / f"{name}.md").write_text("present", encoding="utf-8")
    targets = {
        "still_defect": _entry(status="defect"),
        "became_verified": _entry(status="verified"),
    }
    unused = unused_finding_sources(targets)
    assert len(unused) == 1
    assert unused[0].startswith("became_verified: ")
    assert "`verified`" in unused[0]


@pytest.mark.parametrize(
    ("inputs_repr", "expected_total"),
    [
        ("(array([1.0, 2.0, 3.0]),)", 3),
        ("(array([1.0, 2.0]), array([3.0, 4.0, 5.0]))", 5),
        ("(1.5,)", 1),
    ],
    ids=["single-array", "two-arrays", "scalar"],
)
def test_reconstruct_args_and_total_numbers(inputs_repr: str, expected_total: int) -> None:
    """reconstruct_args rebuilds the input tuple; _total_numbers counts its scalars."""
    args = reconstruct_args(inputs_repr)
    assert _total_numbers(args) == expected_total


def _cr(
    *,
    function: str = "f",
    quantity: str = "value",
    family: str = "normal",
    within: bool = True,
    oracle: str = "",
    quantcore_raised: str | None = None,
    length_bucket: str = "short",
) -> CaseResult:
    return CaseResult(
        function=function,
        family=family,
        length_bucket=length_bucket,
        abs_err=1e-9,
        rel_err=1e-9,
        within_tolerance=within,
        quantcore_raised=quantcore_raised,
        reference_raised=None,
        quantity=quantity,
        oracle=oracle,
    )


def _rd(
    *,
    function: str = "f",
    quantity: str = "value",
    family: str = "normal",
    seed: int | None = 1,
    inputs_repr: str = "(array([1.0, 2.0]),)",
    abs_err: float | None = 1e-9,
    rel_err: float | None = 1e-9,
    expected_abs: float = 1.0,
    oracle: str = "",
) -> ReproDetail:
    return ReproDetail(
        function=function,
        family=family,
        quantity=quantity,
        seed=seed,
        abs_err=abs_err,
        rel_err=rel_err,
        expected_abs=expected_abs,
        quantcore_value="0.0",
        reference_value="1.0",
        inputs_repr=inputs_repr,
        oracle=oracle,
    )


def test_matched_pairs_filters_and_zips_by_function_and_quantity() -> None:
    """Only (function, quantity)-matching cases/details are paired, in recording order."""
    cases = [_cr(function="f", quantity="value"), _cr(function="g", quantity="value")]
    details = [_rd(function="f", quantity="value"), _rd(function="g", quantity="value")]
    pairs = matched_pairs(cases, details, function="f", quantity="value")
    assert pairs == [(cases[0], details[0])]


def test_matched_pairs_raises_on_length_mismatch() -> None:
    """Mismatched filtered lengths signal a broken recording invariant and must not be silenced."""
    cases = [_cr(function="f")]
    details: list[ReproDetail] = []
    with pytest.raises(ValueError, match="shorter"):
        matched_pairs(cases, details, function="f", quantity="value")


def test_select_shortest_failing_detail_picks_the_shortest_among_failing_cases() -> None:
    """Among failing cases, the one with fewer total numbers is chosen."""
    short_failing = _rd(inputs_repr="(array([1.0, 2.0]),)")
    long_failing = _rd(inputs_repr="(array([1.0, 2.0, 3.0, 4.0]),)")
    pairs = [(_cr(within=False), short_failing), (_cr(within=False), long_failing)]
    assert select_shortest_failing_detail(pairs) is short_failing


def test_select_shortest_failing_detail_ties_break_toward_smaller_abs_values() -> None:
    """Equal-length failing cases break the tie toward the smaller sum of absolute values."""
    small_values = _rd(inputs_repr="(array([1.0, 1.0]),)")
    large_values = _rd(inputs_repr="(array([9.0, 9.0]),)")
    pairs = [(_cr(within=False), large_values), (_cr(within=False), small_values)]
    assert select_shortest_failing_detail(pairs) is small_values


def test_select_shortest_failing_detail_ignores_passing_cases() -> None:
    """A case within tolerance is never selected, even if it is the only recorded case."""
    pairs = [(_cr(within=True), _rd())]
    assert select_shortest_failing_detail(pairs) is None


def test_evidence_for_quantity_reuses_the_conformance_maps_own_error_numbers() -> None:
    """worst_abs/worst_rel come straight from QuantityErrors, matching the summary table."""
    pairs = [
        (_cr(family="normal", within=False), _rd(family="normal")),
        (_cr(family="ar1", within=True), _rd(family="ar1")),
    ]
    quantity_errors = QuantityErrors(cases=2, max_abs_error=4.2e-8, max_rel_error=1.1e-9)
    evidence = evidence_for_quantity(pairs, quantity_errors)
    assert evidence.cases == 2
    assert evidence.failing == 1
    assert evidence.ill_conditioned == 0
    assert evidence.worst_abs == 4.2e-8
    assert evidence.worst_rel == 1.1e-9
    assert evidence.failing_by_family == {"normal": 1}
    assert evidence.failing_by_bucket == {"short": 1}


def test_evidence_for_quantity_does_not_count_ill_conditioned_cases_as_failing() -> None:
    """A case excluded from the decision is reported as ill-conditioned, never as failing."""
    excluded = dataclasses.replace(_cr(family="tiny_scale", within=False), ill_conditioned=True)
    pairs = [
        (excluded, _rd(family="tiny_scale")),
        (_cr(family="normal", within=False), _rd(family="normal")),
        (_cr(family="ar1", within=True), _rd(family="ar1")),
    ]
    evidence = evidence_for_quantity(pairs, None)
    assert evidence.cases == 3
    assert evidence.failing == 1
    assert evidence.ill_conditioned == 1
    assert evidence.failing_by_family == {"normal": 1}


def test_evidence_for_quantity_handles_no_quantity_errors() -> None:
    """A missing QuantityErrors (an empty run) yields None error magnitudes, not a crash."""
    evidence = evidence_for_quantity([], None)
    assert evidence.cases == 0
    assert evidence.failing == 0
    assert evidence.worst_abs is None
    assert evidence.worst_rel is None
    assert evidence.worst_family is None
    assert evidence.failing_by_family == {}
    assert evidence.failing_by_bucket == {}


def test_render_minimal_repro_inlines_a_short_maximum_drawdown_case() -> None:
    """A short (<=12-number) input inlines with full precision; imports numpy/quantcore/oracle."""
    detail = _rd(
        function="maximum_drawdown",
        quantity="value",
        inputs_repr="(array([-0.5]),)",
    )
    code = render_minimal_repro("maximum_drawdown", "value", detail)
    assert "conformance" not in code
    assert "returns = np.array([-0.5])" in code
    assert 'print("expected (oracle):"' in code
    assert "rng.default_rng" not in code


def test_render_minimal_repro_regenerates_a_long_engle_granger_case_from_its_seed() -> None:
    """A >12-number input regenerates from its seed instead of inlining raw numbers."""
    y = ", ".join(str(float(i)) for i in range(15))
    x = ", ".join(str(float(i)) for i in range(15))
    detail = _rd(
        function="engle_granger_test",
        quantity="statistic",
        seed=42,
        family="cointegrated",
        inputs_repr=f"(array([{y}]), array([{x}]))",
    )
    code = render_minimal_repro("engle_granger_test", "statistic", detail)
    assert "np.random.default_rng(42)" in code
    assert "n = 15" in code
    assert "common = np.cumsum" in code
    assert "conformance" not in code


def test_render_minimal_repro_raises_without_a_seed_for_a_long_input() -> None:
    """A long input with no recorded seed cannot be regenerated and fails loudly, not silently."""
    y = ", ".join(str(float(i)) for i in range(15))
    detail = _rd(
        function="maximum_drawdown", quantity="value", seed=None, inputs_repr=f"(array([{y}]),)"
    )
    with pytest.raises(ValueError, match="no recorded seed"):
        render_minimal_repro("maximum_drawdown", "value", detail)


def test_run_repro_subprocess_pass_path_returns_stdout_and_zero_returncode() -> None:
    """A well-behaved snippet returns its printed stdout and exit code 0."""
    result = run_repro_subprocess("print('hello')")
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert not result.timed_out


def test_run_repro_subprocess_crash_path_returns_nonzero_returncode() -> None:
    """A snippet that raises returns a non-zero exit code and captures the traceback in stderr."""
    result = run_repro_subprocess("raise RuntimeError('boom')")
    assert result.returncode != 0
    assert "boom" in result.stderr
    assert not result.timed_out


def test_run_repro_subprocess_timeout_path_sets_timed_out() -> None:
    """A snippet that outlives its timeout is killed and reported as timed out."""
    result = run_repro_subprocess("import time; time.sleep(5)", timeout=0.2)
    assert result.timed_out
    assert result.returncode is None


def test_parse_repro_stdout_extracts_the_three_printed_numbers() -> None:
    """The expected/actual/abs-diff lines a repro prints are parsed back into floats."""
    stdout = "expected (oracle): 1.5\nactual (quantcore): 2.5\nabs diff: 1.0\n"
    assert parse_repro_stdout(stdout) == (1.5, 2.5, 1.0)


def test_parse_repro_stdout_raises_when_a_line_is_missing() -> None:
    """Output missing any one of the three required lines fails loudly, not with a wrong parse."""
    with pytest.raises(ValueError, match="missing"):
        parse_repro_stdout("expected (oracle): 1.5\nactual (quantcore): 2.5\n")


def test_repro_still_mismatches_true_beyond_tolerance_false_within() -> None:
    """The mismatch check uses the same |a-e| <= atol + rtol*|e| rule as the rest of the harness."""
    tolerance = Tolerance(rtol=1e-3, atol=1e-6, justification="test")
    assert repro_still_mismatches(expected=1.0, actual=1.1, tolerance=tolerance) is True
    assert repro_still_mismatches(expected=1.0, actual=1.0000001, tolerance=tolerance) is False


def test_verify_repro_pass_path_returns_a_verified_repro(monkeypatch) -> None:
    """A repro that executes and still mismatches beyond tolerance returns a VerifiedRepro."""
    monkeypatch.setattr(
        build_report_module,
        "render_minimal_repro",
        lambda function, quantity, detail: "print('x')",
    )
    monkeypatch.setattr(
        build_report_module,
        "run_repro_subprocess",
        lambda code, timeout=30.0: ReproRunResult(
            returncode=0,
            stdout="expected (oracle): 0.0\nactual (quantcore): 1.0\nabs diff: 1.0\n",
            stderr="",
            timed_out=False,
        ),
    )
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("f", "value", _rd(), tolerance)
    assert error is None
    assert verified is not None
    assert verified.expected == 0.0
    assert verified.actual == 1.0


def test_verify_repro_fail_to_reproduce_path_returns_an_error(monkeypatch) -> None:
    """A repro whose values are within tolerance is reported as an error, not written."""
    monkeypatch.setattr(
        build_report_module, "render_minimal_repro", lambda function, quantity, detail: "print('x')"
    )
    monkeypatch.setattr(
        build_report_module,
        "run_repro_subprocess",
        lambda code, timeout=30.0: ReproRunResult(
            returncode=0,
            stdout="expected (oracle): 1.0\nactual (quantcore): 1.0\nabs diff: 0.0\n",
            stderr="",
            timed_out=False,
        ),
    )
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("f", "value", _rd(), tolerance)
    assert verified is None
    assert error is not None
    assert "no longer mismatches" in error


def test_verify_repro_crash_path_returns_an_error(monkeypatch) -> None:
    """A repro that crashes is reported as an error naming the non-zero exit code."""
    monkeypatch.setattr(
        build_report_module, "render_minimal_repro", lambda function, quantity, detail: "1/0"
    )
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("f", "value", _rd(), tolerance)
    assert verified is None
    assert error is not None
    assert "exited" in error


def test_verify_repro_timeout_path_returns_an_error(monkeypatch) -> None:
    """A repro that times out is reported as an error naming the timeout."""
    monkeypatch.setattr(
        build_report_module,
        "render_minimal_repro",
        lambda function, quantity, detail: "import time; time.sleep(5)",
    )
    monkeypatch.setattr(build_report_module, "REPRO_TIMEOUT_SECONDS", 0.2)
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("f", "value", _rd(), tolerance)
    assert verified is None
    assert error is not None
    assert "timed out" in error


def test_render_minimal_repro_uses_a_summary_expr_for_an_array_valued_quantity(
    monkeypatch,
) -> None:
    """A quantity declared in `summary_expr` prints one `summary` line, not expected/actual/diff.

    Synthetic array-valued case: the quantity's raw output (a decoded-state array) is
    not itself a scalar, so the repro instead prints a reduced scalar SUMMARY (here, the
    agreement rate between two arrays).
    """
    config = ReproCallConfig(
        module_import="import numpy as _np",
        quantcore_call="_np.array([1, 1, 0, 1])",
        oracle_import="",
        oracle_call="_np.array([1, 1, 1, 1])",
        arg_names=("states",),
        quantity_index={},
        summary_expr={"agreement": "float(_np.mean(quantcore_result == oracle_result))"},
        summary_ok={"agreement": lambda summary: summary >= 1.0},
    )
    monkeypatch.setitem(build_report_module.REPRO_CONFIGS, "synthetic_decoder", config)
    detail = _rd(
        function="synthetic_decoder", quantity="agreement", inputs_repr="(array([1.0, 2.0]),)"
    )
    code = render_minimal_repro("synthetic_decoder", "agreement", detail)
    assert 'print("summary:", repr(summary))' in code
    assert "expected (oracle)" not in code

    result = run_repro_subprocess(code)
    assert result.returncode == 0
    summary = parse_repro_summary(result.stdout)
    assert summary == pytest.approx(0.75)


def test_verify_repro_summary_path_returns_a_verified_repro_with_a_summary(monkeypatch) -> None:
    """A summary-quantity repro that still fails `summary_ok` returns a `VerifiedRepro.summary`."""
    config = ReproCallConfig(
        module_import="",
        quantcore_call="",
        oracle_import="",
        oracle_call="",
        arg_names=("states",),
        quantity_index={},
        summary_expr={"agreement": "0.75"},
        summary_ok={"agreement": lambda summary: summary >= 1.0},
    )
    monkeypatch.setitem(build_report_module.REPRO_CONFIGS, "synthetic_decoder", config)
    monkeypatch.setattr(
        build_report_module,
        "render_minimal_repro",
        lambda function, quantity, detail: "print('summary:', repr(0.75))",
    )
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("synthetic_decoder", "agreement", _rd(), tolerance)
    assert error is None
    assert verified is not None
    assert verified.summary == pytest.approx(0.75)
    assert verified.expected is None
    assert verified.actual is None


def test_verify_repro_summary_path_reports_stale_when_summary_ok(monkeypatch) -> None:
    """A summary that satisfies `summary_ok` means the case does not reproduce the finding."""
    config = ReproCallConfig(
        module_import="",
        quantcore_call="",
        oracle_import="",
        oracle_call="",
        arg_names=("states",),
        quantity_index={},
        summary_expr={"agreement": "1.0"},
        summary_ok={"agreement": lambda summary: summary >= 1.0},
    )
    monkeypatch.setitem(build_report_module.REPRO_CONFIGS, "synthetic_decoder", config)
    monkeypatch.setattr(
        build_report_module,
        "render_minimal_repro",
        lambda function, quantity, detail: "print('summary:', repr(1.0))",
    )
    tolerance = Tolerance(rtol=1e-10, atol=1e-10, justification="test")
    verified, error = verify_repro("synthetic_decoder", "agreement", _rd(), tolerance)
    assert verified is None
    assert error is not None
    assert "no longer reproduces" in error


def test_verified_functions_from_map_selects_only_verified_status() -> None:
    """Only targets whose status is exactly 'verified' are returned, others are excluded."""
    conformance_map = _map(
        targets={
            "sharpe_ratio": _entry(status="verified"),
            "engle_granger_test": _entry(status="defect"),
            "maximum_drawdown": _entry(status="non-standard"),
            "select_arima_order": _entry(status="unverifiable"),
        }
    )
    assert verified_functions_from_map(conformance_map) == {"sharpe_ratio"}


def test_oracle_cross_check_rows_ignores_primary_oracle_cases() -> None:
    """A case with oracle == '' (the primary comparison) never produces a cross-check row."""
    cases = [_cr(function="sharpe_ratio", oracle="")]
    details = [_rd(function="sharpe_ratio", oracle="")]
    assert oracle_cross_check_rows(cases, details) == []


def test_oracle_cross_check_rows_aggregates_per_function_oracle_quantity() -> None:
    """Cases sharing (function, oracle, quantity) collapse into one row with the worst errors."""
    cases = [
        _cr(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", within=True),
        _cr(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", within=True),
    ]
    details = [
        _rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", rel_err=1e-9),
        _rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", rel_err=5e-8),
    ]
    rows = oracle_cross_check_rows(cases, details)
    assert len(rows) == 1
    row = rows[0]
    assert (row.function, row.oracle, row.quantity) == ("sharpe_ratio", "quantstats", "rf_zero")
    assert row.cases == 2
    assert row.agrees is True
    assert row.raised_or_nonfinite == 0
    assert row.max_rel_diff == pytest.approx(5e-8, rel=1e-6)


def test_oracle_cross_check_rows_disagrees_when_any_case_out_of_tolerance() -> None:
    """A single out-of-tolerance case among agreeing ones marks the whole row as disagreeing."""
    cases = [
        _cr(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", within=True),
        _cr(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", within=False),
    ]
    details = [
        _rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats"),
        _rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats"),
    ]
    rows = oracle_cross_check_rows(cases, details)
    assert rows[0].agrees is False


def test_oracle_cross_check_rows_counts_raised_or_nonfinite_cases() -> None:
    """A case where the second oracle raised is counted separately from an ordinary disagreement."""
    cases = [
        _cr(
            function="sharpe_ratio",
            quantity="rf_zero",
            oracle="quantstats",
            within=False,
            quantcore_raised="ValueError: boom",
        )
    ]
    details = [_rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats")]
    rows = oracle_cross_check_rows(cases, details)
    assert rows[0].raised_or_nonfinite == 1


def test_oracle_disagreement_examples_empty_when_every_row_agrees() -> None:
    """No disagreement examples are produced when every recorded cross-check case agreed."""
    cases = [_cr(function="sharpe_ratio", oracle="quantstats", within=True)]
    details = [_rd(function="sharpe_ratio", oracle="quantstats")]
    assert oracle_disagreement_examples(cases, details) == []


def test_oracle_disagreement_examples_picks_the_shortest_failing_case() -> None:
    """The shortest recorded disagreeing input is picked as the example, matching findings' rule."""
    cases = [
        _cr(function="sharpe_ratio", oracle="quantstats", within=False),
        _cr(function="sharpe_ratio", oracle="quantstats", within=False),
    ]
    details = [
        _rd(
            function="sharpe_ratio",
            oracle="quantstats",
            inputs_repr="(array([1.0, 2.0, 3.0, 4.0]),)",
        ),
        _rd(function="sharpe_ratio", oracle="quantstats", inputs_repr="(array([1.0, 2.0]),)"),
    ]
    examples = oracle_disagreement_examples(cases, details)
    assert len(examples) == 1
    assert examples[0].function == "sharpe_ratio"
    assert examples[0].oracle == "quantstats"


def test_explore_mismatches_reports_a_mismatch_for_a_verified_function() -> None:
    """A case that is out of tolerance for a `verified` target is reported as a mismatch."""
    cases = [_cr(function="sharpe_ratio", within=False)]
    details = [_rd(function="sharpe_ratio")]
    mismatches = explore_mismatches(cases, details, verified_functions={"sharpe_ratio"})
    assert len(mismatches) == 1
    assert mismatches[0].function == "sharpe_ratio"
    assert "abs_err" in mismatches[0].description


def test_explore_mismatches_ignores_a_defect_targets_mismatches() -> None:
    """A mismatch recorded for a target *not* in `verified_functions` is silently ignored."""
    cases = [_cr(function="engle_granger_test", within=False)]
    details = [_rd(function="engle_granger_test")]
    assert explore_mismatches(cases, details, verified_functions={"sharpe_ratio"}) == []


def test_explore_mismatches_empty_when_nothing_mismatches() -> None:
    """A verified target with every case within tolerance produces no mismatches."""
    cases = [_cr(function="sharpe_ratio", within=True)]
    details = [_rd(function="sharpe_ratio")]
    assert explore_mismatches(cases, details, verified_functions={"sharpe_ratio"}) == []


def test_explore_mismatches_reports_a_quantcore_raise_as_a_mismatch() -> None:
    """A case where quantcore raised is a mismatch even though within_tolerance is already False."""
    cases = [
        _cr(
            function="sharpe_ratio",
            within=False,
            quantcore_raised="ValueError: boom",
        )
    ]
    details = [_rd(function="sharpe_ratio")]
    mismatches = explore_mismatches(cases, details, verified_functions={"sharpe_ratio"})
    assert len(mismatches) == 1
    assert "quantcore raised" in mismatches[0].description


def test_explore_mismatches_ignores_second_oracle_cases() -> None:
    """A second-oracle (e.g. quantstats) disagreement is never itself reported as a mismatch."""
    cases = [_cr(function="sharpe_ratio", oracle="quantstats", within=False)]
    details = [_rd(function="sharpe_ratio", oracle="quantstats")]
    assert explore_mismatches(cases, details, verified_functions={"sharpe_ratio"}) == []


def test_remove_stale_findings_deletes_only_unreferenced_generated_files(tmp_path) -> None:
    """A renumbered or resolved finding's old file goes; current and hand-written files stay."""
    generated = GENERATED_FILE_COMMENT + "\n\n# finding\n"
    (tmp_path / "001-adf_test.md").write_text(generated)
    (tmp_path / "005-engle_granger_test.md").write_text(generated)
    (tmp_path / "002-adf_test.md").write_text(generated)
    (tmp_path / "notes.md").write_text("# hand-written notes\n")

    removed = remove_stale_findings(tmp_path, ["001-adf_test.md"])

    assert [path.name for path in removed] == ["002-adf_test.md", "005-engle_granger_test.md"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["001-adf_test.md", "notes.md"]


# ---------------------------------------------------------------------------
# Rendering a finding
# ---------------------------------------------------------------------------

_FINDING_SOURCE_TEXT = """\
## Title

toy_ratio is off by a half

## Severity

Low -- a toy.

## Location

nowhere

## Root cause

It adds a half.

## Standard definition and citation

The ratio itself.

## Proposed patch

Do not add a half.

## Promotion test

Every case within tolerance.
"""

_FINDING_SOURCE = FindingSource(
    title="toy_ratio is off by a half",
    severity="Low -- a toy.",
    location="nowhere",
    root_cause="It adds a half.",
    standard_definition="The ratio itself.",
    proposed_patch="Do not add a half.",
    promotion_test="Every case within tolerance.",
)


def test_render_finding_merges_the_source_with_the_runs_evidence() -> None:
    """The finding starts with the generated-file marker and carries source, evidence and repro."""
    entry = _entry(status="defect", finding="conformance/findings/001-toy_ratio.md")
    evidence = evidence_for_quantity(
        [(_cr(within=False), _rd()), (_cr(within=True), _rd())], QuantityErrors(2, 0.5, 0.5)
    )
    repro = build_report_module.VerifiedRepro(
        quantity="value", code="print('repro')", expected=1.0, actual=1.5, abs_diff=0.5
    )

    text = render_finding(
        "toy_ratio",
        entry,
        _map(targets={"toy_ratio": entry}),
        _FINDING_SOURCE,
        evidence_by_quantity={"value": evidence},
        repros=[repro],
    )

    assert text.splitlines()[0] == GENERATED_FILE_COMMENT
    assert "# toy_ratio is off by a half" in text
    assert "- **Status:** `defect`" in text
    assert "- **Target:** `toy_ratio`" in text
    assert "- cases run: 2; failing: 1" in text
    assert "`expected (oracle)` = `1.0`, `actual (quantcore)` = `1.5`" in text
    assert "print('repro')" in text
    assert "Do not add a half." in text
    assert "Sign test" not in text


def test_render_finding_shows_the_sign_test_for_an_iterative_estimator() -> None:
    """An iterative estimator's finding states the objective's sense and the sign test's counts."""
    estimator = IterativeEstimator(frozenset({"normal"}), "objective", "minimize")
    cases = [_cr(function="toy_optimiser", quantity="objective") for _ in range(6)]
    details = [
        dataclasses.replace(
            _rd(function="toy_optimiser", quantity="objective"),
            quantcore_value="2.0",
            reference_value="1.0",
        )
        for _ in range(6)
    ]
    decision = build_report_module.classify_iterative_estimator(
        objective_cases_for("toy_optimiser", cases, details, estimator), rtol=1e-6, atol=1e-8
    )
    entry = _entry(status="defect", finding="conformance/findings/001-toy_optimiser.md")

    text = render_finding(
        "toy_optimiser",
        entry,
        _map(targets={"toy_optimiser": entry}),
        _FINDING_SOURCE,
        evidence_by_quantity={},
        repros=[],
        sign_test=(estimator, decision),
    )

    assert "### Sign test on the attained objective" in text
    assert "`objective` (minimized: a smaller value is the better solution)" in text
    assert "quantcore better on 0, tied on 0, worse on 6" in text


# ---------------------------------------------------------------------------
# Iterative estimators: which way is better
# ---------------------------------------------------------------------------


def test_oriented_objective_negates_only_a_minimised_objective() -> None:
    """A larger oriented value is always the better solution, whichever way the objective runs."""
    assert oriented_objective(3.0, "maximize") == 3.0
    assert oriented_objective(3.0, "minimize") == -3.0
    better_variance, worse_variance = 1.0e-4, 2.0e-4
    assert oriented_objective(better_variance, "minimize") > oriented_objective(
        worse_variance, "minimize"
    )


def _objective_run(
    name: str, *, quantcore: float, reference: float, n: int = 20, quantity: str = "objective"
) -> tuple[list[CaseResult], list[ReproDetail]]:
    """``n`` cases on which quantcore attains ``quantcore`` and the oracle ``reference``."""
    cases = [_cr(function=name, quantity=quantity, within=False) for _ in range(n)]
    details = [
        dataclasses.replace(
            _rd(function=name, quantity=quantity),
            quantcore_value=repr(quantcore),
            reference_value=repr(reference),
        )
        for _ in range(n)
    ]
    return cases, details


@pytest.mark.parametrize(
    ("sense", "quantcore", "reference", "expected_status", "expected_wins", "expected_losses"),
    [
        ("maximize", 2.0, 1.0, "verified", 20, 0),
        ("maximize", 1.0, 2.0, "defect", 0, 20),
        ("minimize", 1.0, 2.0, "verified", 20, 0),
        ("minimize", 2.0, 1.0, "defect", 0, 20),
    ],
    ids=[
        "maximised-and-larger-wins",
        "maximised-and-smaller-loses",
        "minimised-and-smaller-wins",
        "minimised-and-larger-loses",
    ],
)
def test_iterative_estimator_decision_respects_the_objectives_sense(
    monkeypatch,
    sense: str,
    quantcore: float,
    reference: float,
    expected_status: str,
    expected_wins: int,
    expected_losses: int,
) -> None:
    """A win is the better objective: the larger one if maximised, the smaller one if minimised.

    A solver that attains a larger value of a minimised objective (a higher portfolio variance,
    say) on every case has lost every case and must not be `verified`; the same numbers under a
    maximised objective (a log-likelihood) are twenty wins.
    """
    monkeypatch.setitem(
        build_report_module.TOLERANCES, "toy_optimiser", Tolerance(1e-6, 1e-8, "test")
    )
    estimator = IterativeEstimator(frozenset({"normal"}), "objective", sense)  # type: ignore[arg-type]
    cases, details = _objective_run("toy_optimiser", quantcore=quantcore, reference=reference)

    decision = iterative_estimator_decision("toy_optimiser", cases, details, estimator)

    assert decision.status == expected_status
    assert (decision.wins, decision.losses) == (expected_wins, expected_losses)
    assert (decision.median_gap > 0.0) == (expected_status == "verified")


@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_objective_cases_for_counts_a_quantcore_raise_as_a_loss_either_way(sense: str) -> None:
    """A case with no quantcore value is the worst possible outcome under either sense."""
    estimator = IterativeEstimator(frozenset({"normal"}), "objective", sense)  # type: ignore[arg-type]
    cases = [_cr(function="toy_optimiser", quantity="objective", within=False)]
    details = [
        dataclasses.replace(
            _rd(function="toy_optimiser", quantity="objective"),
            quantcore_value=None,
            reference_value="1.0",
        )
    ]

    (case,) = objective_cases_for("toy_optimiser", cases, details, estimator)

    assert case.quantcore_objective == -math.inf
    assert case.quantcore_objective < case.reference_objective


def test_objective_cases_for_marks_only_declared_families_well_specified() -> None:
    """A family outside the estimator's well-specified set is recorded as misspecified."""
    estimator = IterativeEstimator(frozenset({"normal"}), "log_likelihood", "maximize")
    cases = [
        _cr(function="toy_fit", quantity="log_likelihood", family="normal"),
        _cr(function="toy_fit", quantity="log_likelihood", family="student_t3"),
    ]
    details = [
        _rd(function="toy_fit", quantity="log_likelihood", family="normal"),
        _rd(function="toy_fit", quantity="log_likelihood", family="student_t3"),
    ]

    normal, heavy_tailed = objective_cases_for("toy_fit", cases, details, estimator)

    assert normal.well_specified is True
    assert heavy_tailed.well_specified is False


def test_build_conformance_map_judges_an_iterative_estimator_by_its_sign_test(monkeypatch) -> None:
    """A minimised objective that quantcore attains higher on every case is never `verified`.

    Every case is also recorded outside tolerance, which for an ordinary target would be a
    `defect` regardless of direction; here the status must follow the sign test, in which
    attaining a smaller minimised objective on every case is `verified`.
    """
    monkeypatch.setitem(
        build_report_module.TOLERANCES, "toy_optimiser", Tolerance(1e-6, 1e-8, "test")
    )
    monkeypatch.setitem(
        build_report_module.ITERATIVE_ESTIMATORS,
        "toy_optimiser",
        IterativeEstimator(frozenset({"normal"}), "objective", "minimize"),
    )
    monkeypatch.setattr(build_report_module, "FINDINGS_SRC_DIR", Path("/nonexistent"))
    targets = {"toy_optimiser": _fake_target(quantcore=_noop, reference=_noop)}

    worse, worse_details = _objective_run("toy_optimiser", quantcore=2.0, reference=1.0)
    better, better_details = _objective_run("toy_optimiser", quantcore=1.0, reference=2.0)

    worse_status = build_conformance_map(worse, worse_details, "ci", targets).targets
    better_status = build_conformance_map(better, better_details, "ci", targets).targets
    assert worse_status["toy_optimiser"].status == "defect"
    assert better_status["toy_optimiser"].status == "verified"


def _shorting_helps_problem() -> tuple[np.ndarray, np.ndarray]:
    """Expected returns and a covariance for which the long-only constraint binds.

    Two highly correlated assets with different volatilities and a third, uncorrelated one: the
    unconstrained optimum shorts the more volatile of the correlated pair, so relaxing the
    long-only constraint strictly improves the optimum.
    """
    cov = np.array([[0.04, 0.054, 0.0], [0.054, 0.09, 0.0], [0.0, 0.0, 0.0625]])
    mu = np.array([0.08, 0.03, 0.05])
    return mu, cov


def _relaxed_and_constrained_objectives(name: str) -> tuple[float, float]:
    """The registered oracle's optimal objective for a program and for a relaxation of it."""
    mu, cov = _shorting_helps_problem()
    targets = registry()
    reference = targets[name].reference
    assert reference is not None
    if name == "min_variance_weights":
        return reference(cov, True)[0], reference(cov, False)[0]
    if name == "mean_variance_weights":
        return reference(mu, cov, 3.0, True)[0], reference(mu, cov, 3.0, False)[0]
    if name == "l1_turnover_penalized_weights":
        previous = np.full(3, 1.0 / 3.0)
        return (
            reference(mu, cov, previous, 5.0, 3.0, True)[0],
            reference(mu, cov, previous, 5.0, 3.0, False)[0],
        )
    if name == "unconstrained_mean_variance_weights":
        constrained = targets["mean_variance_weights"].reference
        assert constrained is not None
        return reference(mu, cov, 3.0)[0], constrained(mu, cov, 3.0, False)[0]
    raise AssertionError(f"no relaxation written for {name!r}")


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name, estimator in ITERATIVE_ESTIMATORS.items()
        if estimator.quantity == "objective"
    ),
)
def test_declared_sense_matches_what_the_registered_oracle_optimises(name: str) -> None:
    """Relaxing a constraint moves the oracle's optimum the way the declared sense says is better.

    The optimum of a relaxed program is at least as good as the optimum of the constrained one.
    So if the recorded objective falls when the long-only constraint is dropped the oracle
    minimises it, and if it rises the oracle maximises it. This pins `ITERATIVE_ESTIMATORS`'s
    `sense` to the registered oracle itself rather than to a comment about it.
    """
    relaxed, constrained = _relaxed_and_constrained_objectives(name)
    assert abs(relaxed - constrained) > 1e-6, "the constraint must bind for this to say anything"
    observed = "minimize" if relaxed < constrained else "maximize"
    assert ITERATIVE_ESTIMATORS[name].sense == observed


def test_every_log_likelihood_estimator_is_maximised() -> None:
    """A fitter judged on its attained log-likelihood is better the larger that likelihood is."""
    for name, estimator in ITERATIVE_ESTIMATORS.items():
        if estimator.quantity == "log_likelihood":
            assert estimator.sense == "maximize", name


def test_min_variance_weights_is_the_minimised_estimator() -> None:
    """Portfolio variance is minimised; the three utility objectives are maximised."""
    senses = {name: estimator.sense for name, estimator in ITERATIVE_ESTIMATORS.items()}
    assert senses["min_variance_weights"] == "minimize"
    assert senses["mean_variance_weights"] == "maximize"
    assert senses["unconstrained_mean_variance_weights"] == "maximize"
    assert senses["l1_turnover_penalized_weights"] == "maximize"


def test_explore_mismatches_judges_a_verified_iterative_estimator_by_its_sign_test(
    monkeypatch,
) -> None:
    """Cases outside tolerance are not mismatches for an estimator whose sign test still passes.

    A `verified` iterative estimator has individual cases outside tolerance by design (every win
    is one), so the explore report runs its sign test again: wins on every case report nothing,
    losses on every case report one mismatch for the target as a whole.
    """
    monkeypatch.setitem(
        build_report_module.TOLERANCES, "toy_optimiser", Tolerance(1e-6, 1e-8, "test")
    )
    monkeypatch.setitem(
        build_report_module.ITERATIVE_ESTIMATORS,
        "toy_optimiser",
        IterativeEstimator(frozenset({"normal"}), "objective", "maximize"),
    )
    wins, win_details = _objective_run("toy_optimiser", quantcore=2.0, reference=1.0)
    losses, loss_details = _objective_run("toy_optimiser", quantcore=1.0, reference=2.0)

    assert explore_mismatches(wins, win_details, verified_functions={"toy_optimiser"}) == []

    mismatches = explore_mismatches(losses, loss_details, verified_functions={"toy_optimiser"})
    assert len(mismatches) == 1
    assert mismatches[0].function == "toy_optimiser"
    assert mismatches[0].quantity == "objective"
    assert "sign test" in mismatches[0].description
    assert "losses=20" in mismatches[0].description


# ---------------------------------------------------------------------------
# Configuration: registry, coverage table and tolerances must agree
# ---------------------------------------------------------------------------

_COVERAGE = DeclaredCoverage(families=("normal",), buckets=("short",))
_TOLERANCE = Tolerance(rtol=1e-10, atol=1e-10, justification="test")


def test_configuration_errors_empty_for_a_consistent_configuration() -> None:
    """Complete targets, each declared and with a tolerance, raise no configuration error."""
    targets = {
        "sharpe_ratio": _fake_target(quantcore=_noop, reference=_noop),
        "select_hmm_n_states": _fake_target(quantcore=_noop),
    }
    errors = configuration_errors(
        targets, {"sharpe_ratio": _COVERAGE}, {"sharpe_ratio": _TOLERANCE}, {"sharpe_ratio"}
    )
    assert errors == {}


def test_configuration_errors_names_every_target_without_a_quantcore_callable() -> None:
    """Each target with nothing to judge is reported under its own name, all in one call."""
    targets = {
        "ulcer_index": _fake_target(reference=_noop),
        "kpss": _fake_target(reference=_noop),
        "sharpe_ratio": _fake_target(quantcore=_noop, reference=_noop),
    }
    declared = dict.fromkeys(targets, _COVERAGE)
    tolerances = dict.fromkeys(targets, _TOLERANCE)

    errors = configuration_errors(targets, declared, tolerances)

    assert sorted(errors) == ["kpss", "ulcer_index"]
    assert errors["kpss"] == [
        "kpss: registered without a quantcore callable, so there is nothing to judge"
    ]


@pytest.mark.parametrize(
    ("declared", "tolerances", "recorded", "name", "expected_substring"),
    [
        ({}, {"f": _TOLERANCE}, set(), "f", "missing from COVERAGE_SOURCES"),
        ({"f": _COVERAGE}, {}, set(), "f", "no entry in TOLERANCES"),
        (
            {"f": _COVERAGE, "ghost": _COVERAGE},
            {"f": _TOLERANCE},
            set(),
            "ghost",
            "not a registered target",
        ),
        ({"f": _COVERAGE}, {"f": _TOLERANCE}, {"f", "stale"}, "stale", "has recorded cases"),
        (
            {"f": DeclaredCoverage(families=(), buckets=("short",))},
            {"f": _TOLERANCE},
            set(),
            "f",
            "no family or no bucket",
        ),
        (
            {"f": DeclaredCoverage(families=("normal",), buckets=("short",), min_len=60)},
            {"f": _TOLERANCE},
            set(),
            "f",
            "lies entirely below its declared minimum length 60",
        ),
    ],
    ids=[
        "missing-from-coverage-table",
        "missing-tolerance",
        "declared-but-not-registered",
        "recorded-but-not-registered",
        "empty-declaration",
        "unreachable-bucket",
    ],
)
def test_configuration_errors_reports_each_kind_of_mismatch(
    declared, tolerances, recorded, name: str, expected_substring: str
) -> None:
    """Every disagreement between registry, coverage table, tolerances and results is named."""
    targets = {"f": _fake_target(quantcore=_noop, reference=_noop)}
    errors = configuration_errors(targets, declared, tolerances, recorded)
    assert list(errors) == [name]
    assert len(errors[name]) == 1
    assert errors[name][0].startswith(f"{name}: ")
    assert expected_substring in errors[name][0]


def test_configuration_errors_flags_declared_coverage_for_a_target_without_an_oracle() -> None:
    """Coverage declared for a target that can never record a case is a mistake."""
    targets = {"select_hmm_n_states": _fake_target(quantcore=_noop)}
    errors = configuration_errors(targets, {"select_hmm_n_states": _COVERAGE}, {})
    assert "no oracle to record a case against" in errors["select_hmm_n_states"][0]


@pytest.mark.parametrize(
    ("min_len", "buckets", "unreachable"),
    [
        (2, ("short", "medium", "long"), []),
        (10, ("short", "medium", "long"), []),
        (11, ("short", "medium", "long"), ["short"]),
        (501, ("short", "medium", "long"), ["short", "medium"]),
        (60, ("medium", "long"), []),
        (None, ("short",), []),
        (5000, ("scalar", "dim_small"), []),
    ],
    ids=[
        "minimum-two",
        "minimum-at-the-short-boundary",
        "minimum-above-short",
        "minimum-above-medium",
        "only-reachable-buckets-declared",
        "no-minimum-declared",
        "non-length-buckets-are-not-ordered",
    ],
)
def test_unreachable_declared_buckets(min_len, buckets, unreachable) -> None:
    """A length bucket is unreachable exactly when its whole range is below the minimum length."""
    declared = {"f": DeclaredCoverage(families=("normal",), buckets=buckets, min_len=min_len)}
    problems = unreachable_declared_buckets(declared)
    found = [
        bucket for bucket in buckets if any(repr(bucket) in line for line in problems.get("f", []))
    ]
    assert found == unreachable


def test_declared_coverage_from_reads_the_modules_constants() -> None:
    """Families, buckets and the optional minimum length come from `<PREFIX>_*` constants."""
    module = SimpleNamespace(
        __name__="toy_module",
        TOY_FAMILIES=("normal", "ar1"),
        TOY_BUCKETS=("short",),
        TOY_MIN_LEN=3,
        SCALAR_FAMILIES=("typical",),
        SCALAR_BUCKETS=("scalar",),
    )
    assert declared_coverage_from(module, "TOY") == DeclaredCoverage(  # type: ignore[arg-type]
        families=("normal", "ar1"), buckets=("short",), min_len=3
    )
    assert declared_coverage_from(module, "SCALAR").min_len is None  # type: ignore[arg-type]


def test_declared_coverage_from_names_a_missing_constant() -> None:
    """A prefix the module does not use fails with the module and the constant in the message."""
    module = SimpleNamespace(__name__="toy_module", TOY_FAMILIES=("normal",))
    with pytest.raises(LookupError, match=r"toy_module declares no TOY_BUCKETS"):
        declared_coverage_from(module, "TOY")  # type: ignore[arg-type]


def test_coverage_sources_name_only_registered_targets_with_an_oracle() -> None:
    """Every line of the coverage table belongs to a registered target that has an oracle."""
    targets = registry()
    for name in COVERAGE_SOURCES:
        assert name in targets, f"{name} is in COVERAGE_SOURCES but is not registered"
        assert targets[name].reference is not None, f"{name} has declared coverage but no oracle"


def test_every_complete_registered_target_declares_coverage_and_a_tolerance() -> None:
    """A target with both a quantcore callable and an oracle has a coverage line and a tolerance.

    A target registered with an oracle only is still being added and is reported by the builder
    itself; once it gains its quantcore callable it needs a line in `COVERAGE_SOURCES`.
    """
    complete = {
        name
        for name, target in registry().items()
        if target.quantcore is not None and target.reference is not None
    }
    assert sorted(complete - set(COVERAGE_SOURCES)) == []
    assert sorted(complete - set(TOLERANCES)) == []


def test_no_declared_length_bucket_is_below_its_declared_minimum_length() -> None:
    """The real coverage table declares no bucket its target's minimum length rules out."""
    assert unreachable_declared_buckets(DECLARED_COVERAGE) == {}


def test_builder_tables_name_only_registered_targets() -> None:
    """Repro configurations, regeneration recipes and iterative estimators match the registry."""
    targets = registry()
    for table in (REPRO_CONFIGS, REGEN_RECIPES, ITERATIVE_ESTIMATORS):
        assert sorted(set(table) - set(targets)) == []
    for name, estimator in ITERATIVE_ESTIMATORS.items():
        assert estimator.well_specified_families <= set(DECLARED_COVERAGE[name].families), name


def test_missing_coverage_by_target_groups_the_lines_of_missing_coverage() -> None:
    """The grouped form holds the same lines as the flat one, keyed by target."""
    declared = {
        "a": DeclaredCoverage(families=("normal", "ar1"), buckets=("short",)),
        "b": DeclaredCoverage(families=("normal",), buckets=("short",)),
    }
    counts = {"a": ({"normal": 1}, {}), "b": ({"normal": 1}, {"short": 1})}
    by_target = missing_coverage_by_target(declared, counts)
    assert list(by_target) == ["a"]
    assert by_target["a"] == missing_coverage(declared, counts)
    assert len(by_target["a"]) == 2


def test_incomplete_targets_merges_configuration_errors_and_coverage_gaps() -> None:
    """Each name lists its configuration errors first, then its coverage gaps, sorted by name."""
    merged = incomplete_targets(
        {"b": ["b: no quantcore"], "a": ["a: no tolerance"]},
        {"a": ["a: no cases for family 'x'"], "c": ["c: no cases for bucket 'y'"]},
    )
    assert list(merged) == ["a", "b", "c"]
    assert merged["a"] == ["a: no tolerance", "a: no cases for family 'x'"]


# ---------------------------------------------------------------------------
# The map as JSON, and the library versions it records
# ---------------------------------------------------------------------------


def test_conformance_map_round_trips_through_json() -> None:
    """Parsing the serialised map gives back an equal map, including an absent tolerance."""
    conformance_map = _map(
        targets={
            "adf_test": _entry(
                status="defect",
                finding="conformance/findings/001-adf_test.md",
                by_quantity={
                    "statistic": QuantityErrors(200, 1e-11, 4e-14),
                    "p_value": QuantityErrors(200, 1e-4, None),
                },
            ),
            "select_hmm_n_states": dataclasses.replace(
                _entry(status="unverifiable"), tolerance=None, max_abs_error=None
            ),
        }
    )
    text = conformance_map_to_json(conformance_map)
    assert conformance_map_from_json(text) == conformance_map
    assert text.endswith("}\n")


def test_conformance_map_json_has_the_documented_fields_in_sorted_order() -> None:
    """Top-level and per-target keys are the documented ones, sorted at every level."""
    raw = json.loads(
        conformance_map_to_json(_map(targets={"b": _entry(), "a": _entry(status="defect")}))
    )
    assert list(raw) == [
        "hypothesis_profile",
        "oracle_versions",
        "quantcore_version",
        "schema_version",
        "targets",
    ]
    assert list(raw["targets"]) == ["a", "b"]
    assert list(raw["targets"]["a"]) == [
        "by_quantity",
        "cases",
        "finding",
        "max_abs_error",
        "max_rel_error",
        "oracle",
        "reason",
        "status",
        "tolerance",
    ]
    assert list(raw["targets"]["a"]["tolerance"]) == ["atol", "justification", "rtol"]


def test_conformance_map_from_json_rejects_another_schema_version() -> None:
    """A map of a schema version this builder does not read is refused, naming the version."""
    raw = json.loads(conformance_map_to_json(_map()))
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 2"):
        conformance_map_from_json(json.dumps(raw))


def test_oracle_extra_distributions_keeps_only_the_oracle_extra() -> None:
    """Only requirements whose marker selects the `oracle` extra are named, in input order."""
    requirements = [
        "numpy>=1.26",
        'ruff>=0.6; extra == "dev"',
        'statsmodels==0.15.0; extra == "oracle"',
        "empyrical-reloaded==0.5.12 ; extra == 'oracle'",
        'py_vollib==1.0.12; python_version >= "3.9" and extra == "oracle"',
    ]
    assert oracle_extra_distributions(requirements) == [
        "statsmodels",
        "empyrical-reloaded",
        "py_vollib",
    ]


def test_oracle_versions_covers_every_oracle_library_and_the_numerical_base() -> None:
    """Every distribution of the installed `oracle` extra is recorded, with numpy and scipy."""
    versions = oracle_versions()
    assert list(versions) == sorted(versions)
    for distribution in (
        "numpy",
        "scipy",
        "statsmodels",
        "arch",
        "scikit-learn",
        "linearmodels",
        "empyrical-reloaded",
        "quantstats",
        "QuantLib",
        "py_vollib",
        "hmmlearn",
        "pykalman",
        "cvxpy",
        "pandas",
        "hypothesis",
    ):
        assert versions[distribution], distribution


def test_oracle_versions_fails_when_the_oracle_extra_is_not_declared(monkeypatch) -> None:
    """Metadata with no `oracle` extra cannot enumerate the oracle libraries, and says so."""
    monkeypatch.setattr(
        build_report_module.importlib.metadata, "requires", lambda name: ["numpy>=1.26"]
    )
    with pytest.raises(RuntimeError, match="'oracle' extra"):
        oracle_versions()


# ---------------------------------------------------------------------------
# Regenerating a long input from its seed
# ---------------------------------------------------------------------------


def _regenerate(lines: list[str], seed: int, n: int) -> dict[str, Any]:
    """Execute a regeneration recipe's lines the way a repro snippet does."""
    namespace: dict[str, Any] = {"np": np, "rng": np.random.default_rng(seed), "n": n}
    exec("\n".join(lines), namespace)
    return namespace


@settings(max_examples=40, deadline=None, derandomize=True)
@given(tagged=returns_series(min_len=1, max_len=60))
def test_returns_recipe_reproduces_the_returns_generator(tagged) -> None:
    """The plain-numpy lines rebuild `returns_series`'s array bit for bit from its seed."""
    recipe = REGEN_RECIPES["maximum_drawdown"]
    namespace = _regenerate(recipe.lines(tagged.family), tagged.seed, tagged.n)
    np.testing.assert_array_equal(namespace["returns"], tagged.values)


def test_returns_recipe_knows_every_return_family() -> None:
    """Every family `returns_series` can draw has a regeneration recipe."""
    for family in RETURN_FAMILIES:
        assert REGEN_RECIPES["calmar_ratio"].lines(family)
    with pytest.raises(ValueError, match="no returns_series regeneration recipe"):
        REGEN_RECIPES["calmar_ratio"].lines("unknown_family")


@settings(max_examples=30, deadline=None, derandomize=True)
@given(pair=cointegration_pair(min_len=2, max_len=60))
def test_cointegration_recipe_reproduces_the_pair_generator(pair) -> None:
    """The plain-numpy lines rebuild `cointegration_pair`'s two series bit for bit."""
    recipe = REGEN_RECIPES["engle_granger_test"]
    namespace = _regenerate(recipe.lines(pair.family), pair.seed, pair.y.shape[0])
    np.testing.assert_array_equal(namespace["y"], pair.y)
    np.testing.assert_array_equal(namespace["x"], pair.x)
    assert set(COINTEGRATION_FAMILIES) >= {pair.family}


@settings(max_examples=30, deadline=None, derandomize=True)
@given(drawn=test_tier2_filtering.hedge_ratio_problem(min_len=2, max_len=60))
def test_hedge_ratio_recipe_reproduces_the_hedge_generator(drawn) -> None:
    """The plain-numpy lines rebuild `hedge_ratio_problem`'s two series bit for bit."""
    y, x, family, seed = drawn
    recipe = REGEN_RECIPES["dynamic_hedge_ratio"]
    namespace = _regenerate(recipe.lines(family), seed, y.shape[0])
    np.testing.assert_array_equal(namespace["y"], y)
    np.testing.assert_array_equal(namespace["x"], x)


@settings(max_examples=30, deadline=None, derandomize=True)
@given(drawn=test_tier1_statistics_extra._adf_series())
def test_adf_recipe_reproduces_the_adf_series_generator(drawn) -> None:
    """The plain-numpy lines rebuild `_adf_series`'s series bit for bit."""
    series, family, seed, n = drawn
    recipe = REGEN_RECIPES["adf_test"]
    namespace = _regenerate(recipe.lines(family), seed, n)
    np.testing.assert_array_equal(namespace["series"], series)


def test_every_repro_config_has_a_recipe_for_its_array_arguments() -> None:
    """Each repro configuration can regenerate a long input: its recipe binds its own arguments."""
    for name, config in REPRO_CONFIGS.items():
        assert name in REGEN_RECIPES, name
        assert set(REGEN_RECIPES[name].arrays) <= set(config.arg_names), name


def test_render_minimal_repro_regenerates_arrays_and_inlines_scalars() -> None:
    """A long input's array is regenerated at its own length; the scalar argument is inlined.

    The length comes from the array argument alone, not from the total count of numbers in the
    call, and the non-array argument keeps its recorded value.
    """
    values = ", ".join(str(float(i)) for i in range(40))
    detail = _rd(
        function="calmar_ratio",
        seed=7,
        family="tiny_scale",
        inputs_repr=f"(array([{values}]), 52,)",
    )
    code = render_minimal_repro("calmar_ratio", "value", detail)
    assert "rng = np.random.default_rng(7)" in code
    assert "n = 40" in code
    assert "values = rng.normal(0.0, 1e-8, n)" in code
    assert "periods_per_year = 52" in code


def test_render_minimal_repro_sizes_a_summarised_array_from_its_shape() -> None:
    """An array numpy printed in abbreviated form is regenerated at the length its shape states."""
    detail = _rd(
        function="adf_test",
        quantity="p_value",
        seed=3,
        family="unit_root",
        inputs_repr="(array([0.1, 0.2, 0.3, ..., 0.7, 0.8, 0.9], shape=(2265,)), 2,)",
    )
    code = render_minimal_repro("adf_test", "p_value", detail)
    assert "n = 2265" in code
    assert "series = np.cumsum(rng.normal(0.0, 1.0, n))" in code
    assert "max_lags = 2" in code


def test_render_minimal_repro_raises_for_a_long_input_without_a_recipe(monkeypatch) -> None:
    """A long input of a target with no regeneration recipe fails loudly, naming the target."""
    monkeypatch.delitem(build_report_module.REGEN_RECIPES, "maximum_drawdown")
    values = ", ".join(str(float(i)) for i in range(15))
    detail = _rd(function="maximum_drawdown", inputs_repr=f"(array([{values}]),)")
    with pytest.raises(ValueError, match=r"maximum_drawdown\.value.*no recipe"):
        render_minimal_repro("maximum_drawdown", "value", detail)


def _run_repro_in_process(code: str) -> tuple[float, float]:
    """Execute a rendered repro in this interpreter and return its printed (expected, actual)."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(code, {"__name__": "__repro__"})
    expected, actual, _abs_diff = parse_repro_stdout(stdout.getvalue())
    return expected, actual


def _repro_inputs(name: str, seed: int, n: int, family: str, scalars: dict[str, Any]) -> tuple:
    """Build one target's call arguments from its regeneration recipe and the given scalars."""
    recipe = REGEN_RECIPES[name]
    namespace = _regenerate(recipe.lines(family), seed, n)
    return tuple(
        namespace[arg] if arg in recipe.arrays else scalars[arg]
        for arg in REPRO_CONFIGS[name].arg_names
    )


@pytest.mark.parametrize(
    ("name", "quantity", "n", "family", "scalars", "component"),
    [
        ("maximum_drawdown", "value", 40, "normal", {}, None),
        ("calmar_ratio", "value", 40, "normal", {"periods_per_year": 52}, None),
        ("calmar_ratio_arithmetic", "value", 40, "normal", {"periods_per_year": 52}, None),
        ("engle_granger_test", "statistic", 60, "cointegrated", {}, 0),
        ("engle_granger_test", "p_value", 60, "cointegrated", {}, 1),
        ("adf_test", "statistic", 60, "stationary", {"max_lags": 1}, 0),
        ("adf_test", "p_value", 60, "stationary", {"max_lags": 1}, 1),
        ("fit_garch_11", "log_likelihood", 400, "garch11", {}, 0),
        ("egarch_fit", "log_likelihood", 500, "garch11", {}, 0),
        ("gjr_garch_fit", "log_likelihood", 400, "garch11", {}, 0),
        (
            "dynamic_hedge_ratio",
            "value",
            40,
            "stable_hedge",
            {"obs_var": 1e-3, "proc_var": 1e-5},
            -1,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_repro_config_computes_what_the_registered_callables_compute(
    name: str, quantity: str, n: int, family: str, scalars: dict[str, Any], component: int | None
) -> None:
    """A rendered repro prints the registered oracle's and quantcore adapter's own numbers.

    Each `ReproCallConfig` restates a target's two callables with public imports only. Run on an
    input regenerated from a seed, the snippet's `expected` must equal the registered `reference`
    and its `actual` the registered `quantcore`, or the finding it backs would show a different
    comparison from the one that was recorded.
    """
    seed = 11
    args = _repro_inputs(name, seed, n, family, scalars)
    detail = _rd(function=name, quantity=quantity, seed=seed, family=family)
    detail = dataclasses.replace(detail, inputs_repr=_repr_inputs(args))
    target = registry()[name]
    assert target.quantcore is not None
    assert target.reference is not None

    expected, actual = _run_repro_in_process(render_minimal_repro(name, quantity, detail))

    reference_value = target.reference(*args)
    quantcore_value = target.quantcore(*args)
    if component is not None:
        reference_value = reference_value[component]
        quantcore_value = quantcore_value[component]
    assert expected == pytest.approx(float(reference_value), rel=1e-12, abs=1e-15)
    assert actual == pytest.approx(float(quantcore_value), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# The human-readable report
# ---------------------------------------------------------------------------


def _report(conformance_map: ConformanceMap, **kwargs: Any) -> str:
    return render_conformance_md(conformance_map, [], [], {}, [], [], **kwargs)


def test_render_conformance_md_first_line_says_it_is_generated_and_how() -> None:
    """The report opens by naming the command that generates it."""
    first_line = _report(_map()).splitlines()[0]
    assert "Generated by `uv run python -m conformance.build_report`" in first_line
    assert "Do not edit by hand" in first_line


def test_render_conformance_md_has_one_row_per_quantity_with_status_and_tolerance() -> None:
    """A two-quantity target has two rows carrying the target's status, tolerance and finding."""
    entry = _entry(
        status="defect",
        finding="conformance/findings/001-adf_test.md",
        oracle="statsmodels",
        by_quantity={
            "statistic": QuantityErrors(200, 1.01e-11, 4.65e-14),
            "p_value": QuantityErrors(200, 1e-4, None),
        },
        reason="adf_test's p-value table is off",
    )
    text = _report(_map(targets={"adf_test": entry}))
    assert (
        "| `adf_test` | `p_value` | statsmodels | 200 | 1.000e-04 | — | 1e-10 | 1e-09 | defect | "
        "`conformance/findings/001-adf_test.md` |"
    ) in text
    assert "| `adf_test` | `statistic` | statsmodels | 200 | 1.010e-11 | 4.650e-14 |" in text
    assert "- `adf_test` (`defect`): adf_test's p-value table is off" in text
    assert "(../conformance/findings_src/adf_test.md)" in text


def test_render_conformance_md_counts_targets_per_status() -> None:
    """The summary table counts the targets of each status."""
    text = _report(
        _map(
            targets={
                "a": _entry(),
                "b": _entry(),
                "c": _entry(status="defect"),
                "d": _entry(status="unverifiable"),
            }
        )
    )
    for row in (
        "| verified | 2 |",
        "| non-standard | 0 |",
        "| defect | 1 |",
        "| unverifiable | 1 |",
        "| total | 4 |",
    ):
        assert row in text


def test_render_conformance_md_escapes_a_pipe_in_an_oracle_name() -> None:
    """A `|` in an oracle description does not split its table cell."""
    entry = _entry(oracle="hand-computed (mean / |max drawdown|)")
    text = _report(_map(targets={"calmar_ratio_arithmetic": entry}))
    assert "hand-computed (mean / \\|max drawdown\\|)" in text


def test_render_conformance_md_shows_a_target_without_an_oracle_with_dashes() -> None:
    """A target that was never compared has one row with no quantity, errors or tolerance."""
    entry = dataclasses.replace(
        _entry(
            status="unverifiable", cases=0, max_abs_error=None, max_rel_error=None, oracle="none"
        ),
        tolerance=None,
        reason=build_report_module.NO_ORACLE_REASON,
    )
    text = _report(_map(targets={"select_hmm_n_states": entry}))
    assert "| `select_hmm_n_states` | — | none | 0 | — | — | — | — | unverifiable | — |" in text
    assert (
        f"- `select_hmm_n_states` (`unverifiable`): {build_report_module.NO_ORACLE_REASON}" in text
    )


def test_render_conformance_md_is_free_of_local_paths_and_reproducible() -> None:
    """Two renderings of the same map are identical and name no directory of this machine."""
    conformance_map = _map(targets={"b": _entry(), "a": _entry(status="defect")})
    text = _report(conformance_map)
    assert text == _report(conformance_map)
    assert str(build_report_module.REPO_ROOT) not in text
    assert text.index("| `a` |") < text.index("| `b` |")
    assert text.endswith("\n")


def test_render_conformance_md_names_the_targets_an_incomplete_build_left_out() -> None:
    """A report built with `--allow-incomplete` says so and lists what is missing from it."""
    text = _report(_map(), left_out={"kpss": ["kpss: registered without a quantcore callable"]})
    assert "## Left out of this build" in text
    assert "- kpss: registered without a quantcore callable" in text
    assert "## Left out of this build" not in _report(_map())


def test_render_conformance_md_reports_cross_checks_with_rounded_values() -> None:
    """A disagreeing second oracle is shown with its example rounded, not at full precision."""
    cases = [_cr(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats", within=False)]
    details = [
        dataclasses.replace(
            _rd(function="sharpe_ratio", quantity="rf_zero", oracle="quantstats"),
            quantcore_value="1.2345678901234567",
            reference_value="2.3456789012345678",
            abs_err=1.1111110111111111,
        )
    ]
    text = render_conformance_md(
        _map(),
        cases,
        details,
        per_function_counts(cases),
        oracle_cross_check_rows(cases, details),
        oracle_disagreement_examples(cases, details),
    )
    assert "| `sharpe_ratio` | `quantstats` | `rf_zero` | 1 | 1.110e+00 |" in text
    assert "second oracle value = `1.23457`, reference value = `2.34568`" in text
    assert "1.2345678901234567" not in text


# ---------------------------------------------------------------------------
# main(): the command end to end, on a toy registry in a temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_suite(tmp_path, monkeypatch) -> SimpleNamespace:
    """Point the builder at a temporary directory holding one toy target's recorded run.

    `toy_ratio` has both callables, a tolerance and declared coverage; its three recorded cases
    are all within tolerance. The returned namespace gives the paths the builder reads and
    writes, and `write_results` to replace the recorded run.
    """
    results = tmp_path / "results"
    results.mkdir()
    paths = SimpleNamespace(
        raw=results / "raw.json",
        details=results / "details.json",
        meta=results / "meta.json",
        map=tmp_path / "conformance_map.json",
        report=tmp_path / "docs" / "CONFORMANCE.md",
        findings=tmp_path / "findings",
        findings_src=tmp_path / "findings_src",
    )
    paths.report.parent.mkdir()
    paths.findings_src.mkdir()
    monkeypatch.setattr(build_report_module, "RAW_RESULTS_PATH", paths.raw)
    monkeypatch.setattr(build_report_module, "DETAILS_PATH", paths.details)
    monkeypatch.setattr(build_report_module, "META_PATH", paths.meta)
    monkeypatch.setattr(build_report_module, "CONFORMANCE_MAP_PATH", paths.map)
    monkeypatch.setattr(build_report_module, "CONFORMANCE_MD_PATH", paths.report)
    monkeypatch.setattr(build_report_module, "FINDINGS_DIR", paths.findings)
    monkeypatch.setattr(build_report_module, "FINDINGS_SRC_DIR", paths.findings_src)
    monkeypatch.delenv("QUANTCORE_CONFORMANCE_PROFILE", raising=False)

    targets = {"toy_ratio": _fake_target(quantcore=_noop, reference=_noop, oracle="hand formula")}
    monkeypatch.setattr(build_report_module, "registry", lambda: targets)
    monkeypatch.setattr(
        build_report_module,
        "DECLARED_COVERAGE",
        {"toy_ratio": DeclaredCoverage(families=("normal",), buckets=("short",))},
    )
    monkeypatch.setitem(
        build_report_module.TOLERANCES,
        "toy_ratio",
        Tolerance(rtol=1e-6, atol=1e-8, justification="toy tolerance"),
    )

    def write_results(
        cases: list[CaseResult], details: list[ReproDetail], profile: str = "ci"
    ) -> None:
        paths.raw.write_text(json.dumps([dataclasses.asdict(c) for c in cases]), encoding="utf-8")
        paths.details.write_text(
            json.dumps([dataclasses.asdict(d) for d in details]), encoding="utf-8"
        )
        paths.meta.write_text(json.dumps({"hypothesis_profile": profile}), encoding="utf-8")

    paths.write_results = write_results
    paths.targets = targets
    write_results(*_toy_run("toy_ratio", within=[True, True, True]))
    return paths


def _written_files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def test_main_writes_the_map_and_the_report_then_check_passes(toy_suite, capsys) -> None:
    """A clean run writes both files; `--check` on the same results then exits zero."""
    assert build_report_module.main([]) == 0

    written = conformance_map_from_json(toy_suite.map.read_text(encoding="utf-8"))
    assert written.targets["toy_ratio"].status == "verified"
    assert written.hypothesis_profile == "ci"
    assert toy_suite.report.read_text(encoding="utf-8").startswith("_Generated by `uv run python")

    assert build_report_module.main(["--check"]) == 0
    assert "conformance_map.json is up to date." in capsys.readouterr().out


def test_main_check_writes_nothing(toy_suite, tmp_path) -> None:
    """`--check` creates no file, whether or not a committed map exists."""
    before = _written_files(tmp_path)
    assert build_report_module.main(["--check"]) == 1  # no committed map yet
    assert _written_files(tmp_path) == before

    assert build_report_module.main([]) == 0
    after_build = {name: (tmp_path / name).read_bytes() for name in _written_files(tmp_path)}
    assert build_report_module.main(["--check"]) == 0
    assert {
        name: (tmp_path / name).read_bytes() for name in _written_files(tmp_path)
    } == after_build


def test_main_check_fails_when_a_decision_differs_from_the_committed_map(toy_suite, capsys) -> None:
    """A committed map recording another status makes `--check` exit non-zero, naming the field."""
    assert build_report_module.main([]) == 0
    committed = json.loads(toy_suite.map.read_text(encoding="utf-8"))
    committed["targets"]["toy_ratio"]["status"] = "defect"
    toy_suite.map.write_text(json.dumps(committed), encoding="utf-8")
    capsys.readouterr()

    assert build_report_module.main(["--check"]) == 1

    err = capsys.readouterr().err
    assert "is stale (decisions changed)" in err
    assert "toy_ratio.status: defect -> verified" in err


def test_main_check_ignores_error_magnitudes_and_case_counts(toy_suite) -> None:
    """A committed map differing only in evidence figures still passes `--check`."""
    assert build_report_module.main([]) == 0
    committed = json.loads(toy_suite.map.read_text(encoding="utf-8"))
    committed["targets"]["toy_ratio"]["cases"] = 5000
    committed["targets"]["toy_ratio"]["max_abs_error"] = 9.99e-9
    committed["oracle_versions"] = {"statsmodels": "0.0.1"}
    toy_suite.map.write_text(json.dumps(committed), encoding="utf-8")

    assert build_report_module.main(["--check"]) == 0


def test_main_check_fails_when_the_committed_map_lacks_a_target(toy_suite, capsys) -> None:
    """A target the committed map does not know makes `--check` exit non-zero."""
    assert build_report_module.main([]) == 0
    committed = json.loads(toy_suite.map.read_text(encoding="utf-8"))
    committed["targets"]["old_target"] = committed["targets"].pop("toy_ratio")
    toy_suite.map.write_text(json.dumps(committed), encoding="utf-8")
    capsys.readouterr()

    assert build_report_module.main(["--check"]) == 1

    err = capsys.readouterr().err
    assert "old_target: removed from the regenerated map" in err
    assert "toy_ratio: added in the regenerated map" in err


def test_main_fails_on_an_empty_declared_coverage_cell(toy_suite, monkeypatch, capsys) -> None:
    """A declared family with no recorded case fails the build and the check, naming the cell."""
    monkeypatch.setattr(
        build_report_module,
        "DECLARED_COVERAGE",
        {"toy_ratio": DeclaredCoverage(families=("normal", "ar1"), buckets=("short",))},
    )
    for argv in ([], ["--check"]):
        assert build_report_module.main(argv) == 1
        assert "toy_ratio: no recorded cases for family 'ar1'" in capsys.readouterr().err
    assert not toy_suite.map.exists()


def test_main_fails_when_too_many_cases_are_ill_conditioned(toy_suite, capsys) -> None:
    """Fewer than 90% well-conditioned cases fails the build, with or without the flag."""
    toy_suite.write_results(
        *_toy_run("toy_ratio", within=[True, True, True], ill=[True, False, False])
    )
    for argv in ([], ["--allow-incomplete"], ["--check"]):
        assert build_report_module.main(argv) == 1
        assert "toy_ratio: 1/3 cases ill_conditioned" in capsys.readouterr().err
    assert not toy_suite.map.exists()


def test_main_names_every_incomplete_target_at_once(toy_suite, capsys) -> None:
    """Targets without a quantcore callable or a coverage line are all listed in one failure."""
    toy_suite.targets["ulcer_index"] = _fake_target(reference=_noop)
    toy_suite.targets["kpss"] = _fake_target(reference=_noop)

    assert build_report_module.main([]) == 1

    err = capsys.readouterr().err
    assert "2 target(s) are incomplete" in err
    assert "kpss: registered without a quantcore callable" in err
    assert "ulcer_index: registered without a quantcore callable" in err
    assert "ulcer_index: registered with an oracle but missing from COVERAGE_SOURCES" in err
    assert "--allow-incomplete" in err
    assert not toy_suite.map.exists()


def test_main_allow_incomplete_builds_from_the_complete_targets(toy_suite, capsys) -> None:
    """With the flag the incomplete targets are listed, left out, and named in the report."""
    toy_suite.targets["ulcer_index"] = _fake_target(reference=_noop)

    assert build_report_module.main(["--allow-incomplete"]) == 0

    assert "leaving 1 incomplete target(s) out of the map" in capsys.readouterr().err
    written = conformance_map_from_json(toy_suite.map.read_text(encoding="utf-8"))
    assert set(written.targets) == {"toy_ratio"}
    report = toy_suite.report.read_text(encoding="utf-8")
    assert "## Left out of this build" in report
    assert "ulcer_index: registered without a quantcore callable" in report

    assert build_report_module.main(["--check", "--allow-incomplete"]) == 0
    assert build_report_module.main(["--check"]) == 1


def test_main_fails_when_a_defect_has_no_finding_source(toy_suite, capsys) -> None:
    """A `defect` target without `findings_src/<name>.md` fails the build and the check."""
    toy_suite.write_results(*_toy_run("toy_ratio", within=[True, False]))
    for argv in ([], ["--check"]):
        assert build_report_module.main(argv) == 1
        err = capsys.readouterr().err
        assert "missing a findings_src/ file" in err
        assert "toy_ratio: missing" in err
    assert not toy_suite.map.exists()


def test_main_writes_a_finding_for_a_defect_and_removes_it_once_verified(toy_suite) -> None:
    """A defect's finding is generated next to the map; a later clean run sweeps it away."""
    (toy_suite.findings_src / "toy_ratio.md").write_text(_FINDING_SOURCE_TEXT, encoding="utf-8")
    toy_suite.write_results(*_toy_run("toy_ratio", within=[True, False]))

    assert build_report_module.main([]) == 0

    finding = toy_suite.findings / "001-toy_ratio.md"
    text = finding.read_text(encoding="utf-8")
    assert text.startswith(GENERATED_FILE_COMMENT)
    assert "- cases run: 2; failing: 1" in text
    written = conformance_map_from_json(toy_suite.map.read_text(encoding="utf-8"))
    assert written.targets["toy_ratio"].finding == "conformance/findings/001-toy_ratio.md"
    assert written.targets["toy_ratio"].reason == "toy_ratio is off by a half"

    toy_suite.write_results(*_toy_run("toy_ratio", within=[True, True]))
    assert build_report_module.main([]) == 0
    assert not finding.exists()


def test_main_explore_profile_prints_mismatches_and_writes_nothing(
    toy_suite, tmp_path, monkeypatch, capsys
) -> None:
    """Under `explore` a mismatch for a `verified` target is printed and nothing is written."""
    assert build_report_module.main([]) == 0
    monkeypatch.setenv("QUANTCORE_CONFORMANCE_PROFILE", "explore")
    before = {name: (tmp_path / name).read_bytes() for name in _written_files(tmp_path)}

    toy_suite.write_results(*_toy_run("toy_ratio", within=[True, True]), profile="explore")
    assert build_report_module.main([]) == 0
    assert "no mismatches recorded" in capsys.readouterr().out

    toy_suite.write_results(*_toy_run("toy_ratio", within=[True, False]), profile="explore")
    recorded = {toy_suite.raw.name, toy_suite.details.name, toy_suite.meta.name}
    assert build_report_module.main([]) == 1
    err = capsys.readouterr().err
    assert "1 mismatch(es) recorded for verified target(s)" in err
    assert "toy_ratio.value family='normal' length_bucket='short' seed=1" in err

    after = {
        name: (tmp_path / name).read_bytes()
        for name in _written_files(tmp_path)
        if Path(name).name not in recorded
    }
    assert after == {name: data for name, data in before.items() if Path(name).name not in recorded}
    assert build_report_module.main(["--check"]) == 1
    assert "meaningless combined with the explore profile" in capsys.readouterr().err


def test_main_refuses_results_recorded_under_another_profile(toy_suite, monkeypatch) -> None:
    """Results recorded under one profile cannot be built under another."""
    toy_suite.write_results(*_toy_run("toy_ratio", within=[True]), profile="full")
    with pytest.raises(SystemExit, match="records hypothesis_profile='full'"):
        build_report_module.main([])

    monkeypatch.setenv("QUANTCORE_CONFORMANCE_PROFILE", "full")
    assert build_report_module.main([]) == 0
    written = conformance_map_from_json(toy_suite.map.read_text(encoding="utf-8"))
    assert written.hypothesis_profile == "full"


def test_main_says_how_to_produce_missing_results(toy_suite) -> None:
    """Without recorded results the builder names the command that produces them."""
    toy_suite.raw.unlink()
    with pytest.raises(SystemExit, match="uv run pytest conformance"):
        build_report_module.main([])


def test_results_dir_follows_the_environment_variable(tmp_path) -> None:
    """`QUANTCORE_CONFORMANCE_RESULTS_DIR` names the directory the recorded run is read from."""
    assert build_report_module.results_dir({}) == build_report_module.CONFORMANCE_DIR / "results"
    assert (
        build_report_module.results_dir({"QUANTCORE_CONFORMANCE_RESULTS_DIR": str(tmp_path)})
        == tmp_path
    )
