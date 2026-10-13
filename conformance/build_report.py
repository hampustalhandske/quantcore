"""`python -m conformance.build_report`: turn recorded conformance cases into the shipped record.

Reads `raw.json`, `details.json` and `meta.json` from the results directory the pytest plugin in
`conformance/conftest.py` wrote after `uv run pytest conformance` (`conformance/results/`, or the
directory named by `QUANTCORE_CONFORMANCE_RESULTS_DIR`), gives every registered
`ConformanceTarget` a status, and writes:

- `conformance/conformance_map.json`: the committed record of every target's status, oracle,
  case count, worst errors, tolerance and finding.
- `docs/CONFORMANCE.md`: the same record as a human-readable report, with coverage counts, the
  iterative-estimator sign tests, the condition-aware tolerance counts and the second-oracle
  cross-checks.
- `conformance/findings/NNN-<target>.md` (git-ignored): one write-up per `defect` or
  `non-standard` target, merging the hand-written prose in `conformance/findings_src/<target>.md`
  with this run's evidence (case counts, worst error, a minimal repro that is executed in a
  subprocess before being written).

`--check` rebuilds the map in memory and exits non-zero if its decisions differ from the committed
map, without touching anything on disk: the CI gate against a stale record. It compares decisions
only (see `decision_diff`) and neither renders nor executes findings.

The build refuses to run, naming every problem at once, when the registry is misconfigured (see
`configuration_errors`), when a declared family or length bucket recorded no case, or when too
many of a target's cases are ill-conditioned. `--allow-incomplete` turns the first two into
warnings and leaves the affected targets out of the map, so the rest can be built while a target
is still being added.

Under the `explore` profile nothing is written: the run's mismatches against the targets the
committed map calls `verified` are printed, and the exit code is non-zero if there is one.
"""

import argparse
import ast
import dataclasses
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np

import conformance.targets  # noqa: F401 - registers every ConformanceTarget
from conformance import test_tier1_execution as tier1_execution
from conformance import test_tier1_performance as tier1_performance
from conformance import test_tier1_performance_extra as tier1_performance_extra
from conformance import test_tier1_risk as tier1_risk
from conformance import test_tier1_statistics as tier1_statistics
from conformance import test_tier1_statistics_extra as tier1_statistics_extra
from conformance import test_tier2_filtering as tier2_filtering
from conformance import test_tier2_portfolio as tier2_portfolio
from conformance import test_tier2_pricing_extra as tier2_pricing_extra
from conformance import test_tier2_regime as tier2_regime
from conformance import test_tier2_risk_models as tier2_risk_models
from conformance import test_tier2_time_series as tier2_time_series
from conformance.classify import (
    SIGN_TEST_ALPHA,
    TIE_ABS_GAP,
    CaseResult,
    ConformanceStatus,
    IterativeEstimatorDecision,
    ObjectiveCase,
    classify,
    classify_iterative_estimator,
    tally_objective_outcomes,
)
from conformance.compare import ConformanceTarget, Tolerance, registry
from conformance.findings import FindingSource, load_finding_source
from conformance.generators import COVARIANCE_FAMILIES, length_bucket_for
from conformance.harness import ReproDetail
from conformance.tolerances import ILL_CONDITIONED_RTOL_CAP, TOLERANCES

# Below this magnitude a quantity's oracle (reference) value is indistinguishable from zero, so a
# relative error computed against it is not meaningful and would otherwise let a single near-zero
# denominator dominate the aggregate (dividing by the `1e-300` comparator floor can otherwise print
# as a spurious `~1e+296` in a generated report).
_ZERO_FLOOR = 1e-12

# Significant digits every aggregated error magnitude is rounded to before it is stored in the map
# or printed in a report.
_ERROR_SIG_DIGITS = 3

CONFORMANCE_DIR = Path(__file__).parent
REPO_ROOT = CONFORMANCE_DIR.parent


def results_dir(environ: Mapping[str, str]) -> Path:
    """Directory a suite run's results are read from, by the rule `conformance/conftest.py` writes.

    Args:
        environ: The process environment, ordinarily `os.environ`.

    Returns:
        The directory `QUANTCORE_CONFORMANCE_RESULTS_DIR` names, or `conformance/results/` when
        that variable is not set.
    """
    return Path(environ.get("QUANTCORE_CONFORMANCE_RESULTS_DIR", CONFORMANCE_DIR / "results"))


RESULTS_DIR = results_dir(os.environ)
RAW_RESULTS_PATH = RESULTS_DIR / "raw.json"
DETAILS_PATH = RESULTS_DIR / "details.json"
META_PATH = RESULTS_DIR / "meta.json"
FINDINGS_DIR = CONFORMANCE_DIR / "findings"
# Where the map's `finding` paths point, relative to the repository root.
FINDINGS_RELATIVE_DIR = "conformance/findings"
FINDINGS_SRC_DIR = CONFORMANCE_DIR / "findings_src"
CONFORMANCE_MAP_PATH = CONFORMANCE_DIR / "conformance_map.json"
CONFORMANCE_MD_PATH = REPO_ROOT / "docs" / "CONFORMANCE.md"

BUILD_COMMAND = "uv run python -m conformance.build_report"

GENERATED_FILE_COMMENT = (
    "<!-- generated by conformance.build_report; edit conformance/findings_src/ instead -->"
)

SCHEMA_VERSION = 1

# Distributions recorded in `oracle_versions` besides the ones the `oracle` extra pins: several
# oracles are closed forms or hand recursions built directly on these two.
_NUMERICAL_BASE_DISTRIBUTIONS = ("numpy", "scipy")

_ORACLE_EXTRA_RE = re.compile(r"""extra\s*==\s*['"]oracle['"]""")
_REQUIREMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")

type HypothesisProfile = Literal["ci", "full", "explore"]
type MapWritableProfile = Literal["ci", "full"]


# ---------------------------------------------------------------------------
# The conformance map
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToleranceRecord:
    """The tolerance a target's cases were judged against, as stored in the map.

    Attributes:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        justification: Why this tolerance (not a tighter one) is correct for this target.
    """

    rtol: float
    atol: float
    justification: str


@dataclasses.dataclass(frozen=True)
class QuantityErrors:
    """Conformance evidence for one named quantity a target records (e.g. `"p_value"`).

    A target returning several named results has one per name, so a defect in one quantity is not
    averaged away.

    Attributes:
        cases: Differential test cases recorded for this quantity.
        max_abs_error: Largest absolute error between quantcore and the oracle (3 significant
            digits); `None` if quantcore raised on every case.
        max_rel_error: Largest relative error over cases above the near-zero oracle floor (3
            significant digits); `None` if there is none.
    """

    cases: int
    max_abs_error: float | None
    max_rel_error: float | None


@dataclasses.dataclass(frozen=True)
class TargetRecord:
    """Conformance outcome for one registered target, as stored in the map.

    Attributes:
        status: Conformance status of quantcore's implementation.
        oracle: Independent library or published source quantcore is judged against.
        cases: Generated inputs compared for this target (one input counts once, however many
            quantities it records).
        max_abs_error: Largest absolute error against the oracle over every quantity; `None` if
            nothing was compared.
        max_rel_error: Largest relative error over cases with a non-near-zero oracle magnitude.
        by_quantity: Per-quantity breakdown keyed by quantity name; empty with no cases.
        tolerance: The tolerance quantcore's output was judged against; `None` for a target
            with no entry in `conformance.tolerances.TOLERANCES` (one with no oracle, which is
            never compared).
        reason: One sentence explaining a status other than `verified`; empty for `verified`.
        finding: Repository-relative path of the generated write-up for a `defect` or
            `non-standard` status (e.g. `"conformance/findings/001-adf_test.md"`); `None`
            otherwise.
    """

    status: ConformanceStatus
    oracle: str
    cases: int
    max_abs_error: float | None
    max_rel_error: float | None
    by_quantity: dict[str, QuantityErrors]
    tolerance: ToleranceRecord | None
    reason: str
    finding: str | None


@dataclasses.dataclass(frozen=True)
class ConformanceMap:
    """The generated record of every registered target's conformance status and evidence.

    Attributes:
        schema_version: Format version, bumped on breaking changes.
        quantcore_version: `quantcore` distribution version the map was generated against.
        oracle_versions: Installed version of every oracle distribution (see `oracle_versions`),
            by distribution name.
        hypothesis_profile: Hypothesis profile (`"ci"` or `"full"`) that generated the map.
        targets: Per-target outcome, keyed by target name.
    """

    schema_version: int
    quantcore_version: str
    oracle_versions: dict[str, str]
    hypothesis_profile: MapWritableProfile
    targets: dict[str, TargetRecord]


def conformance_map_to_json(conformance_map: ConformanceMap) -> str:
    """Serialise a map to the text written to `conformance_map.json`.

    Keys are sorted at every level and the text ends with a newline, so two builds of the same
    decisions and evidence produce identical files whatever order the targets were registered in.
    """
    return json.dumps(dataclasses.asdict(conformance_map), indent=2, sort_keys=True) + "\n"


def conformance_map_from_json(text: str) -> ConformanceMap:
    """Parse the text of a `conformance_map.json` back into a `ConformanceMap`.

    Raises:
        ValueError: The text is not a map of the supported `schema_version`.
        KeyError: A field the schema requires is missing.
    """
    raw = json.loads(text)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported conformance map schema_version {raw.get('schema_version')!r}; "
            f"this builder reads version {SCHEMA_VERSION}"
        )
    targets: dict[str, TargetRecord] = {}
    for name, entry in raw["targets"].items():
        tolerance = entry["tolerance"]
        tolerance_record = (
            None
            if tolerance is None
            else ToleranceRecord(
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                justification=tolerance["justification"],
            )
        )
        targets[name] = TargetRecord(
            status=entry["status"],
            oracle=entry["oracle"],
            cases=entry["cases"],
            max_abs_error=entry["max_abs_error"],
            max_rel_error=entry["max_rel_error"],
            by_quantity={
                quantity: QuantityErrors(
                    cases=errors["cases"],
                    max_abs_error=errors["max_abs_error"],
                    max_rel_error=errors["max_rel_error"],
                )
                for quantity, errors in entry["by_quantity"].items()
            },
            tolerance=tolerance_record,
            reason=entry["reason"],
            finding=entry["finding"],
        )
    return ConformanceMap(
        schema_version=raw["schema_version"],
        quantcore_version=raw["quantcore_version"],
        oracle_versions=dict(raw["oracle_versions"]),
        hypothesis_profile=raw["hypothesis_profile"],
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Declared coverage
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeclaredCoverage:
    """One target's declared conformance coverage, read from its tier test module.

    Attributes:
        families: Generator families the target's conformance test draws inputs from.
        buckets: Length buckets (`generators.length_bucket_for`) the target's test can reach
            once lengths are clipped from below to the target's minimum; a bucket entirely
            below that minimum is not declared.
        min_len: Shortest input the target's test draws, or `None` for a target whose test
            module declares no minimum (its input has no length, e.g. scalar arguments).
    """

    families: tuple[str, ...]
    buckets: tuple[str, ...]
    min_len: int | None = None


def declared_coverage_from(module: ModuleType, prefix: str) -> DeclaredCoverage:
    """Read one target's coverage declaration from the constants of its tier test module.

    Args:
        module: The `conformance.test_tier*` module that tests the target.
        prefix: Prefix of the constants declared there: `<prefix>_FAMILIES`, `<prefix>_BUCKETS`
            and, optionally, `<prefix>_MIN_LEN`.

    Returns:
        The declaration those constants hold.

    Raises:
        LookupError: The module has no `<prefix>_FAMILIES` or no `<prefix>_BUCKETS`.
    """
    missing = [
        constant
        for constant in (f"{prefix}_FAMILIES", f"{prefix}_BUCKETS")
        if not hasattr(module, constant)
    ]
    if missing:
        raise LookupError(
            f"{module.__name__} declares no {' or '.join(missing)}; COVERAGE_SOURCES in "
            "conformance/build_report.py names a prefix that module does not use"
        )
    return DeclaredCoverage(
        families=tuple(getattr(module, f"{prefix}_FAMILIES")),
        buckets=tuple(getattr(module, f"{prefix}_BUCKETS")),
        min_len=getattr(module, f"{prefix}_MIN_LEN", None),
    )


# Where each target's coverage is declared: the tier test module that tests it, and the prefix of
# the constants that module declares for it (`<PREFIX>_FAMILIES`, `<PREFIX>_BUCKETS` and, where
# the input has a length, `<PREFIX>_MIN_LEN`). Targets tested on the same generated inputs share a
# prefix. Every registered target with an oracle needs a line here; `configuration_errors` names
# any that is missing.
COVERAGE_SOURCES: dict[str, tuple[ModuleType, str]] = {
    "sharpe_ratio": (tier1_performance, "SHARPE_RATIO"),
    "maximum_drawdown": (tier1_performance, "MAXIMUM_DRAWDOWN"),
    "newey_west_cov": (tier1_statistics, "NEWEY_WEST_COV"),
    "engle_granger_test": (tier1_statistics, "ENGLE_GRANGER_TEST"),
    "square_root_market_impact": (tier1_execution, "SQUARE_ROOT_MARKET_IMPACT"),
    "short_borrow_cost": (tier1_execution, "SHORT_BORROW_COST"),
    "margin_financing_cost": (tier1_execution, "MARGIN_FINANCING_COST"),
    "futures_roll_yield": (tier1_execution, "FUTURES_ROLL_YIELD"),
    "sortino_ratio": (tier1_performance_extra, "SORTINO_RATIO"),
    "calmar_ratio": (tier1_performance_extra, "CALMAR_RATIO"),
    "calmar_ratio_arithmetic": (tier1_performance_extra, "CALMAR_RATIO_ARITHMETIC"),
    "information_ratio": (tier1_performance_extra, "INFORMATION_RATIO"),
    "hit_rate": (tier1_performance_extra, "HIT_RATE"),
    "profit_factor": (tier1_performance_extra, "PROFIT_FACTOR"),
    "rolling_sharpe": (tier1_performance_extra, "ROLLING_SHARPE"),
    "ulcer_index": (tier1_performance_extra, "ULCER_INDEX"),
    "ulcer_performance_index": (tier1_performance_extra, "ULCER_PERFORMANCE_INDEX"),
    "gain_to_pain": (tier1_performance_extra, "GAIN_TO_PAIN"),
    "common_sense_ratio": (tier1_performance_extra, "COMMON_SENSE_RATIO"),
    "risk_of_ruin": (tier1_performance_extra, "RISK_OF_RUIN"),
    "serenity_index": (tier1_performance_extra, "SERENITY_INDEX"),
    "outlier_win_ratio": (tier1_performance_extra, "OUTLIER_WIN_RATIO"),
    "outlier_loss_ratio": (tier1_performance_extra, "OUTLIER_LOSS_RATIO"),
    "cpc_index": (tier1_performance_extra, "CPC_INDEX"),
    "value_at_risk": (tier1_risk, "VALUE_AT_RISK"),
    "conditional_value_at_risk": (tier1_risk, "CONDITIONAL_VALUE_AT_RISK"),
    "value_at_risk_cornish_fisher": (tier1_risk, "VALUE_AT_RISK_CORNISH_FISHER"),
    "ols_coefficients": (tier1_statistics_extra, "OLS"),
    "ols_r_squared": (tier1_statistics_extra, "OLS"),
    "newey_west_optimal_lags": (tier1_statistics_extra, "NEWEY_WEST_OPTIMAL_LAGS"),
    "ljung_box_test": (tier1_statistics_extra, "LJUNG_BOX_TEST"),
    "adf_test": (tier1_statistics_extra, "ADF_TEST"),
    "jarque_bera": (tier1_statistics_extra, "JARQUE_BERA"),
    "arch_lm": (tier1_statistics_extra, "ARCH_LM"),
    "kpss": (tier1_statistics_extra, "KPSS"),
    "ewma_variance": (tier2_risk_models, "EWMA_VARIANCE"),
    "garch_11_variance": (tier2_risk_models, "GARCH_11_VARIANCE"),
    "egarch_11_variance": (tier2_risk_models, "EGARCH_11_VARIANCE"),
    "gjr_garch_11_variance": (tier2_risk_models, "GJR_GARCH_11_VARIANCE"),
    "fit_garch_11": (tier2_risk_models, "FIT_GARCH_11"),
    "egarch_fit": (tier2_risk_models, "EGARCH_FIT"),
    "gjr_garch_fit": (tier2_risk_models, "GJR_GARCH_FIT"),
    "hmm_fit_transition_matrix": (tier2_regime, "HMM_FIT"),
    "hmm_fit_means": (tier2_regime, "HMM_FIT"),
    "hmm_fit_variances": (tier2_regime, "HMM_FIT"),
    "hmm_fit_initial_probs": (tier2_regime, "HMM_FIT"),
    "hmm_fit_log_likelihood": (tier2_regime, "HMM_FIT"),
    "hmm_decode": (tier2_regime, "HMM_DECODE"),
    "hmm_predict_proba": (tier2_regime, "HMM_PREDICT_PROBA"),
    "hmm_filtered_proba": (tier2_regime, "HMM_FILTERED_PROBA"),
    "kalman_filter_means": (tier2_filtering, "KALMAN_FILTER_MEANS"),
    "kalman_filter_covs": (tier2_filtering, "KALMAN_FILTER_COVS"),
    "kalman_smooth_means": (tier2_filtering, "KALMAN_SMOOTH_MEANS"),
    "kalman_smooth_covs": (tier2_filtering, "KALMAN_SMOOTH_COVS"),
    "dynamic_hedge_ratio": (tier2_filtering, "DYNAMIC_HEDGE_RATIO"),
    "black_scholes_merton_price": (tier2_pricing_extra, "BSM"),
    "bsm_delta": (tier2_pricing_extra, "BSM"),
    "bsm_vega": (tier2_pricing_extra, "BSM"),
    "bs_gamma": (tier2_pricing_extra, "BSM"),
    "bs_theta": (tier2_pricing_extra, "BSM"),
    "bs_rho": (tier2_pricing_extra, "BSM"),
    "implied_volatility": (tier2_pricing_extra, "IV"),
    "heston_cos_call": (tier2_pricing_extra, "HESTON"),
    "heston_cos_put": (tier2_pricing_extra, "HESTON"),
    "monte_carlo_call_price": (tier2_pricing_extra, "MONTE_CARLO"),
    "monte_carlo_put_price": (tier2_pricing_extra, "MONTE_CARLO"),
    "option_expiry_payoff": (tier2_pricing_extra, "OPTION_EXPIRY_PAYOFF"),
    "option_position_pnl": (tier2_pricing_extra, "OPTION_POSITION_PNL"),
    "option_roll_schedule": (tier2_pricing_extra, "OPTION_ROLL_SCHEDULE"),
    "protective_put_sizing": (tier2_pricing_extra, "PROTECTIVE_PUT_SIZING"),
    "sample_covariance": (tier2_portfolio, "SAMPLE_COVARIANCE"),
    "ewma_covariance": (tier2_portfolio, "EWMA_COVARIANCE"),
    "ledoit_wolf_shrinkage": (tier2_portfolio, "LEDOIT_WOLF_SHRINKAGE"),
    "min_variance_weights": (tier2_portfolio, "MIN_VARIANCE_WEIGHTS"),
    "mean_variance_weights": (tier2_portfolio, "MEAN_VARIANCE_WEIGHTS"),
    "unconstrained_mean_variance_weights": (
        tier2_portfolio,
        "UNCONSTRAINED_MEAN_VARIANCE_WEIGHTS",
    ),
    "l1_turnover_penalized_weights": (tier2_portfolio, "L1_TURNOVER_PENALIZED_WEIGHTS"),
    "risk_parity_weights": (tier2_portfolio, "RISK_PARITY_WEIGHTS"),
    "kelly_fraction": (tier2_portfolio, "KELLY_FRACTION"),
    "implied_equilibrium_returns": (tier2_portfolio, "IMPLIED_EQUILIBRIUM_RETURNS"),
    "black_litterman": (tier2_portfolio, "BLACK_LITTERMAN"),
    "factor_loadings_loadings": (tier2_time_series, "FACTOR_LOADINGS"),
    "factor_loadings_standard_errors": (tier2_time_series, "FACTOR_LOADINGS"),
    "factor_loadings_t_stats": (tier2_time_series, "FACTOR_LOADINGS"),
    "factor_loadings_p_values": (tier2_time_series, "FACTOR_LOADINGS"),
    "factor_loadings_r_squared": (tier2_time_series, "FACTOR_LOADINGS"),
    "factor_loadings_residuals": (tier2_time_series, "FACTOR_LOADINGS"),
    "fama_macbeth_mean_risk_premia": (tier2_time_series, "FAMA_MACBETH"),
    "fama_macbeth_standard_errors": (tier2_time_series, "FAMA_MACBETH"),
    "arima_fit": (tier2_time_series, "ARIMA_FIT"),
    "arima_residuals": (tier2_time_series, "ARIMA_RESIDUALS"),
    "arima_forecast": (tier2_time_series, "ARIMA_FORECAST"),
    "select_arima_order": (tier2_time_series, "SELECT_ARIMA_ORDER"),
    "var_fit_coef_matrix": (tier2_time_series, "VAR_FIT"),
    "var_fit_sigma_u": (tier2_time_series, "VAR_FIT"),
    "var_forecast": (tier2_time_series, "VAR_FORECAST"),
    "impulse_response": (tier2_time_series, "IMPULSE_RESPONSE"),
    "select_var_lag_order": (tier2_time_series, "SELECT_VAR_LAG_ORDER"),
    "granger_causality_test": (tier2_time_series, "GRANGER_CAUSALITY_TEST"),
    "ou_half_life": (tier2_time_series, "OU_HALF_LIFE"),
    "spread_zscore": (tier2_time_series, "SPREAD_ZSCORE"),
}

# Declared, per target, by that target's own tier test module. `missing_coverage` checks every
# recorded run against this: a family or bucket declared here with zero recorded `CaseResult`s
# fails the build.
DECLARED_COVERAGE: dict[str, DeclaredCoverage] = {
    name: declared_coverage_from(module, prefix)
    for name, (module, prefix) in COVERAGE_SOURCES.items()
}

# Hand-written prose explaining a recorded second-oracle disagreement, keyed by (target, oracle).
# Every number quoted alongside this text in a generated report comes from the run that produced
# it -- only the prose is fixed.
ORACLE_DISAGREEMENT_EXPLANATIONS: dict[tuple[str, str], str] = {
    ("sharpe_ratio", "quantstats"): (
        tier1_performance.SHARPE_RATIO_QUANTSTATS_DISAGREEMENT_EXPLANATION
    ),
}

# The length buckets `generators.length_bucket_for` assigns, shortest first. A coverage
# declaration may also use other bucket names (`"scalar"`, a matrix-dimension bucket), which have
# no ordering against an input length.
_LENGTH_BUCKET_ORDER: tuple[str, ...] = ("short", "medium", "long")


def unreachable_declared_buckets(declared: Mapping[str, DeclaredCoverage]) -> dict[str, list[str]]:
    """Find every declared length bucket that lies entirely below its target's minimum length.

    Pure and I/O-free. A test that draws no input shorter than `min_len` can never record a case
    in a bucket whose whole range is shorter than that, so declaring one is a mistake in the
    declaration, not a gap in a particular run.

    Args:
        declared: Per-target `DeclaredCoverage`, ordinarily `DECLARED_COVERAGE`.

    Returns:
        Mapping from each target with an unreachable bucket to one line per such bucket, naming
        the target, the bucket and the minimum length; empty when every declared length bucket
        can be reached.
    """
    problems: dict[str, list[str]] = {}
    for name in sorted(declared):
        coverage = declared[name]
        if coverage.min_len is None:
            continue
        lowest_reachable = _LENGTH_BUCKET_ORDER.index(length_bucket_for(coverage.min_len))
        for bucket in coverage.buckets:
            if bucket not in _LENGTH_BUCKET_ORDER:
                continue
            if _LENGTH_BUCKET_ORDER.index(bucket) < lowest_reachable:
                problems.setdefault(name, []).append(
                    f"{name}: declares length bucket {bucket!r}, which lies entirely below its "
                    f"declared minimum length {coverage.min_len}"
                )
    return problems


def configuration_errors(
    targets: Mapping[str, ConformanceTarget[..., Any]],
    declared: Mapping[str, DeclaredCoverage],
    tolerances: Mapping[str, Tolerance],
    recorded_functions: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Find every way the registry, the coverage table and the tolerances disagree.

    Pure and I/O-free. A target cannot be given a status when it has no quantcore callable to
    judge, and a target with an oracle cannot be gated or judged without a coverage declaration
    and a tolerance.

    Args:
        targets: Every registered target, ordinarily `registry()`.
        declared: Per-target `DeclaredCoverage`, ordinarily `DECLARED_COVERAGE`.
        tolerances: Per-target tolerance, ordinarily `conformance.tolerances.TOLERANCES`.
        recorded_functions: Target names that appear in the recorded cases of the run being
            built; a name here that is not registered means the results belong to a different
            state of the suite.

    Returns:
        Mapping from a target name to the messages describing what is wrong with it, each
        message starting with that name. Empty when the configuration is consistent. A name
        that is declared or recorded but not registered is keyed by that name as well.
    """
    errors: dict[str, list[str]] = {}

    def add(name: str, message: str) -> None:
        errors.setdefault(name, []).append(f"{name}: {message}")

    for name in sorted(targets):
        target = targets[name]
        if target.quantcore is None:
            add(name, "registered without a quantcore callable, so there is nothing to judge")
        if target.reference is None:
            if name in declared:
                add(name, "has declared coverage but no oracle to record a case against")
            continue
        if name not in declared:
            add(
                name,
                "registered with an oracle but missing from COVERAGE_SOURCES in "
                "conformance/build_report.py",
            )
        elif not declared[name].families or not declared[name].buckets:
            add(name, "its coverage declaration has no family or no bucket")
        if name not in tolerances:
            add(name, "has no entry in TOLERANCES in conformance/tolerances.py")

    for name in sorted(set(declared) - set(targets)):
        add(name, "in COVERAGE_SOURCES but not a registered target")
    for name in sorted(set(recorded_functions) - set(targets)):
        add(name, "has recorded cases but is not a registered target (results from another run?)")
    for name, messages in unreachable_declared_buckets(declared).items():
        errors.setdefault(name, []).extend(messages)
    return errors


# --- iterative estimators without a unique optimum ---------------------------------------------

type ObjectiveSense = Literal["maximize", "minimize"]


@dataclasses.dataclass(frozen=True)
class IterativeEstimator:
    """How one iterative estimator's attained objective is read from the recorded cases.

    Attributes:
        well_specified_families: Generator families that match the estimator's own assumptions;
            only cases from these decide its status (see `classify_iterative_estimator`).
        quantity: Name of the recorded quantity holding the attained objective.
        sense: Whether a better solution has a larger (`"maximize"`) or a smaller (`"minimize"`)
            value of that quantity. Stated per estimator, with no default, because the sign test
            is only meaningful when a win is the better objective.
    """

    well_specified_families: frozenset[str]
    quantity: str
    sense: ObjectiveSense


# The GARCH-family fitters assume Gaussian innovations: of the families
# `conformance/test_tier2_risk_models.py` draws for them, `"normal"` and `"garch11"` are simulated
# with `rng.normal(...)` innovations (`conformance.generators._simulate_returns`), while
# `"student_t3"` is heavy-tailed and so misspecified under a Gaussian likelihood.
_GAUSSIAN_INNOVATION_FAMILIES = frozenset({"normal", "garch11"})

# The mean-variance optimisers are convex programs, not likelihood fits, so no covariance family
# is misspecified for them: every one counts.
_ALL_COVARIANCE_FAMILIES = frozenset(COVARIANCE_FAMILIES)

# Each fitter records the log-likelihood it attains (larger is better). Each optimiser records the
# value of the objective its oracle optimises, evaluated at each solver's own weights
# (`conformance.targets.portfolio`): portfolio variance for `min_variance_weights`, which the
# program minimises, and mean-variance utility for the other three, which it maximises.
ITERATIVE_ESTIMATORS: dict[str, IterativeEstimator] = {
    "fit_garch_11": IterativeEstimator(_GAUSSIAN_INNOVATION_FAMILIES, "log_likelihood", "maximize"),
    "egarch_fit": IterativeEstimator(_GAUSSIAN_INNOVATION_FAMILIES, "log_likelihood", "maximize"),
    "gjr_garch_fit": IterativeEstimator(
        _GAUSSIAN_INNOVATION_FAMILIES, "log_likelihood", "maximize"
    ),
    "min_variance_weights": IterativeEstimator(_ALL_COVARIANCE_FAMILIES, "objective", "minimize"),
    "mean_variance_weights": IterativeEstimator(_ALL_COVARIANCE_FAMILIES, "objective", "maximize"),
    "unconstrained_mean_variance_weights": IterativeEstimator(
        _ALL_COVARIANCE_FAMILIES, "objective", "maximize"
    ),
    "l1_turnover_penalized_weights": IterativeEstimator(
        _ALL_COVARIANCE_FAMILIES, "objective", "maximize"
    ),
    # Every family `conformance/test_tier2_time_series.py` draws for `arima_fit` is fit at its own
    # true (p, d, q) order, so none is misspecified.
    "arima_fit": IterativeEstimator(
        frozenset(tier2_time_series.ARIMA_FIT_FAMILIES), "log_likelihood", "maximize"
    ),
}


def oriented_objective(value: float, sense: ObjectiveSense) -> float:
    """Return an attained objective on the larger-is-better scale `ObjectiveCase` expects.

    A maximised objective is returned as is and a minimised one is negated, so that in both cases
    a larger returned value is the better solution and the sign test's win is a genuine win.
    """
    return value if sense == "maximize" else -value


def objective_cases_for(
    name: str,
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    estimator: IterativeEstimator,
) -> list[ObjectiveCase]:
    """Build one `ObjectiveCase` per recorded decision-quantity case, for the win/tie/loss rule.

    Pure and I/O-free: reads the estimator's already-recorded `CaseResult`s and `ReproDetail`s
    (see `matched_pairs`) rather than re-running anything.

    Args:
        name: Target name, e.g. `"fit_garch_11"`.
        cases: Every recorded `CaseResult` from one conformance run.
        details: Every recorded `ReproDetail` from the same run.
        estimator: Which quantity holds the attained objective, whether it is maximised or
            minimised, and which families are well-specified.

    Returns:
        One `ObjectiveCase` per recorded case of the estimator's decision quantity, in recording
        order, with both objectives oriented so that larger is better (`oriented_objective`). A
        case where quantcore raised (no attained objective to evaluate at all) is recorded as
        `-inf`, so it always counts as a loss.
    """
    pairs = matched_pairs(cases, details, function=name, quantity=estimator.quantity)
    result: list[ObjectiveCase] = []
    for case, detail in pairs:
        quantcore_objective = (
            float("-inf")
            if detail.quantcore_value is None
            else oriented_objective(float(detail.quantcore_value), estimator.sense)
        )
        result.append(
            ObjectiveCase(
                family=case.family,
                well_specified=case.family in estimator.well_specified_families,
                quantcore_objective=quantcore_objective,
                reference_objective=oriented_objective(
                    float(detail.reference_value), estimator.sense
                ),
            )
        )
    return result


def iterative_estimator_decision(
    name: str,
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    estimator: IterativeEstimator,
) -> IterativeEstimatorDecision:
    """Run the sign test that decides one iterative estimator's status.

    The win/tie/loss margin is the target's own declared `Tolerance`, the same one every other
    comparison of that target uses.

    Args:
        name: Target name; must have an entry in `conformance.tolerances.TOLERANCES`.
        cases: Every recorded `CaseResult` from one conformance run.
        details: Every recorded `ReproDetail` from the same run.
        estimator: How the target's attained objective is read (see `objective_cases_for`).

    Returns:
        The decision, with its counts, p-value and median gap on the larger-is-better scale.

    Raises:
        ValueError: The run recorded no case, or no well-specified case, for the target.
    """
    tolerance = TOLERANCES[name]
    return classify_iterative_estimator(
        objective_cases_for(name, cases, details, estimator),
        rtol=tolerance.rtol,
        atol=tolerance.atol,
    )


def per_function_counts(
    cases: Sequence[CaseResult],
) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """Tally recorded case counts per family and per length bucket, grouped by target.

    Pure and I/O-free: reduces a flat list of `CaseResult`s to the shape `missing_coverage` and the
    report tables need, without touching disk.

    Args:
        cases: Every recorded `CaseResult`, ordinarily every case from one conformance run.

    Returns:
        Mapping from target name to `(family_counts, bucket_counts)`, each a mapping from name to
        how many recorded cases had that family / length bucket. A target with no recorded cases
        at all is absent from the result.
    """
    counts: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    for case in cases:
        family_counts, bucket_counts = counts.setdefault(case.function, ({}, {}))
        family_counts[case.family] = family_counts.get(case.family, 0) + 1
        bucket_counts[case.length_bucket] = bucket_counts.get(case.length_bucket, 0) + 1
    return counts


# At least 90% of a target's cases must be well-conditioned or `build_report` fails.
MIN_WELL_CONDITIONED_FRACTION = 0.9


def ill_conditioned_counts(cases: Sequence[CaseResult]) -> dict[str, tuple[int, int]]:
    """Per-target `(ill_conditioned_count, total_count)`, over the primary comparison only.

    Pure and I/O-free. A target whose tolerance is not condition-aware
    (`conformance.tolerances.ConditionAwareTolerance`) never has an `ill_conditioned` case, so it
    reports `(0, total)` here -- trivially at 100% well-conditioned.

    Args:
        cases: Every recorded `CaseResult`, ordinarily every case from one conformance run.

    Returns:
        Mapping from target name to `(ill_conditioned_count, total_count)`, counting only
        `oracle == ""` (primary quantcore-vs-reference) cases. A target with no recorded primary
        cases is absent.
    """
    counts: dict[str, tuple[int, int]] = {}
    for case in cases:
        if case.oracle != "":
            continue
        ill, total = counts.get(case.function, (0, 0))
        counts[case.function] = (ill + case.ill_conditioned, total + 1)
    return counts


def ill_conditioned_gate_failures(
    counts: dict[str, tuple[int, int]], *, min_well_conditioned_fraction: float
) -> list[str]:
    """List every target whose well-conditioned case fraction falls below the required minimum.

    Pure and I/O-free.

    Args:
        counts: Per-target `(ill_conditioned_count, total_count)`, as `ill_conditioned_counts`
            returns.
        min_well_conditioned_fraction: The minimum fraction of a target's cases that must be
            well-conditioned (0.9).

    Returns:
        One line per target below the threshold, naming its ill-conditioned/total counts and
        the resulting fraction; empty when every target meets the threshold.
    """
    failures: list[str] = []
    for name in sorted(counts):
        ill, total = counts[name]
        if total == 0:
            continue
        well_conditioned_fraction = 1.0 - (ill / total)
        if well_conditioned_fraction < min_well_conditioned_fraction:
            failures.append(
                f"{name}: {ill}/{total} cases ill_conditioned "
                f"({well_conditioned_fraction:.1%} well-conditioned, need "
                f"{min_well_conditioned_fraction:.0%})"
            )
    return failures


def missing_coverage_by_target(
    declared: Mapping[str, DeclaredCoverage],
    counts: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> dict[str, list[str]]:
    """Group every declared (target, family) and (target, bucket) cell with zero cases by target.

    Pure and I/O-free: `build_report.main()` calls this on the real run's counts; unit tests call
    `missing_coverage` on synthetic `declared`/`counts` to exercise the pass and fail paths
    without running the (slow, hypothesis-driven) conformance suite itself.

    Args:
        declared: Per-target `DeclaredCoverage`, ordinarily `DECLARED_COVERAGE`.
        counts: Per-target `(family_counts, bucket_counts)`, as `per_function_counts` returns. A
            target absent here is treated as having zero cases in every family and bucket.

    Returns:
        Mapping from each target with an empty declared cell to one line per such cell, each
        starting with the target's name and naming the empty family or bucket, or to a single
        line when the target has no recorded case at all; a target whose every declared cell
        has at least one recorded case is absent.
    """
    missing: dict[str, list[str]] = {}
    for function in sorted(declared):
        coverage = declared[function]
        family_counts, bucket_counts = counts.get(function, ({}, {}))
        if not family_counts and not bucket_counts:
            missing[function] = [f"{function}: no recorded cases at all"]
            continue
        lines: list[str] = []
        for family in coverage.families:
            if family_counts.get(family, 0) == 0:
                lines.append(f"{function}: no recorded cases for family {family!r}")
        for bucket in coverage.buckets:
            if bucket_counts.get(bucket, 0) == 0:
                lines.append(f"{function}: no recorded cases for length bucket {bucket!r}")
        if lines:
            missing[function] = lines
    return missing


def missing_coverage(
    declared: Mapping[str, DeclaredCoverage],
    counts: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> list[str]:
    """List every declared (target, family) and (target, bucket) cell with zero recorded cases.

    The flat form of `missing_coverage_by_target`: the same lines, in target order.
    """
    by_target = missing_coverage_by_target(declared, counts)
    return [line for lines in by_target.values() for line in lines]


def decision_diff(committed: ConformanceMap, regenerated: ConformanceMap) -> list[str]:
    """Compare two conformance maps on decisions only, ignoring run-to-run numeric noise.

    A "decision" is: the set of target names; and, per target, its `status`, `tolerance` (rtol,
    atol, justification), and `finding`. Error magnitudes, case counts, per-quantity breakdowns,
    and library versions are deliberately excluded: they can differ in the last bits between BLAS
    builds or platforms without the underlying status having changed, so including them would
    make `--check` flag a stale-looking diff on every run rather than only when something that
    actually matters changed.

    Pure and I/O-free: takes two already-parsed `ConformanceMap`s, does not read or write anything.

    Args:
        committed: The map currently on disk (the one CI treats as authoritative).
        regenerated: A freshly rebuilt map from the same run's recorded cases.

    Returns:
        One line per differing field, `"<target>.<field>: <old> -> <new>"` (or a line naming a
        target that was added or removed); empty when the two maps represent the same decisions.
    """
    lines: list[str] = []
    committed_names = set(committed.targets)
    regenerated_names = set(regenerated.targets)
    for name in sorted(committed_names - regenerated_names):
        lines.append(f"{name}: removed from the regenerated map")
    for name in sorted(regenerated_names - committed_names):
        lines.append(f"{name}: added in the regenerated map")

    for name in sorted(committed_names & regenerated_names):
        old = committed.targets[name]
        new = regenerated.targets[name]
        if old.status != new.status:
            lines.append(f"{name}.status: {old.status} -> {new.status}")
        if old.tolerance != new.tolerance:
            lines.append(f"{name}.tolerance: {old.tolerance!r} -> {new.tolerance!r}")
        if old.finding != new.finding:
            lines.append(f"{name}.finding: {old.finding} -> {new.finding}")
    return lines


def _as_hypothesis_profile(value: str) -> HypothesisProfile:
    if value == "ci":
        return "ci"
    if value == "full":
        return "full"
    if value == "explore":
        return "explore"
    raise SystemExit(
        f"QUANTCORE_CONFORMANCE_PROFILE must be 'ci', 'full', or 'explore', got {value!r}"
    )


def _load_cases() -> list[CaseResult]:
    if not RAW_RESULTS_PATH.exists():
        raise SystemExit(
            f"{RAW_RESULTS_PATH} not found; run `uv run pytest conformance -q` before "
            f"`{BUILD_COMMAND}`"
        )
    raw = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    return [CaseResult(**r) for r in raw]


def _load_details() -> list[ReproDetail]:
    if not DETAILS_PATH.exists():
        return []
    raw = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    return [ReproDetail(**r) for r in raw]


def _load_recorded_profile() -> HypothesisProfile:
    """The hypothesis profile this run's results were recorded under.

    Read from `meta.json` in the results directory (written by `conformance/conftest.py` at the
    end of the pytest session that produced `raw.json`), not solely from
    `QUANTCORE_CONFORMANCE_PROFILE` at the time `build_report` runs -- the two must agree, since
    the environment variable could in principle have changed between the test run and this
    invocation.

    Raises:
        SystemExit: `meta.json` is missing (run the conformance suite first), or its recorded
            profile disagrees with the currently-set `QUANTCORE_CONFORMANCE_PROFILE`.
    """
    env_profile = _as_hypothesis_profile(os.environ.get("QUANTCORE_CONFORMANCE_PROFILE", "ci"))
    if not META_PATH.exists():
        raise SystemExit(
            f"{META_PATH} not found; run `uv run pytest conformance -q` before `{BUILD_COMMAND}`"
        )
    recorded = json.loads(META_PATH.read_text(encoding="utf-8"))["hypothesis_profile"]
    if recorded != env_profile:
        raise SystemExit(
            f"{META_PATH} records hypothesis_profile={recorded!r} but "
            f"QUANTCORE_CONFORMANCE_PROFILE={env_profile!r} is set now; re-run the conformance "
            "suite with a consistent profile before building the report"
        )
    return env_profile


@dataclasses.dataclass(frozen=True)
class ExploreMismatch:
    """One recorded mismatch for a `verified` target's primary comparison, from an explore run.

    Attributes:
        function: Target name.
        quantity: Which quantity mismatched.
        family: Generator family the input came from, or `"-"` for a mismatch that belongs to
            the target's cases as a whole (a failed sign test) rather than to one input.
        length_bucket: Declared coverage bucket the input's length falls into, or `"-"`.
        seed: Seed reproducing the input, if generated.
        description: What went wrong: `"quantcore raised ..."`, the recorded abs/rel error, or
            the sign test's counts and outcome.
    """

    function: str
    quantity: str
    family: str
    length_bucket: str
    seed: int | None
    description: str


def explore_mismatches(
    cases: Sequence[CaseResult], details: Sequence[ReproDetail], verified_functions: set[str]
) -> list[ExploreMismatch]:
    """Every recorded primary-oracle mismatch for a target the committed map calls `verified`.

    Pure and I/O-free: `main()`'s explore-report path calls this on a real explore run's recorded
    cases; `tests/unit/conformance/test_build_report.py` calls it on synthetic ones (a `verified`
    target with a mismatch, a `defect` target with mismatches that must be ignored, and no
    mismatches at all) without running the slow, hypothesis-driven explore suite itself.

    A mismatch follows the rule that gave the target its status. For an ordinary target it is any
    case where quantcore raised, or `within_tolerance` is `False`. Only `oracle == ""` (the
    primary quantcore-vs-reference comparison) is considered: a second-oracle cross-check
    disagreeing (e.g. `quantstats`) is never, by itself, a mismatch this function reports, since
    it never decided any status in the first place. A case flagged `ill_conditioned` is likewise
    never a mismatch: it is excluded from the decision, so it must be excluded from this report
    too, not just happen to pass because its effective tolerance was widened.

    An iterative estimator (`ITERATIVE_ESTIMATORS`) is `verified` by a sign test over all its
    well-specified cases, under which a single case outside tolerance is expected (it may be a
    win, or belong to a misspecified family). Its individual cases are therefore never mismatches;
    instead the sign test is run again on this run's cases, and one mismatch is reported if its
    outcome is anything other than `verified`.

    Args:
        cases: Every recorded `CaseResult` from an explore run.
        details: Every recorded `ReproDetail` from the same run.
        verified_functions: Names of every target the *committed* conformance map (the one this
            explore run is checking against, not one it would generate) classifies `verified`.

    Returns:
        One `ExploreMismatch` per mismatching case or failed sign test, in a deterministic order
        (sorted by target, then quantity, then family, then seed).
    """
    keys = sorted(
        {
            (c.function, c.quantity)
            for c in cases
            if c.oracle == ""
            and c.function in verified_functions
            and c.function not in ITERATIVE_ESTIMATORS
        }
    )
    mismatches: list[ExploreMismatch] = []
    for function, quantity in keys:
        pairs = matched_pairs(cases, details, function=function, quantity=quantity, oracle="")
        for case, detail in pairs:
            if case.ill_conditioned:
                continue
            if case.quantcore_raised is None and case.within_tolerance:
                continue
            description = (
                f"quantcore raised {case.quantcore_raised}"
                if case.quantcore_raised is not None
                else f"abs_err={case.abs_err!r} rel_err={case.rel_err!r}"
            )
            mismatches.append(
                ExploreMismatch(
                    function=function,
                    quantity=quantity,
                    family=case.family,
                    length_bucket=case.length_bucket,
                    seed=detail.seed,
                    description=description,
                )
            )

    recorded = {c.function for c in cases if c.oracle == ""}
    for function in sorted(verified_functions & recorded & set(ITERATIVE_ESTIMATORS)):
        estimator = ITERATIVE_ESTIMATORS[function]
        decision = iterative_estimator_decision(function, cases, details, estimator)
        if decision.status == "verified":
            continue
        mismatches.append(
            ExploreMismatch(
                function=function,
                quantity=estimator.quantity,
                family="-",
                length_bucket="-",
                seed=None,
                description=(
                    f"sign test on the attained objective gives {decision.status!r}: "
                    f"wins={decision.wins} ties={decision.ties} losses={decision.losses} "
                    f"p_value={decision.p_value:.4g} median_gap={decision.median_gap:.4g}"
                ),
            )
        )
    return sorted(mismatches, key=lambda m: (m.function, m.quantity, m.family, m.seed or -1))


def _run_explore_report(cases: Sequence[CaseResult], details: Sequence[ReproDetail]) -> int:
    """Print every recorded mismatch for a `verified` target under the `explore` profile.

    Never writes any file. Reads the *committed* `conformance_map.json` to learn which targets
    are `verified` -- an explore run never has, and must never produce, a map of its own.

    Returns:
        `1` if at least one mismatch was recorded for a `verified` target, else `0`; `1` (with a
        message) if the committed map does not exist to check against.
    """
    if not CONFORMANCE_MAP_PATH.exists():
        print(
            f"{CONFORMANCE_MAP_PATH} does not exist; nothing to explore against",
            file=sys.stderr,
        )
        return 1
    committed_map = conformance_map_from_json(CONFORMANCE_MAP_PATH.read_text(encoding="utf-8"))
    mismatches = explore_mismatches(cases, details, verified_functions_from_map(committed_map))
    if not mismatches:
        print("explore profile: no mismatches recorded for any verified target.")
        return 0
    print(
        f"explore profile: {len(mismatches)} mismatch(es) recorded for verified target(s):",
        file=sys.stderr,
    )
    for m in mismatches:
        print(
            f"  {m.function}.{m.quantity} family={m.family!r} length_bucket="
            f"{m.length_bucket!r} seed={m.seed!r}: {m.description}",
            file=sys.stderr,
        )
    return 1


def verified_functions_from_map(conformance_map: ConformanceMap) -> set[str]:
    """Names of every target ``conformance_map`` classifies `verified`.

    Pure and I/O-free; `main()`'s explore-report path calls this on the committed
    `conformance_map.json`.
    """
    return {name for name, entry in conformance_map.targets.items() if entry.status == "verified"}


def oracle_extra_distributions(requirements: Iterable[str]) -> list[str]:
    """Pick the distribution names of the `oracle` extra out of a requirement list.

    Pure and I/O-free.

    Args:
        requirements: Requirement strings as `importlib.metadata.requires` returns them, e.g.
            `'statsmodels==0.15.0; extra == "oracle"'`.

    Returns:
        The name of every requirement whose marker selects the `oracle` extra, in input order.
    """
    names: list[str] = []
    for requirement in requirements:
        if not _ORACLE_EXTRA_RE.search(requirement):
            continue
        match = _REQUIREMENT_NAME_RE.match(requirement.strip())
        if match is not None:
            names.append(match.group())
    return names


def oracle_versions() -> dict[str, str]:
    """Installed version of every distribution an oracle is built on.

    That is every distribution quantcore's `oracle` extra pins (the oracle libraries, and
    hypothesis, whose version fixes which cases a derandomized run draws), plus numpy and scipy,
    on which the closed-form and hand-recursion oracles are computed directly.

    Returns:
        Mapping from distribution name to installed version, sorted by name. A distribution of
        the extra that is not installed is absent.

    Raises:
        RuntimeError: quantcore's installed metadata lists no `oracle` extra at all, so the
            oracle libraries cannot be enumerated.
    """
    extra = oracle_extra_distributions(importlib.metadata.requires("quantcore") or [])
    if not extra:
        raise RuntimeError(
            "the installed quantcore distribution declares no 'oracle' extra; reinstall it with "
            "`uv sync --extra dev --extra oracle`"
        )
    versions: dict[str, str] = {}
    for distribution_name in sorted({*extra, *_NUMERICAL_BASE_DISTRIBUTIONS}):
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _worst_detail(details: Sequence[ReproDetail]) -> ReproDetail | None:
    """Pick the case with the largest well-defined relative error, for a report or a finding.

    Excludes cases whose oracle magnitude is at or below `_ZERO_FLOOR`, matching
    `aggregate_errors`'s exclusion, so the worst case shown in a finding is always one that also
    contributed to the summary table's `max_rel_error` (never a near-zero-denominator outlier).
    """
    candidates = [d for d in details if d.rel_err is not None and d.expected_abs > _ZERO_FLOOR]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.rel_err if d.rel_err is not None else -1.0)


def round_significant(value: float, digits: int = _ERROR_SIG_DIGITS) -> float:
    """Round ``value`` to ``digits`` significant decimal digits.

    Zero and non-finite values pass through unchanged (there is no meaningful "significant digit"
    of 0.0, NaN, or infinity to round to).

    Args:
        value: The magnitude to round. Expected to be non-negative (an error magnitude), but the
            rounding itself works for any finite non-zero float.
        digits: How many significant digits to keep.

    Returns:
        ``value`` rounded to ``digits`` significant digits.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(abs(value)))
    decimals = digits - int(exponent) - 1
    return round(value, decimals)


def aggregate_errors(details: Sequence[ReproDetail]) -> tuple[float | None, float | None]:
    """Reduce a list of per-case error records to one (max abs, max rel) pair for a report.

    `max_abs_error` is the largest `abs_err` among cases where quantcore produced a value to
    compare at all (``abs_err`` is `None` when quantcore raised). `max_rel_error` is the largest
    `rel_err` among that same set, further restricted to cases whose oracle magnitude
    (`expected_abs`) exceeds `_ZERO_FLOOR`: near a true zero, relative error is dividing by
    (approximately) nothing and is not informative, so those cases are excluded rather than
    letting one dominate the aggregate. Both results are rounded to 3 significant digits.

    Args:
        details: Per-case error records, ordinarily every `ReproDetail` recorded for one target
            (or one quantity of one target).

    Returns:
        ``(max_abs_error, max_rel_error)``; either is `None` if no case qualified.
    """
    abs_candidates = [d.abs_err for d in details if d.abs_err is not None]
    rel_candidates = [
        d.rel_err for d in details if d.rel_err is not None and d.expected_abs > _ZERO_FLOOR
    ]
    max_abs = round_significant(max(abs_candidates)) if abs_candidates else None
    max_rel = round_significant(max(rel_candidates)) if rel_candidates else None
    return max_abs, max_rel


REPORT_ONLY_REASON = (
    "comparator is report-only (conformance.compare.ReportOnlyScalarComparator): agreement with "
    "the oracle is recorded for the report but never asserted, so this target is never "
    "classified verified whatever the recorded error magnitudes"
)

NO_ORACLE_REASON = "no independent oracle is registered for this target"


def is_report_only(target: ConformanceTarget[..., Any]) -> bool:
    """Whether a target's whole comparator records agreement without ever judging it.

    Such a comparator (`conformance.compare.ReportOnlyScalarComparator`) marks itself with
    `report_only_top_level`; every case it produces reads `within_tolerance=True` however far
    apart quantcore and the oracle are.
    """
    return bool(getattr(target.compare, "report_only_top_level", False))


def _reason(
    status: ConformanceStatus,
    target: ConformanceTarget[..., Any],
    quantcore_ever_raised: bool,
    finding_title: str | None,
) -> str:
    """The map's `reason` field for one target: empty for `verified`, one sentence otherwise."""
    if status == "verified":
        return ""
    if status == "unverifiable":
        return NO_ORACLE_REASON if target.reference is None else REPORT_ONLY_REASON
    if finding_title is not None:
        # The reason for a `defect` or `non-standard` status is its finding's own one-line title.
        return finding_title
    if quantcore_ever_raised:
        return "quantcore raised on a valid generated input"
    return "reason not determined"


def _classify_target(
    name: str,
    target: ConformanceTarget[..., Any],
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    decision_cases: Sequence[CaseResult],
) -> ConformanceStatus:
    """Decide one target's status, in priority order.

    1. No oracle: `unverifiable`, there is nothing to compare quantcore against.
    2. A report-only comparator as the target's *entire* `compare` (e.g. `select_arima_order`):
       `unverifiable`. Its `assert_close` never raises, so every recorded case reads
       `within_tolerance=True` regardless of actual agreement, and `classify()` would call it
       `verified`; such a target is never run through `classify()`.
    3. An iterative estimator without a unique optimum (`ITERATIVE_ESTIMATORS`): the sign test on
       the attained objective, never a comparison of raw parameters.
    4. Anything else: `classify()` on the cases that are not ill-conditioned.

    Raises:
        ValueError: The target has an oracle but the run recorded no case for it.
    """
    if target.reference is None or is_report_only(target):
        return "unverifiable"
    estimator = ITERATIVE_ESTIMATORS.get(name)
    if estimator is not None:
        return iterative_estimator_decision(name, cases, details, estimator).status
    if not decision_cases:
        raise ValueError(
            f"{name!r} has an oracle but no recorded cases to classify; run its conformance "
            "test before building the report"
        )
    return classify(decision_cases, has_reference=True)


def build_conformance_map(
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    hypothesis_profile: MapWritableProfile,
    targets: Mapping[str, ConformanceTarget[..., Any]] | None = None,
) -> ConformanceMap:
    """Classify every registered target and assemble the full conformance map.

    Never called under the `explore` profile: `explore` draws fresh, non-reproducible inputs on
    every run, so it must never produce a map anyone could commit (see `main`'s explore branch).

    Args:
        cases: Every `CaseResult` recorded by the conformance suite's most recent run.
        details: Every `ReproDetail` recorded alongside ``cases``, used to compute
            `max_abs_error`/`max_rel_error`, to read an iterative estimator's attained objective
            and to pick the worst case for a findings write-up.
        hypothesis_profile: Which hypothesis profile (`"ci"` or `"full"`) produced ``cases``.
        targets: The targets to classify; `None` means every registered one (`registry()`).

    Returns:
        The conformance map ready to write to `conformance_map.json`.

    Raises:
        ValueError: A target has no quantcore callable (there is nothing to give a status to),
            or has an oracle but no recorded case.
    """
    if targets is None:
        targets = registry()
    without_quantcore = sorted(name for name, target in targets.items() if target.quantcore is None)
    if without_quantcore:
        raise ValueError(
            "registered without a quantcore callable, so there is nothing to judge: "
            + ", ".join(without_quantcore)
        )

    records: dict[str, TargetRecord] = {}
    for name, target in sorted(targets.items()):
        # Only the primary comparison (oracle == "") may influence a status; a second-oracle
        # cross-check (e.g. quantstats vs. sharpe_ratio's reference) is reported separately in
        # the "Oracle cross-checks" section and must never reach classify() here.
        function_cases = [c for c in cases if c.function == name and c.oracle == ""]
        function_details = [d for d in details if d.function == name and d.oracle == ""]
        # A case whose effective tolerance exceeded the ill-conditioning cap is excluded from
        # the decision here, but still counted and reported (see `ill_conditioned_counts`).
        decision_cases = [c for c in function_cases if not c.ill_conditioned]

        status = _classify_target(name, target, cases, details, decision_cases)

        max_abs_error, max_rel_error = aggregate_errors(function_details)
        quantcore_ever_raised = any(c.quantcore_raised for c in function_cases)

        by_quantity: dict[str, QuantityErrors] = {}
        for quantity in sorted({d.quantity for d in function_details}):
            quantity_details = [d for d in function_details if d.quantity == quantity]
            quantity_abs, quantity_rel = aggregate_errors(quantity_details)
            by_quantity[quantity] = QuantityErrors(
                cases=len(quantity_details),
                max_abs_error=quantity_abs,
                max_rel_error=quantity_rel,
            )

        tolerance = TOLERANCES.get(name)
        tolerance_record = (
            None
            if tolerance is None
            else ToleranceRecord(
                rtol=tolerance.rtol, atol=tolerance.atol, justification=tolerance.justification
            )
        )

        # One underlying generated input yields one CaseResult/ReproDetail per quantity, so the
        # target-level case count is any one quantity's count (they are all equal by
        # construction), not the sum across quantities.
        representative_cases = next(iter(by_quantity.values())).cases if by_quantity else 0

        needs_finding = status in ("defect", "non-standard")
        finding_title = _finding_title(name) if needs_finding else None

        records[name] = TargetRecord(
            status=status,
            oracle=target.oracle,
            cases=representative_cases,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            by_quantity=by_quantity,
            tolerance=tolerance_record,
            reason=_reason(status, target, quantcore_ever_raised, finding_title),
            finding=None,  # filled in by _assign_findings once every status is known
        )

    return ConformanceMap(
        schema_version=SCHEMA_VERSION,
        quantcore_version=importlib.metadata.version("quantcore"),
        oracle_versions=oracle_versions(),
        hypothesis_profile=hypothesis_profile,
        targets=_assign_findings(records),
    )


def _finding_source_path(name: str) -> Path:
    return FINDINGS_SRC_DIR / f"{name}.md"


def _finding_title(name: str) -> str | None:
    """Best-effort title of a target's finding source, or `None` if the source is missing.

    Used only to populate `reason` (which must equal the finding's title); the authoritative,
    build-failing check that every `defect`/`non-standard` target has a source file lives in
    `missing_finding_sources`.
    """
    path = _finding_source_path(name)
    if not path.exists():
        return None
    try:
        return load_finding_source(path).title
    except ValueError:
        return None


def missing_finding_sources(targets: Mapping[str, TargetRecord]) -> list[str]:
    """List every `defect`/`non-standard` target with no `findings_src/<target>.md`.

    Pure aside from a plain path existence check (no parsing).

    Args:
        targets: The conformance map's per-target entries.

    Returns:
        One line per missing source path; empty when every `defect`/`non-standard` target has
        a `findings_src/<target>.md`.
    """
    missing: list[str] = []
    for name, entry in sorted(targets.items()):
        if entry.status in ("defect", "non-standard") and not _finding_source_path(name).exists():
            missing.append(f"{name}: missing {_finding_source_path(name)}")
    return missing


def unused_finding_sources(targets: Mapping[str, TargetRecord]) -> list[str]:
    """List every `findings_src/<target>.md` whose target has no finding to write.

    A source is unused when its target is in the map with a status other than `defect` or
    `non-standard`. It does no harm to a build, but it describes a difference the run did not
    find, so it should be deleted. A source whose name is not in the map at all is not listed:
    its target may have been left out of an incomplete build.

    Args:
        targets: The conformance map's per-target entries.

    Returns:
        One line per unused source, naming the file and its target's status, sorted by name.
    """
    unused: list[str] = []
    for path in sorted(FINDINGS_SRC_DIR.glob("*.md")):
        entry = targets.get(path.stem)
        if entry is not None and entry.status not in ("defect", "non-standard"):
            unused.append(
                f"{path.stem}: {path} is not used, the target is `{entry.status}`; delete it"
            )
    return unused


def finding_path(index: int, name: str) -> str:
    """Repository-relative path of the generated finding numbered ``index`` for target ``name``."""
    return f"{FINDINGS_RELATIVE_DIR}/{index:03d}-{name}.md"


def _assign_findings(targets: dict[str, TargetRecord]) -> dict[str, TargetRecord]:
    needing_findings = sorted(
        name for name, entry in targets.items() if entry.status in ("defect", "non-standard")
    )
    result = dict(targets)
    for index, name in enumerate(needing_findings, start=1):
        result[name] = dataclasses.replace(result[name], finding=finding_path(index, name))
    return result


# ---------------------------------------------------------------------------
# Minimal, executed repros
# ---------------------------------------------------------------------------

# Total count of numbers (scalars, or elements across every array argument) at or below which a
# repro inlines its input literally; above this it regenerates the input from its recorded seed
# instead, so a finding's repro stays standalone and small.
REPRO_INLINE_MAX_NUMBERS = 12

REPRO_TIMEOUT_SECONDS = 30.0


@dataclasses.dataclass(frozen=True)
class ReproCallConfig:
    """How to build a standalone repro snippet for one target.

    The two call expressions restate, with public imports only, what the target's registered
    `quantcore` and `reference` callables compute, so the snippet runs without this suite.

    Attributes:
        module_import: Import line for the quantcore module the repro calls.
        quantcore_call: Expression calling quantcore, given the bound argument names.
        oracle_import: Import line(s) for the oracle library the repro calls.
        oracle_call: Expression calling the oracle, given the same bound argument names.
        arg_names: Names the reconstructed input tuple's elements are bound to, in call order.
        quantity_index: Per quantity, `(quantcore_index, oracle_index)` into each call's return
            value; `None` for either side means that call already returns the quantity directly.
            Ignored for a quantity that instead declares a `summary_expr`.
        summary_expr: Per quantity, a Python expression computing a single scalar SUMMARY from
            `quantcore_result`/`oracle_result` (e.g. `"float(np.mean(quantcore_result ==
            oracle_result))"` for a decoded-state agreement rate), for a quantity whose raw
            output is not itself a scalar. When a quantity is a key here,
            `render_minimal_repro` prints only this one number (labelled `summary`) instead of
            `expected`/`actual`/`abs diff`, and `verify_repro` checks it with `summary_ok` rather
            than the target's ordinary rtol/atol tolerance. A quantity absent from this dict
            uses the ordinary `quantity_index`-based scalar comparison.
        summary_ok: Per quantity with a `summary_expr`, whether a printed summary value means the
            case does not reproduce the finding (i.e. is "close enough" that the finding would
            be stale). Required for every key of `summary_expr`.
    """

    module_import: str
    quantcore_call: str
    oracle_import: str
    oracle_call: str
    arg_names: tuple[str, ...]
    quantity_index: dict[str, tuple[int | None, int | None]]
    summary_expr: dict[str, str] = dataclasses.field(default_factory=dict)
    summary_ok: dict[str, Callable[[float], bool]] = dataclasses.field(default_factory=dict)


REPRO_CONFIGS: dict[str, ReproCallConfig] = {
    "engle_granger_test": ReproCallConfig(
        module_import="import quantcore.statistics as qc",
        quantcore_call="qc.engle_granger_test(y, x)",
        oracle_import="from statsmodels.tsa.stattools import coint",
        oracle_call='coint(y, x, trend="c", maxlag=1, autolag=None)',
        arg_names=("y", "x"),
        # quantcore.engle_granger_test returns (beta, statistic, p_value); statsmodels.coint
        # returns (statistic, p_value, critical_values).
        quantity_index={"statistic": (1, 0), "p_value": (2, 1)},
    ),
    "maximum_drawdown": ReproCallConfig(
        module_import="import quantcore.performance as qc",
        quantcore_call="qc.maximum_drawdown(returns)",
        oracle_import="import empyrical",
        oracle_call="-empyrical.max_drawdown(returns)",
        arg_names=("returns",),
        quantity_index={"value": (None, None)},
    ),
    "calmar_ratio": ReproCallConfig(
        module_import="import quantcore.performance as qc",
        quantcore_call='qc.calmar_ratio(returns, periods_per_year, return_method="cagr")',
        oracle_import="import empyrical",
        oracle_call="empyrical.calmar_ratio(returns, annualization=periods_per_year)",
        arg_names=("returns", "periods_per_year"),
        quantity_index={"value": (None, None)},
    ),
    "calmar_ratio_arithmetic": ReproCallConfig(
        module_import="import quantcore.performance as qc",
        quantcore_call='qc.calmar_ratio(returns, periods_per_year, return_method="arithmetic")',
        oracle_import="import numpy as _np, empyrical",
        oracle_call=(
            "float(_np.mean(returns)) * periods_per_year / abs(-empyrical.max_drawdown(returns))"
        ),
        arg_names=("returns", "periods_per_year"),
        quantity_index={"value": (None, None)},
    ),
    "adf_test": ReproCallConfig(
        module_import="import quantcore.statistics as qc",
        quantcore_call='qc.adf_test(series, max_lags, trend="c", autolag=None)',
        oracle_import="from statsmodels.tsa.stattools import adfuller",
        oracle_call='adfuller(series, maxlag=max_lags, regression="c", autolag=None)',
        arg_names=("series", "max_lags"),
        quantity_index={"statistic": (0, 0), "p_value": (1, 1)},
    ),
    # The three iterative-estimator fitters below repro only `log_likelihood`, the decision
    # quantity: `params` never fails its (no-op) comparator, so `select_shortest_failing_detail`
    # never has a failing case to build a repro from for it. The oracle expression evaluates
    # `arch`'s fitted parameters as they are, without the clip into the stationary region the
    # registered oracle applies first; where `arch`'s fit lands outside that region the shared
    # likelihood cannot be evaluated, the snippet exits non-zero, and `verify_repro` reports it
    # instead of writing a finding.
    "fit_garch_11": ReproCallConfig(
        module_import=(
            "import quantcore.risk.volatility as qc\n\n\n"
            "def _ll(eps, omega, alpha, beta):\n"
            "    v = qc.garch_11_variance(eps, omega, alpha, beta)\n"
            "    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(v) + eps**2 / v))"
        ),
        quantcore_call="_ll(returns - returns.mean(), *qc.fit_garch_11(returns))",
        oracle_import="from arch.univariate import arch_model",
        oracle_call=(
            "_ll(returns - returns.mean(), *[float(arch_model(returns - returns.mean(), "
            'mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off").params[k]) '
            'for k in ["omega", "alpha[1]", "beta[1]"]])'
        ),
        arg_names=("returns",),
        quantity_index={"log_likelihood": (None, None)},
    ),
    "egarch_fit": ReproCallConfig(
        module_import=(
            "import quantcore.risk.egarch as qc\n\n\n"
            "def _ll(eps, omega, alpha, gamma, beta):\n"
            "    v = np.exp(qc.egarch_11_variance(eps, omega, alpha, gamma, beta))\n"
            "    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(v) + eps**2 / v))"
        ),
        quantcore_call=(
            "_ll(returns - returns.mean(), *(lambda f: (f.omega, f.alpha, f.gamma, f.beta))"
            "(qc.egarch_fit(returns)))"
        ),
        oracle_import="from arch.univariate import arch_model",
        oracle_call=(
            "_ll(returns - returns.mean(), *[float(arch_model(returns - returns.mean(), "
            'mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off").params[k]) '
            'for k in ["omega", "alpha[1]", "gamma[1]", "beta[1]"]])'
        ),
        arg_names=("returns",),
        quantity_index={"log_likelihood": (None, None)},
    ),
    "gjr_garch_fit": ReproCallConfig(
        module_import=(
            "import quantcore.risk.egarch as qc\n\n\n"
            "def _ll(eps, omega, alpha, gamma, beta):\n"
            "    v = qc.gjr_garch_11_variance(eps, omega, alpha, gamma, beta)\n"
            "    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(v) + eps**2 / v))"
        ),
        quantcore_call=(
            "_ll(returns - returns.mean(), *(lambda f: (f.omega, f.alpha, f.gamma, f.beta))"
            "(qc.gjr_garch_fit(returns)))"
        ),
        oracle_import="from arch.univariate import arch_model",
        oracle_call=(
            "_ll(returns - returns.mean(), *[float(arch_model(returns - returns.mean(), "
            'mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal").fit(disp="off").params[k]) '
            'for k in ["omega", "alpha[1]", "gamma[1]", "beta[1]"]])'
        ),
        arg_names=("returns",),
        quantity_index={"log_likelihood": (None, None)},
    ),
    "dynamic_hedge_ratio": ReproCallConfig(
        module_import="import quantcore.filtering.kalman as qc",
        quantcore_call="qc.dynamic_hedge_ratio(y, x, obs_var, proc_var)[-1]",
        oracle_import=(
            "def _hand_hedge(y, x, obs_var, proc_var):\n"
            "    beta = np.empty(len(y))\n"
            "    beta_prev, p_prev = 0.0, 1.0\n"
            "    for t in range(len(y)):\n"
            "        p_pred = p_prev + proc_var\n"
            "        gain = p_pred * x[t] / (x[t] * p_pred * x[t] + obs_var)\n"
            "        beta_prev = beta_prev + gain * (y[t] - x[t] * beta_prev)\n"
            "        p_prev = (1.0 - gain * x[t]) * p_pred\n"
            "        beta[t] = beta_prev\n"
            "    return beta"
        ),
        oracle_call="_hand_hedge(y, x, obs_var, proc_var)[-1]",
        arg_names=("y", "x", "obs_var", "proc_var"),
        quantity_index={"value": (None, None)},
    ),
}


def reconstruct_args(inputs_repr: str) -> tuple[Any, ...]:
    """Reconstruct the positional-argument tuple a `ReproDetail.inputs_repr` describes.

    `inputs_repr` is `repr()` of a Python tuple whose elements are numpy arrays or plain
    scalars — numpy's own array `repr()` (e.g. `array([1.0, 2.0])`) has no module prefix, so this
    evaluates the text back into real objects using numpy's `array` constructor for that name,
    with no other names in scope.

    Args:
        inputs_repr: A `ReproDetail.inputs_repr` string, as recorded by `conformance.harness`.

    Returns:
        The reconstructed tuple of positional arguments.
    """
    namespace = {"array": np.array, "nan": float("nan"), "inf": float("inf")}
    return cast(tuple[Any, ...], eval(inputs_repr, {"__builtins__": {}}, namespace))


def _total_numbers(args: tuple[Any, ...]) -> int:
    """Count of individual numbers across every argument (array elements, or 1 per scalar)."""
    total = 0
    for arg in args:
        total += int(arg.size) if isinstance(arg, np.ndarray) else 1
    return total


_TOP_LEVEL_SPLIT_OPEN = "(["
_TOP_LEVEL_SPLIT_CLOSE = ")]"


def _split_top_level(tuple_repr: str) -> list[str]:
    """Split `repr()` of a Python tuple into its top-level element reprs.

    Splits only on commas at nesting depth 0 (outside any `(...)`/`[...]`), so an element that is
    itself `array([1.0, 2.0])` is not split on the comma inside its brackets.
    """
    inner = tuple_repr.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in inner:
        if char in _TOP_LEVEL_SPLIT_OPEN:
            depth += 1
        elif char in _TOP_LEVEL_SPLIT_CLOSE:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


_SHAPE_RE = re.compile(r"shape=\(([^)]*)\)")


def _numbers_in_element_repr(element_repr: str) -> int:
    """Count of numbers one tuple-element repr (an `array(...)` or a bare scalar) describes.

    Truncation-safe: numpy abbreviates a long array's `repr()` with `...` and an explicit
    `shape=(...)` (e.g. `array([1., 2., ..., 9.], shape=(2265,))`), so a truncated array is sized
    from that shape tuple's product rather than by counting the (necessarily incomplete) printed
    elements.
    """
    shape_match = _SHAPE_RE.search(element_repr)
    if shape_match:
        dims = [int(d) for d in shape_match.group(1).split(",") if d.strip()]
        size = 1
        for dim in dims:
            size *= dim
        return size
    if element_repr.startswith("array("):
        bracket_start = element_repr.index("[")
        bracket_end = element_repr.rindex("]")
        elements_text = element_repr[bracket_start + 1 : bracket_end].strip()
        return 0 if not elements_text else elements_text.count(",") + 1
    return 1


def count_numbers_in_repr(inputs_repr: str) -> int:
    """Count the total numbers a `ReproDetail.inputs_repr` describes, without evaluating it.

    Unlike `_total_numbers(reconstruct_args(...))`, this never fails on a truncated numpy repr
    (`array([1., 2., ..., 9.], shape=(2265,))` for a long array), so it is safe to call on every
    recorded case, including ones far too long to ever be inlined.

    Args:
        inputs_repr: A `ReproDetail.inputs_repr` string.

    Returns:
        Total count of numbers across every positional argument the repr describes.
    """
    return sum(_numbers_in_element_repr(part) for part in _split_top_level(inputs_repr))


_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
_DTYPE_NOISE_RE = re.compile(r"dtype=\w+")


def _approx_sum_abs_in_repr(inputs_repr: str) -> float:
    """Sum of absolute values of the numbers visible in ``inputs_repr``.

    A tie-break, not an exact measure: for a truncated array, only the printed edge elements
    contribute (the rest were never printed), and `shape=(...)`/`dtype=...` annotations are
    stripped first so their digits are never mistaken for data.
    """
    cleaned = _DTYPE_NOISE_RE.sub("", _SHAPE_RE.sub("", inputs_repr))
    return sum(abs(float(match.group())) for match in _NUMBER_RE.finditer(cleaned))


def matched_pairs(
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    *,
    function: str,
    quantity: str,
    oracle: str = "",
) -> list[tuple[CaseResult, ReproDetail]]:
    """Pair each recorded case with its detail for one target's quantity.

    `harness.check()` (and `harness.check_second_oracle()`) append one `CaseResult` and one
    `ReproDetail` together, in lockstep, for every recorded quantity of every call, so filtering
    both lists to the same `(function, oracle, quantity)` and zipping preserves the one-to-one
    correspondence between a case and its detail.

    Args:
        cases: Every recorded `CaseResult` from one conformance run.
        details: Every recorded `ReproDetail` from the same run.
        function: Target name to filter both lists to.
        quantity: Quantity name to filter both lists to.
        oracle: Which oracle's cases to filter to; `""` (the default) selects the primary
            quantcore-vs-reference comparison, any other value selects a second-oracle
            cross-check (see `CaseResult.oracle`).

    Returns:
        `(case, detail)` pairs, in recording order.

    Raises:
        ValueError: The filtered lists have different lengths (a broken recording invariant).
    """
    matched_cases = [
        c for c in cases if c.function == function and c.quantity == quantity and c.oracle == oracle
    ]
    matched_details = [
        d
        for d in details
        if d.function == function and d.quantity == quantity and d.oracle == oracle
    ]
    return list(zip(matched_cases, matched_details, strict=True))


@dataclasses.dataclass(frozen=True)
class OracleCrossCheckRow:
    """One `(function, oracle, quantity)` row of the "Oracle cross-checks" report section.

    Attributes:
        function: Target name.
        oracle: Name of the second oracle cross-checked (e.g. `"quantstats"`).
        quantity: Which comparison this row reports (e.g. `"rf_zero"`, `"rf_nonzero"`).
        cases: Number of recorded cross-check cases.
        max_abs_diff: Largest absolute difference between the second oracle's value and the
            target's reference (standard-definition) value, rounded to 3 significant digits;
            `None` if no case produced a finite oracle value.
        max_rel_diff: Largest relative difference among cases whose reference magnitude exceeds
            the near-zero floor; `None` if no such case was recorded.
        agrees: Whether every recorded case was within the target's written tolerance. A case
            where the oracle raised or returned a non-finite value never counts as agreeing.
        raised_or_nonfinite: Count of cases where the oracle raised or returned a non-finite
            value, reported separately so that failure mode stays visible instead of collapsing
            into "disagrees".
    """

    function: str
    oracle: str
    quantity: str
    cases: int
    max_abs_diff: float | None
    max_rel_diff: float | None
    agrees: bool
    raised_or_nonfinite: int


def oracle_cross_check_rows(
    cases: Sequence[CaseResult], details: Sequence[ReproDetail]
) -> list[OracleCrossCheckRow]:
    """Aggregate every recorded second-oracle case into one row per `(function, oracle, quantity)`.

    Pure and I/O-free. Cases with `oracle == ""` (the primary quantcore-vs-reference comparison
    that decides the status) are ignored entirely: this reports second-oracle cross-checks only,
    and never feeds `classify()`.

    Args:
        cases: Every recorded `CaseResult` from one conformance run.
        details: Every recorded `ReproDetail` from the same run.

    Returns:
        One `OracleCrossCheckRow` per distinct `(function, oracle, quantity)` with `oracle != ""`,
        sorted by that key.
    """
    keys = sorted({(c.function, c.oracle, c.quantity) for c in cases if c.oracle})
    rows: list[OracleCrossCheckRow] = []
    for function, oracle, quantity in keys:
        pairs = matched_pairs(cases, details, function=function, quantity=quantity, oracle=oracle)
        matched_cases = [c for c, _d in pairs]
        max_abs, max_rel = aggregate_errors([d for _c, d in pairs])
        rows.append(
            OracleCrossCheckRow(
                function=function,
                oracle=oracle,
                quantity=quantity,
                cases=len(pairs),
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                agrees=all(c.within_tolerance for c in matched_cases),
                raised_or_nonfinite=sum(1 for c in matched_cases if c.quantcore_raised),
            )
        )
    return rows


@dataclasses.dataclass(frozen=True)
class OracleDisagreement:
    """One concrete, small example of a second oracle disagreeing with the reference.

    Attributes:
        function: Target name.
        oracle: Name of the second oracle.
        quantity: Which comparison this example is drawn from.
        seed: Seed reproducing the input, if generated.
        family: Generator family the input came from.
        oracle_value: `repr()` of the second oracle's own value, or `None` if it raised or
            returned a non-finite value.
        reference_value: `repr()` of the target's reference (standard-definition) value.
        abs_diff: Absolute difference between the two, or `None` if ``oracle_value`` is `None`.
        raised: The oracle's raised/non-finite description, or `None` if it returned a value.
    """

    function: str
    oracle: str
    quantity: str
    seed: int | None
    family: str
    oracle_value: str | None
    reference_value: str
    abs_diff: float | None
    raised: str | None


def oracle_disagreement_examples(
    cases: Sequence[CaseResult], details: Sequence[ReproDetail]
) -> list[OracleDisagreement]:
    """Pick one concrete, small disagreeing example per `(function, oracle, quantity)`.

    Pure and I/O-free. A row every one of whose recorded cases agreed contributes no example.

    Args:
        cases: Every recorded `CaseResult` from one conformance run.
        details: Every recorded `ReproDetail` from the same run.

    Returns:
        One `OracleDisagreement` per `(function, oracle, quantity)` with at least one recorded
        case outside the target's tolerance (including a raised/non-finite oracle call) -- each
        the shortest such recorded input (see `select_shortest_failing_detail`) -- sorted by key.
    """
    keys = sorted({(c.function, c.oracle, c.quantity) for c in cases if c.oracle})
    examples: list[OracleDisagreement] = []
    for function, oracle, quantity in keys:
        pairs = matched_pairs(cases, details, function=function, quantity=quantity, oracle=oracle)
        detail = select_shortest_failing_detail(pairs)
        if detail is None:
            continue
        case = next(c for c, d in pairs if d is detail)
        examples.append(
            OracleDisagreement(
                function=function,
                oracle=oracle,
                quantity=quantity,
                seed=detail.seed,
                family=detail.family,
                oracle_value=detail.quantcore_value,
                reference_value=detail.reference_value,
                abs_diff=detail.abs_err,
                raised=case.quantcore_raised,
            )
        )
    return examples


def select_shortest_failing_detail(
    pairs: Sequence[tuple[CaseResult, ReproDetail]],
) -> ReproDetail | None:
    """Pick the shortest failing input recorded for one quantity, for a finding's minimal repro.

    "Shortest" is the total count of numbers across every positional argument; ties break
    toward the smallest sum of absolute values, so a finding's repro stays standalone and small,
    using the shortest failing input recorded for that quantity.

    Args:
        pairs: `(case, detail)` pairs for one target's quantity, as `matched_pairs` returns.

    Returns:
        The chosen `ReproDetail`, or `None` if no case in ``pairs`` failed its tolerance.
    """
    failing = [detail for case, detail in pairs if not case.within_tolerance]
    if not failing:
        return None

    def sort_key(detail: ReproDetail) -> tuple[int, float]:
        return (
            count_numbers_in_repr(detail.inputs_repr),
            _approx_sum_abs_in_repr(detail.inputs_repr),
        )

    return min(failing, key=sort_key)


def headline_case_tolerance_violation(
    case: CaseResult, detail: ReproDetail, tolerance: Tolerance
) -> str | None:
    """Independently re-check that a finding's headline case truly falls outside its tolerance.

    `build_report` refuses to emit a finding whose headline case is within the target's
    declared tolerance: every finding's repro numbers must come from a recorded
    `within_tolerance == False` case, and the report step asserts this before writing. This
    recomputes `tolerance.is_close` from
    ``detail``'s own recorded ``quantcore_value``/``reference_value`` rather than trusting
    ``case.within_tolerance`` outright, so a flag set under any tolerance other than the declared
    one is caught here rather than silently producing a finding whose repro numbers are inside
    tolerance.

    Args:
        case: The `CaseResult` matching ``detail`` (same function/quantity/family/length_bucket).
        detail: The chosen headline `ReproDetail` (ordinarily `select_shortest_failing_detail`'s
            result) a finding's repro is built from.
        tolerance: The target's declared `Tolerance` (`conformance/tolerances.py`).

    Returns:
        `None` if the case is confirmed outside tolerance (or is not a plain scalar quantity, or
        quantcore raised outright -- both unambiguous defects with nothing to recheck
        numerically); otherwise an error string naming the target/quantity, for `build_report`
        to report and exit non-zero on rather than write the finding.
    """
    if case.within_tolerance:
        return (
            f"{case.function}.{case.quantity}: finding headline case has within_tolerance=True "
            "under its declared tolerance -- refusing to write a finding for an in-tolerance case"
        )
    if detail.quantcore_value is None:
        return None
    try:
        actual = float(ast.literal_eval(detail.quantcore_value))
        expected = float(ast.literal_eval(detail.reference_value))
    except (ValueError, SyntaxError, TypeError):
        return None
    if tolerance.is_close(actual, expected):
        return (
            f"{case.function}.{case.quantity}: finding headline case is within the declared "
            f"tolerance on recheck (actual={actual!r}, expected={expected!r}, tol={tolerance!r}) "
            "-- refusing to write a finding for it"
        )
    return None


def _returns_regen_lines(family: str) -> list[str]:
    """Plain-numpy lines reproducing `conformance.generators._simulate_returns` for ``family``.

    Assumes `rng` (a `numpy.random.Generator`) and `n` (int) are already bound in scope; binds
    `returns`, matching the clip `conformance/generators.py` applies to every family so a
    compounded path never touches or crosses zero.
    """
    if family == "normal":
        body = ["values = rng.normal(0.0005, 0.01, n)"]
    elif family == "student_t3":
        body = ["values = rng.standard_t(3, n) * 0.01"]
    elif family == "skewed":
        body = ["values = 0.02 * (rng.lognormal(mean=0.0, sigma=0.5, size=n) - 1.0)"]
    elif family == "ar1":
        body = [
            "eps = rng.normal(0.0, 0.01, n)",
            "values = np.empty(n, dtype=np.float64)",
            "values[0] = eps[0]",
            "phi = 0.6",
            "for t in range(1, n):",
            "    values[t] = phi * values[t - 1] + eps[t]",
        ]
    elif family == "garch11":
        body = [
            "omega, alpha, beta = 1e-6, 0.1, 0.85",
            "variance = np.empty(n, dtype=np.float64)",
            "values = np.empty(n, dtype=np.float64)",
            "z = rng.normal(0.0, 1.0, n)",
            "variance[0] = omega / (1.0 - alpha - beta)",
            "values[0] = np.sqrt(variance[0]) * z[0]",
            "for t in range(1, n):",
            "    variance[t] = omega + alpha * values[t - 1] ** 2 + beta * variance[t - 1]",
            "    values[t] = np.sqrt(variance[t]) * z[t]",
        ]
    elif family == "near_constant":
        body = ["values = np.full(n, 0.0004, dtype=np.float64) + rng.normal(0.0, 1e-9, n)"]
    elif family == "all_positive":
        body = ["values = np.abs(rng.normal(0.005, 0.01, n))"]
    elif family == "all_negative":
        body = ["values = -np.abs(rng.normal(0.005, 0.01, n))"]
    elif family == "tiny_scale":
        body = ["values = rng.normal(0.0, 1e-8, n)"]
    elif family == "huge_scale":
        body = ["values = rng.normal(0.0, 1e3, n)"]
    else:
        raise ValueError(f"no returns_series regeneration recipe for family {family!r}")
    return [*body, "returns = np.clip(np.asarray(values, dtype=np.float64), -0.999, None)"]


def _cointegration_regen_lines(family: str) -> list[str]:
    """Plain-numpy lines reproducing `conformance.generators.cointegration_pair` for ``family``.

    Assumes `rng` and `n` are already bound in scope; binds `y` and `x`.
    """
    if family == "cointegrated":
        return [
            "common = np.cumsum(rng.normal(0.0, 1.0, n))",
            "x = common + rng.normal(0.0, 0.1, n)",
            "y = 1.5 * common + rng.normal(0.0, 0.3, n)",
        ]
    if family == "independent_rw":
        return [
            "x = np.cumsum(rng.normal(0.0, 1.0, n))",
            "y = np.cumsum(rng.normal(0.0, 1.0, n))",
        ]
    if family == "near_unit_root":
        return [
            "phi = 0.995",
            "x = np.empty(n, dtype=np.float64)",
            "y = np.empty(n, dtype=np.float64)",
            "x[0] = rng.normal(0.0, 1.0)",
            "y[0] = rng.normal(0.0, 1.0)",
            "ex = rng.normal(0.0, 1.0, n)",
            "ey = rng.normal(0.0, 1.0, n)",
            "for t in range(1, n):",
            "    x[t] = phi * x[t - 1] + ex[t]",
            "    y[t] = phi * y[t - 1] + ey[t]",
        ]
    raise ValueError(f"no cointegration_pair regeneration recipe for family {family!r}")


def _hedge_ratio_regen_lines(family: str) -> list[str]:
    """Plain-numpy lines reproducing `conformance.test_tier2_filtering.hedge_ratio_problem`.

    Assumes `rng` and `n` are already bound in scope; binds `x` and `y`.
    """
    if family == "stable_hedge":
        return [
            "x = rng.normal(0.0, 1.0, n) + 5.0",
            "y = 1.5 * x + rng.normal(0.0, 0.1, n)",
        ]
    if family == "drifting_hedge":
        return [
            "x = rng.normal(0.0, 1.0, n) + 5.0",
            "beta_path = 1.0 + 0.5 * np.sin(np.linspace(0.0, 3.0, n))",
            "y = beta_path * x + rng.normal(0.0, 0.1, n)",
        ]
    raise ValueError(f"no hedge_ratio_problem regeneration recipe for family {family!r}")


def _adf_series_regen_lines(family: str) -> list[str]:
    """Plain-numpy lines reproducing `conformance.test_tier1_statistics_extra._adf_series`.

    Assumes `rng` and `n` are already bound in scope; binds `series`.
    """
    if family == "stationary":
        return [
            "innovations = rng.normal(0.0, 1.0, n)",
            "phi = 0.5",
            "series = np.empty(n, dtype=np.float64)",
            "series[0] = innovations[0]",
            "for t in range(1, n):",
            "    series[t] = phi * series[t - 1] + innovations[t]",
        ]
    if family == "unit_root":
        return ["series = np.cumsum(rng.normal(0.0, 1.0, n))"]
    raise ValueError(f"no _adf_series regeneration recipe for family {family!r}")


@dataclasses.dataclass(frozen=True)
class RegenRecipe:
    """How to regenerate a target's generated array arguments from a recorded seed.

    Attributes:
        lines: Given a generator family, the plain-numpy lines that bind every name in `arrays`,
            assuming `rng = numpy.random.default_rng(seed)` and the length `n` are in scope. They
            repeat the arithmetic of the generator the target's conformance test draws from.
        arrays: Names of the arguments those lines bind, each an array of length `n`. Every other
            argument of the target's call is a scalar, inlined from the recorded input.
    """

    lines: Callable[[str], list[str]]
    arrays: tuple[str, ...]


_RETURNS_RECIPE = RegenRecipe(lines=_returns_regen_lines, arrays=("returns",))

REGEN_RECIPES: dict[str, RegenRecipe] = {
    "maximum_drawdown": _RETURNS_RECIPE,
    "engle_granger_test": RegenRecipe(lines=_cointegration_regen_lines, arrays=("y", "x")),
    "calmar_ratio": _RETURNS_RECIPE,
    "calmar_ratio_arithmetic": _RETURNS_RECIPE,
    "adf_test": RegenRecipe(lines=_adf_series_regen_lines, arrays=("series",)),
    "fit_garch_11": _RETURNS_RECIPE,
    "egarch_fit": _RETURNS_RECIPE,
    "gjr_garch_fit": _RETURNS_RECIPE,
    "dynamic_hedge_ratio": RegenRecipe(lines=_hedge_ratio_regen_lines, arrays=("y", "x")),
}


def _regenerated_input_lines(function: str, quantity: str, detail: ReproDetail) -> list[str]:
    """Lines binding a repro's arguments by regenerating its arrays from the recorded seed.

    The arrays the target's `RegenRecipe` names are rebuilt from `detail.seed` and the recorded
    array length; every other argument is a scalar and is inlined from `detail.inputs_repr`.

    Raises:
        ValueError: The target has no regeneration recipe, the case has no recorded seed, or a
            non-regenerated argument is not a plain scalar literal.
    """
    config = REPRO_CONFIGS[function]
    label = f"{function}.{quantity}"
    too_long = (
        f"shortest failing input has {count_numbers_in_repr(detail.inputs_repr)} numbers (> "
        f"{REPRO_INLINE_MAX_NUMBERS})"
    )
    recipe = REGEN_RECIPES.get(function)
    if recipe is None:
        raise ValueError(f"{label}: {too_long} and no recipe to regenerate it from its seed")
    if detail.seed is None:
        raise ValueError(f"{label}: {too_long} and no recorded seed to regenerate it from")

    element_reprs = dict(zip(config.arg_names, _split_top_level(detail.inputs_repr), strict=True))
    n = _numbers_in_element_repr(element_reprs[recipe.arrays[0]])
    lines = [
        f"# family={detail.family!r} seed={detail.seed!r}",
        f"rng = np.random.default_rng({detail.seed!r})",
        f"n = {n!r}",
        *recipe.lines(detail.family),
    ]
    for name in config.arg_names:
        if name in recipe.arrays:
            continue
        try:
            scalar = ast.literal_eval(element_reprs[name])
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"{label}: argument {name!r} is neither regenerated nor a scalar literal: "
                f"{element_reprs[name][:80]!r}"
            ) from exc
        lines.append(f"{name} = {scalar!r}")
    return lines


def render_minimal_repro(function: str, quantity: str, detail: ReproDetail) -> str:
    """Build a standalone Python repro for one quantity's shortest failing case.

    Imports only numpy, quantcore, and the target's oracle library, never this suite. Inlines the
    input as recorded when it has at most `REPRO_INLINE_MAX_NUMBERS` numbers in total; otherwise
    regenerates its arrays from `detail.seed` with the same plain-numpy arithmetic the target's
    generator uses for `detail.family` (`REGEN_RECIPES`).

    A quantity whose output is itself a scalar (the common case) always prints
    `"expected (oracle): ..."`, `"actual (quantcore): ..."`, and `"abs diff: ..."`, parsed back by
    `parse_repro_stdout`. A quantity in `config.summary_expr` instead prints a single
    `"summary: ..."` line (parsed by `parse_repro_summary`) -- for a quantity whose raw output is
    not itself a scalar (e.g. a decoded-state array), the SUMMARY expression reduces it to the
    one scalar number that demonstrates the finding.

    Args:
        function: Target name; must be a key of `REPRO_CONFIGS`.
        quantity: Quantity name; must be a key of that config's `quantity_index` or
            `summary_expr`.
        detail: The `ReproDetail` to build a repro for, ordinarily the one
            `select_shortest_failing_detail` chose.

    Returns:
        The repro's full source text.

    Raises:
        ValueError: The input is too long to inline and cannot be regenerated (see
            `_regenerated_input_lines`).
    """
    config = REPRO_CONFIGS[function]
    summary_expr = config.summary_expr.get(quantity)
    total = count_numbers_in_repr(detail.inputs_repr)

    lines = ["import numpy as np", config.module_import, config.oracle_import, ""]

    if total <= REPRO_INLINE_MAX_NUMBERS:
        # Safe to fully reconstruct: numpy only abbreviates a repr once it is far longer than
        # REPRO_INLINE_MAX_NUMBERS, so a repr this short was never truncated.
        args = reconstruct_args(detail.inputs_repr)
        for name, value in zip(config.arg_names, args, strict=True):
            literal = repr(value).replace("array(", "np.array(")
            lines.append(f"{name} = {literal}")
    else:
        lines += _regenerated_input_lines(function, quantity, detail)

    lines.append("")
    lines.append(f"quantcore_result = {config.quantcore_call}")
    lines.append(f"oracle_result = {config.oracle_call}")

    if summary_expr is not None:
        lines.append(f"summary = {summary_expr}")
        lines.append('print("summary:", repr(summary))')
        return "\n".join(lines)

    quantcore_index, oracle_index = config.quantity_index[quantity]
    actual_expr = (
        "quantcore_result" if quantcore_index is None else f"quantcore_result[{quantcore_index}]"
    )
    expected_expr = "oracle_result" if oracle_index is None else f"oracle_result[{oracle_index}]"
    lines.append(f"actual = float({actual_expr})")
    lines.append(f"expected = float({expected_expr})")
    lines.append('print("expected (oracle):", repr(expected))')
    lines.append('print("actual (quantcore):", repr(actual))')
    lines.append('print("abs diff:", repr(abs(actual - expected)))')
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class ReproRunResult:
    """Outcome of executing one repro snippet in a subprocess.

    Attributes:
        returncode: The subprocess's exit code, or `None` if it timed out.
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: Whether the subprocess exceeded its timeout and was killed.
    """

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def run_repro_subprocess(code: str, timeout: float = REPRO_TIMEOUT_SECONDS) -> ReproRunResult:
    """Execute ``code`` as a standalone Python program and capture its outcome.

    A small, isolated function so unit tests can exercise the pass, crash, and timeout paths with
    tiny synthetic snippets instead of a real finding's repro.

    Args:
        code: Python source to run with `sys.executable -c`.
        timeout: Seconds to allow before the subprocess is killed.

    Returns:
        A `ReproRunResult` describing what happened; never raises on a crash or timeout.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ReproRunResult(returncode=None, stdout=stdout, stderr=stderr, timed_out=True)
    return ReproRunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


_EXPECTED_RE = re.compile(r"^expected \(oracle\): (.+)$", re.MULTILINE)
_ACTUAL_RE = re.compile(r"^actual \(quantcore\): (.+)$", re.MULTILINE)
_ABS_DIFF_RE = re.compile(r"^abs diff: (.+)$", re.MULTILINE)
_SUMMARY_RE = re.compile(r"^summary: (.+)$", re.MULTILINE)


def parse_repro_summary(stdout: str) -> float:
    """Parse a repro's printed `summary` value back into a float.

    Counterpart to `parse_repro_stdout`, for a quantity built with `ReproCallConfig.summary_expr`
    (a non-scalar-quantity repro that prints one reduced scalar instead of expected/actual/diff).

    Args:
        stdout: Standard output captured from a repro built by `render_minimal_repro`.

    Returns:
        The parsed summary value.

    Raises:
        ValueError: The `"summary: ..."` line is missing from ``stdout``.
    """
    match = _SUMMARY_RE.search(stdout)
    if not match:
        raise ValueError(f"repro output is missing a summary line: {stdout!r}")
    return float(match.group(1))


def parse_repro_stdout(stdout: str) -> tuple[float, float, float]:
    """Parse a repro's printed `expected`, `actual`, and `abs diff` values back into floats.

    Args:
        stdout: Standard output captured from a repro built by `render_minimal_repro`.

    Returns:
        `(expected, actual, abs_diff)`.

    Raises:
        ValueError: One or more of the three expected lines is missing from ``stdout``.
    """
    expected_match = _EXPECTED_RE.search(stdout)
    actual_match = _ACTUAL_RE.search(stdout)
    abs_diff_match = _ABS_DIFF_RE.search(stdout)
    if not (expected_match and actual_match and abs_diff_match):
        raise ValueError(f"repro output is missing expected/actual/abs-diff lines: {stdout!r}")
    return (
        float(expected_match.group(1)),
        float(actual_match.group(1)),
        float(abs_diff_match.group(1)),
    )


def repro_still_mismatches(expected: float, actual: float, tolerance: Tolerance) -> bool:
    """Whether a repro's printed values still disagree beyond the target's written tolerance.

    Args:
        expected: The oracle value a repro printed.
        actual: The quantcore value a repro printed.
        tolerance: The target's written `Tolerance` (see `conformance/tolerances.py`).

    Returns:
        `True` if `|actual - expected| > atol + rtol * |expected|` (the defect still reproduces).
    """
    return abs(actual - expected) > tolerance.atol + tolerance.rtol * abs(expected)


@dataclasses.dataclass(frozen=True)
class VerifiedRepro:
    """One quantity's executed, verified minimal repro, ready to embed in a finding.

    Exactly one of two shapes: an ordinary scalar-quantity repro carries `expected`/`actual`/
    `abs_diff` (`summary` is `None`); a `ReproCallConfig.summary_expr` repro instead carries
    `summary` (`expected`/`actual`/`abs_diff` are all `None`) -- see docs on `ReproCallConfig` and
    `render_minimal_repro`.

    Attributes:
        quantity: Which named quantity this repro demonstrates.
        code: The repro's full source text.
        expected: The oracle value printed by executing ``code``, or `None` in summary mode.
        actual: The quantcore value printed by executing ``code``, or `None` in summary mode.
        abs_diff: `|actual - expected|`, as printed by executing ``code``, or `None` in summary
            mode.
        summary: The single scalar SUMMARY value printed by executing ``code``, or `None` for an
            ordinary scalar-quantity repro.
    """

    quantity: str
    code: str
    expected: float | None
    actual: float | None
    abs_diff: float | None
    summary: float | None = None


def verify_repro(
    function: str, quantity: str, detail: ReproDetail, tolerance: Tolerance
) -> tuple[VerifiedRepro | None, str | None]:
    """Build, execute, and verify one quantity's minimal repro.

    A quantity declared in `REPRO_CONFIGS[function].summary_expr` is verified against that
    config's `summary_ok` (whether the printed SUMMARY means the case does not reproduce),
    instead of the ordinary rtol/atol tolerance check `repro_still_mismatches` applies to a
    scalar-quantity repro.

    Args:
        function: Target name.
        quantity: Quantity name.
        detail: The `ReproDetail` to build the repro from (ordinarily the shortest failing one).
        tolerance: The target's written tolerance, to confirm the executed mismatch is real.
            Ignored for a `summary_expr` quantity.

    Returns:
        `(verified, error)`: exactly one is `None`. ``error`` names ``function``/``quantity`` and
        explains what went wrong (timeout, non-zero exit, unparsable output, or a mismatch that
        did not reproduce); ``verified`` carries the executed numbers to embed in the finding.
    """
    config = REPRO_CONFIGS.get(function)
    summary_expr = config.summary_expr.get(quantity) if config is not None else None
    code = render_minimal_repro(function, quantity, detail)
    result = run_repro_subprocess(code, timeout=REPRO_TIMEOUT_SECONDS)
    label = f"{function}.{quantity}"
    if result.timed_out:
        return None, f"{label}: repro timed out after {REPRO_TIMEOUT_SECONDS}s"
    if result.returncode != 0:
        return None, f"{label}: repro exited {result.returncode}: {result.stderr.strip()}"

    if summary_expr is not None:
        # `summary_expr` is only ever non-`None` when `config` (its own source) is not `None`.
        assert config is not None
        try:
            summary = parse_repro_summary(result.stdout)
        except ValueError as exc:
            return None, f"{label}: {exc}"
        summary_ok = config.summary_ok[quantity]
        if summary_ok(summary):
            return None, (
                f"{label}: repro executed but summary={summary!r} no longer reproduces the "
                "finding; it would be stale"
            )
        verified = VerifiedRepro(
            quantity=quantity, code=code, expected=None, actual=None, abs_diff=None, summary=summary
        )
        return verified, None

    try:
        expected, actual, abs_diff = parse_repro_stdout(result.stdout)
    except ValueError as exc:
        return None, f"{label}: {exc}"
    if not repro_still_mismatches(expected, actual, tolerance):
        return None, (
            f"{label}: repro executed but no longer mismatches beyond tolerance "
            f"(expected={expected!r}, actual={actual!r}); the finding would be stale"
        )
    verified = VerifiedRepro(
        quantity=quantity, code=code, expected=expected, actual=actual, abs_diff=abs_diff
    )
    return verified, None


# ---------------------------------------------------------------------------
# Evidence blocks
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantityEvidence:
    """One quantity's generated evidence block for a finding.

    `worst_abs`/`worst_rel` are read straight from the conformance map's own `by_quantity` entry
    (never recomputed independently), so they are always equal to `CONFORMANCE.md`'s summary
    table by construction.

    Attributes:
        cases: Total recorded cases for this quantity.
        failing: How many of those were outside tolerance (or quantcore raised), not counting
            ill-conditioned cases.
        ill_conditioned: How many cases were excluded from the decision because their effective
            tolerance exceeded the ill-conditioning cap; none of them counts as failing.
        worst_abs: Largest absolute error, or `None` if none was computable.
        worst_rel: Largest relative error, or `None` if none was computable.
        worst_family: Generator family of the case with the largest relative error.
        failing_by_family: Failing-case count per generator family.
        failing_by_bucket: Failing-case count per length bucket (`"short"`, `"medium"`, `"long"`).
    """

    cases: int
    failing: int
    ill_conditioned: int
    worst_abs: float | None
    worst_rel: float | None
    worst_family: str | None
    failing_by_family: dict[str, int]
    failing_by_bucket: dict[str, int]


def decision_pairs(
    pairs: Sequence[tuple[CaseResult, ReproDetail]],
) -> list[tuple[CaseResult, ReproDetail]]:
    """Drop the ill-conditioned cases from `(case, detail)` pairs.

    An ill-conditioned case is excluded from the status decision, so it can be neither counted
    as a failure in a finding nor used as the finding's headline case.
    """
    return [(case, detail) for case, detail in pairs if not case.ill_conditioned]


def evidence_for_quantity(
    pairs: Sequence[tuple[CaseResult, ReproDetail]],
    quantity_errors: QuantityErrors | None,
) -> QuantityEvidence:
    """Aggregate one quantity's recorded cases into the evidence block a finding shows.

    Pure and I/O-free: works entirely from already-loaded records.

    Args:
        pairs: `(case, detail)` pairs for one target's quantity, as `matched_pairs` returns.
        quantity_errors: That quantity's entry in the conformance map's `by_quantity`, whose
            `max_abs_error`/`max_rel_error` are reused verbatim so the finding's numbers always
            equal the summary table's.

    Returns:
        The aggregated `QuantityEvidence`.
    """
    decided = decision_pairs(pairs)
    failing_pairs = [(case, detail) for case, detail in decided if not case.within_tolerance]
    worst = _worst_detail([detail for _case, detail in pairs])

    failing_by_family: dict[str, int] = {}
    failing_by_bucket: dict[str, int] = {}
    for case, _detail in failing_pairs:
        failing_by_family[case.family] = failing_by_family.get(case.family, 0) + 1
        failing_by_bucket[case.length_bucket] = failing_by_bucket.get(case.length_bucket, 0) + 1

    return QuantityEvidence(
        cases=len(pairs),
        failing=len(failing_pairs),
        ill_conditioned=len(pairs) - len(decided),
        worst_abs=quantity_errors.max_abs_error if quantity_errors is not None else None,
        worst_rel=quantity_errors.max_rel_error if quantity_errors is not None else None,
        worst_family=worst.family if worst is not None else None,
        failing_by_family=failing_by_family,
        failing_by_bucket=failing_by_bucket,
    )


# ---------------------------------------------------------------------------
# Rendering and writing findings
# ---------------------------------------------------------------------------


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def _fmt_err(value: float | None) -> str:
    return "—" if value is None else f"{value:.3e}"


def _sign_test_lines(
    estimator: IterativeEstimator, decision: IterativeEstimatorDecision
) -> list[str]:
    """The evidence block a finding shows for an iterative estimator's sign test."""
    better = "larger" if estimator.sense == "maximize" else "smaller"
    return [
        "### Sign test on the attained objective",
        "",
        f"- decision quantity: `{estimator.quantity}` ({estimator.sense}d: a {better} value is "
        "the better solution)",
        f"- well-specified cases: quantcore better on {decision.wins}, tied on {decision.ties}, "
        f"worse on {decision.losses}",
        f"- one-sided binomial sign test (`H0: p_loss = 0.5`): p-value {decision.p_value:.4g} "
        f"(alpha {SIGN_TEST_ALPHA:g})",
        f"- median gap, positive when quantcore's objective is the better one: "
        f"{decision.median_gap:.4g}",
        f"- well-specified cases where quantcore did not converge: "
        f"{decision.non_convergence_fraction:.1%}",
        "",
    ]


def render_finding(
    name: str,
    entry: TargetRecord,
    conformance_map: ConformanceMap,
    source: FindingSource,
    *,
    evidence_by_quantity: dict[str, QuantityEvidence],
    repros: Sequence[VerifiedRepro],
    sign_test: tuple[IterativeEstimator, IterativeEstimatorDecision] | None = None,
) -> str:
    """Render one target's finding, merging its hand-written source with this run's evidence.

    Args:
        name: Target name.
        entry: This target's entry in the conformance map.
        conformance_map: The full conformance map (for the quantcore and oracle versions).
        source: The parsed `findings_src/<name>.md`.
        evidence_by_quantity: Aggregated evidence, one entry per recorded quantity.
        repros: Executed, verified repros, one per quantity, in the same order as
            ``evidence_by_quantity``.
        sign_test: For an iterative estimator, how its objective is read and the sign test's
            outcome on this run; `None` for any other target.

    Returns:
        The finding's full Markdown text, starting with the generated-file marker comment.
    """
    oracle_versions_text = ", ".join(
        f"{k} {v}" for k, v in sorted(conformance_map.oracle_versions.items())
    )
    repro_by_quantity = {repro.quantity: repro for repro in repros}

    sections = [
        GENERATED_FILE_COMMENT,
        "",
        f"# {source.title}",
        "",
        f"- **Severity:** {source.severity}",
        f"- **Status:** `{entry.status}`",
        f"- **quantcore version:** `{conformance_map.quantcore_version}`",
        f"- **Target:** `{name}`",
        f"- **Location:** {source.location}",
        f"- **Oracle:** `{entry.oracle}`",
        f"- **Library versions:** {oracle_versions_text}",
        "",
        "## Root cause",
        "",
        source.root_cause,
        "",
        "## Standard definition and citation",
        "",
        source.standard_definition,
        "",
        "## Evidence from this run",
        "",
    ]
    if sign_test is not None:
        sections += _sign_test_lines(*sign_test)
    for quantity in sorted(evidence_by_quantity):
        ev = evidence_by_quantity[quantity]
        sections += [
            f"### `{quantity}`",
            "",
            f"- cases run: {ev.cases}; failing: {ev.failing}"
            + (
                f"; ill-conditioned (excluded from the decision): {ev.ill_conditioned}"
                if ev.ill_conditioned
                else ""
            ),
            f"- worst abs error: {_fmt_err(ev.worst_abs)}; "
            f"worst rel error: {_fmt_err(ev.worst_rel)} "
            f"(worst family: `{ev.worst_family or '—'}`)",
            f"- failing cases per family: {_fmt_counts(ev.failing_by_family)}",
            f"- failing cases per length bucket: {_fmt_counts(ev.failing_by_bucket)}",
            "",
        ]
        repro = repro_by_quantity.get(quantity)
        if repro is not None:
            executed_line = (
                f"Executed this run: `summary` = `{repro.summary!r}`."
                if repro.summary is not None
                else (
                    f"Executed this run: `expected (oracle)` = `{repro.expected!r}`, "
                    f"`actual (quantcore)` = `{repro.actual!r}`, abs diff = `{repro.abs_diff!r}`."
                )
            )
            sections += [
                executed_line,
                "",
                "```python",
                repro.code,
                "```",
                "",
            ]
    sections += [
        "## Proposed patch",
        "",
        source.proposed_patch,
        "",
        "## Promotion test",
        "",
        source.promotion_test,
        "",
    ]
    return "\n".join(sections)


def _write_findings(
    conformance_map: ConformanceMap, cases: Sequence[CaseResult], details: Sequence[ReproDetail]
) -> list[str]:
    """Render, verify, and write every `defect`/`non-standard` target's finding.

    Args:
        conformance_map: The full conformance map.
        cases: Every recorded `CaseResult`.
        details: Every recorded `ReproDetail`.

    Returns:
        One error line per finding that could not be produced (missing source, repro failed to
        reproduce, crashed, or timed out); empty on full success. Nothing is written to disk for a
        finding that produced an error.
    """
    errors: list[str] = []
    FINDINGS_DIR.mkdir(exist_ok=True)
    for name, entry in sorted(conformance_map.targets.items()):
        if entry.finding is None:
            continue
        source_path = _finding_source_path(name)
        if not source_path.exists():
            errors.append(f"{name}: missing finding source {source_path}")
            continue
        try:
            source = load_finding_source(source_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        quantities = sorted(entry.by_quantity) or ["value"]
        evidence_by_quantity: dict[str, QuantityEvidence] = {}
        repros: list[VerifiedRepro] = []
        target_had_error = False
        for quantity in quantities:
            pairs = matched_pairs(cases, details, function=name, quantity=quantity)
            evidence_by_quantity[quantity] = evidence_for_quantity(
                pairs, entry.by_quantity.get(quantity)
            )
            decided = decision_pairs(pairs)
            shortest_failing = select_shortest_failing_detail(decided)
            repro_config = REPRO_CONFIGS.get(name)
            quantity_is_reproable = repro_config is not None and (
                quantity in repro_config.quantity_index or quantity in repro_config.summary_expr
            )
            if shortest_failing is None or not quantity_is_reproable:
                # Either this quantity itself never failed its tolerance (e.g. adf_test's
                # statistic, which matches statsmodels exactly while its p_value, a separate
                # quantity, does not), or it has no repro recipe at all (e.g. an iterative
                # estimator's `params`, never compared, so a case where quantcore *raised*
                # entirely -- leaving no quantcore value for any quantity, including this one --
                # can still show up as "failing" here even though there is nothing meaningful to
                # build a scalar repro from). Either way, the target's status still reflects
                # the true failing quantity; this one's evidence section is rendered without a
                # repro block instead of erroring the whole finding.
                continue
            tolerance = TOLERANCES[name]
            headline_case = next(c for c, d in decided if d is shortest_failing)
            violation = headline_case_tolerance_violation(
                headline_case, shortest_failing, tolerance
            )
            if violation is not None:
                errors.append(violation)
                target_had_error = True
                continue
            try:
                verified, error = verify_repro(name, quantity, shortest_failing, tolerance)
            except ValueError as exc:
                verified, error = None, str(exc)
            if error is not None:
                errors.append(error)
                target_had_error = True
                continue
            assert verified is not None
            repros.append(verified)

        if target_had_error:
            continue

        estimator = ITERATIVE_ESTIMATORS.get(name)
        sign_test = (
            None
            if estimator is None
            else (estimator, iterative_estimator_decision(name, cases, details, estimator))
        )
        content = render_finding(
            name,
            entry,
            conformance_map,
            source,
            evidence_by_quantity=evidence_by_quantity,
            repros=repros,
            sign_test=sign_test,
        )
        finding_file = FINDINGS_DIR / Path(entry.finding).name
        _assert_no_extreme_magnitude(content, finding_file)
        finding_file.write_text(content, encoding="utf-8")
    return errors


def remove_stale_findings(findings_dir: Path, referenced: Iterable[str]) -> list[Path]:
    """Delete generated finding files that the current conformance map does not reference.

    A finding's file name carries its position among the current findings, so it changes when a
    target gains or loses a finding; without this sweep the old file would sit next to the new
    one. Only files that start with the generator's header are removed, never a hand-written
    file.

    Args:
        findings_dir: Directory holding the generated `NNN-<target>.md` findings.
        referenced: File names (not paths) the current map's `finding` fields point to.

    Returns:
        The paths that were removed, sorted.
    """
    keep = set(referenced)
    removed: list[Path] = []
    for path in sorted(findings_dir.glob("*.md")):
        if path.name in keep:
            continue
        with path.open(encoding="utf-8") as handle:
            first_line = handle.readline().rstrip("\n")
        if first_line == GENERATED_FILE_COMMENT:
            path.unlink()
            removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# The human-readable report
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3e}"


def _cell(text: str) -> str:
    """Escape text for one Markdown table cell: a literal `|` would otherwise end the cell."""
    return text.replace("|", "\\|")


def _fmt_recorded_value(value_repr: str | None) -> str:
    """Render a recorded value's `repr()` for the report, to 6 significant digits if it is a float.

    The full-precision repr differs in its last digits between platforms; a report that is
    committed keeps only the digits that do not.
    """
    if value_repr is None:
        return "—"
    try:
        return f"{float(value_repr):.6g}"
    except ValueError:
        return value_repr


def _render_oracle_cross_checks(
    oracle_rows: Sequence[OracleCrossCheckRow], oracle_examples: Sequence[OracleDisagreement]
) -> list[str]:
    """Render the "Oracle cross-checks" and, where any row disagrees, "Oracle disagreements"."""
    lines = [
        "",
        "## Oracle cross-checks",
        "",
        "A second, independent oracle checked against the same target's own oracle (its standard "
        "definition), never against quantcore, and never affecting a status.",
        "",
    ]
    if not oracle_rows:
        lines += ["No second-oracle cross-checks were recorded by this run.", ""]
        return lines
    lines += [
        "| target | second oracle | quantity | cases | max abs diff | max rel diff | agrees | "
        "raised/non-finite |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in oracle_rows:
        lines.append(
            f"| `{row.function}` | `{row.oracle}` | `{row.quantity}` | {row.cases} | "
            f"{_fmt(row.max_abs_diff)} | {_fmt(row.max_rel_diff)} | "
            f"{'yes' if row.agrees else 'no'} | {row.raised_or_nonfinite} |"
        )
    lines.append("")

    disagreeing = [row for row in oracle_rows if not row.agrees]
    if not disagreeing:
        return lines

    lines += ["### Oracle disagreements", ""]
    examples_by_key = {(e.function, e.oracle, e.quantity): e for e in oracle_examples}
    for row in disagreeing:
        key = (row.function, row.oracle, row.quantity)
        example = examples_by_key.get(key)
        lines += [f"#### `{row.function}` vs. `{row.oracle}` (`{row.quantity}`)", ""]
        explanation = ORACLE_DISAGREEMENT_EXPLANATIONS.get((row.function, row.oracle))
        if explanation is not None:
            lines += [explanation.rstrip(), ""]
        if example is not None:
            seed_text = f"seed={example.seed!r}" if example.seed is not None else "hand-picked"
            reference_value = _fmt_recorded_value(example.reference_value)
            if example.raised is not None:
                lines += [
                    f"Example this run (family=`{example.family}`, {seed_text}): the second "
                    f"oracle raised or returned a non-finite value (`{example.raised}`); "
                    f"reference value = `{reference_value}`.",
                    "",
                ]
            else:
                lines += [
                    f"Example this run (family=`{example.family}`, {seed_text}): second oracle "
                    f"value = `{_fmt_recorded_value(example.oracle_value)}`, reference value = "
                    f"`{reference_value}`, abs diff = `{_fmt(example.abs_diff)}`.",
                    "",
                ]
    return lines


def _iterative_estimator_section(
    cases: Sequence[CaseResult], details: Sequence[ReproDetail]
) -> list[str]:
    """Render the win/tie/loss-by-family table and the sign-test decision per iterative estimator.

    The report shows win/tie/loss counts per family for each estimator (a reporting granularity,
    `TIE_ABS_GAP`), then the one-sided binomial sign test's outcome on well-specified cases only
    (the decision `classify_iterative_estimator` actually makes, at the target's own declared
    `Tolerance`): its p-value, the median gap, and the resulting status.
    """
    lines = [
        "",
        "## Iterative estimators (win/tie/loss)",
        "",
        "A fitter or optimiser without a unique numerical optimum is judged on the objective it "
        "attains at its own solution, against the objective the oracle attains at its own, both "
        "evaluated by one shared function: the Gaussian log-likelihood for the fitters, the "
        "portfolio objective for the optimisers. A `win` means quantcore attained the better "
        "objective: the larger one where the objective is maximised, the smaller one where it is "
        "minimised. In the first table a `win` or `loss` is a gap of at least "
        f"`{TIE_ABS_GAP:g}` in the objective's own units and anything smaller is a `tie`; this "
        "reporting granularity is separate from the decision rule below.",
        "",
        "| target | family | win | tie | loss |",
        "|---|---|---|---|---|",
    ]
    decision_rows: list[str] = []
    for name, estimator in sorted(ITERATIVE_ESTIMATORS.items()):
        objective_cases = objective_cases_for(name, cases, details, estimator)
        if not objective_cases:
            continue
        tallies = tally_objective_outcomes(objective_cases)
        for family in sorted(tallies):
            tally = tallies[family]
            lines.append(
                f"| `{name}` | `{family}` | {tally['win']} | {tally['tie']} | {tally['loss']} |"
            )
        decision = iterative_estimator_decision(name, cases, details, estimator)
        decision_rows.append(
            f"| `{name}` | `{estimator.quantity}` | {estimator.sense} | {decision.wins} | "
            f"{decision.ties} | {decision.losses} | {decision.p_value:.4g} | "
            f"{_fmt(round_significant(decision.median_gap))} | "
            f"{decision.non_convergence_fraction:.1%} | `{decision.status}` |"
        )

    lines += [
        "",
        "One-sided binomial sign test on well-specified cases only "
        f"(`H0: p_loss = 0.5`, alpha={SIGN_TEST_ALPHA:g}); a case counts as a win or a loss only "
        "when its gap exceeds the target's own declared tolerance, and misspecified-family cases "
        "never enter this test or its counts. The median gap is positive when quantcore's "
        "objective is the better one.",
        "",
        "| target | objective | sense | wins | ties | losses | p-value | median gap | "
        "non-convergence | decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *decision_rows,
    ]
    return lines


def _condition_aware_section(cases: Sequence[CaseResult]) -> list[str]:
    """Render the per-target ill-conditioned case count, for a condition-aware tolerance.

    An ill-conditioned case is excluded from the decision but must still be counted and
    reported.
    """
    counts = ill_conditioned_counts(cases)
    lines = [
        "",
        "## Condition-aware tolerance",
        "",
        f"Cases excluded from the pass/fail decision because their effective tolerance exceeded "
        f"the `{ILL_CONDITIONED_RTOL_CAP:g}` ill-conditioning cap (only targets with a "
        "`ConditionAwareTolerance`, e.g. `sharpe_ratio`, can have any). The build fails if fewer "
        f"than {MIN_WELL_CONDITIONED_FRACTION:.0%} of a target's cases are well-conditioned.",
        "",
        "| target | ill-conditioned | total | well-conditioned |",
        "|---|---|---|---|",
    ]
    header_row_count = len(lines)
    for name in sorted(counts):
        ill, total = counts[name]
        if ill == 0:
            continue
        fraction = 1.0 - (ill / total) if total else 1.0
        lines.append(f"| `{name}` | {ill} | {total} | {fraction:.1%} |")
    if len(lines) == header_row_count:
        lines.append("| _none this run_ | | | |")
    return lines


def _coverage_section(
    declared: Mapping[str, DeclaredCoverage],
    counts: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> list[str]:
    """Render the per-target case-count-per-family and case-count-per-bucket tables."""
    lines = [
        "",
        "## Coverage",
        "",
        "Comparisons recorded by this run, per declared generator family and per declared bucket "
        "(for a series input the length buckets `short` 2-10, `medium` 11-500 and `long` 501 and "
        "above; other targets declare their own bucket names). An input counts once for every "
        "quantity recorded on it, and second-oracle cross-check records are included. The build "
        "fails if any declared cell is zero.",
        "",
        "| target | family | cases |",
        "|---|---|---|",
    ]
    for name in sorted(declared):
        family_counts, _bucket_counts = counts.get(name, ({}, {}))
        for family in declared[name].families:
            lines.append(f"| `{name}` | `{family}` | {family_counts.get(family, 0)} |")
    lines += ["", "| target | bucket | cases |", "|---|---|---|"]
    for name in sorted(declared):
        _family_counts, bucket_counts = counts.get(name, ({}, {}))
        for bucket in declared[name].buckets:
            lines.append(f"| `{name}` | `{bucket}` | {bucket_counts.get(bucket, 0)} |")
    return lines


def _status_table(conformance_map: ConformanceMap) -> list[str]:
    """Render the main table: one row per recorded quantity of every target."""
    lines = [
        "",
        "## Status per target",
        "",
        "One row per recorded quantity: a target recording several named quantities (e.g. "
        "`engle_granger_test`'s `statistic` and `p_value`) has one row per quantity, so a defect "
        "confined to a single quantity is visible rather than averaged away. The status is the "
        "target's; `rtol`/`atol` are the tolerance its cases were judged against "
        "(`|quantcore - oracle| <= atol + rtol * |oracle|`).",
        "",
        "| target | quantity | oracle | cases | max abs err | max rel err | rtol | atol | status | "
        "finding |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in sorted(conformance_map.targets.items()):
        rtol = "—" if entry.tolerance is None else f"{entry.tolerance.rtol:g}"
        atol = "—" if entry.tolerance is None else f"{entry.tolerance.atol:g}"
        finding = "—" if entry.finding is None else f"`{entry.finding}`"
        for quantity in sorted(entry.by_quantity) or ["—"]:
            errors = entry.by_quantity.get(quantity)
            quantity_cell = "—" if errors is None else f"`{quantity}`"
            quantity_cases = entry.cases if errors is None else errors.cases
            max_abs = entry.max_abs_error if errors is None else errors.max_abs_error
            max_rel = entry.max_rel_error if errors is None else errors.max_rel_error
            lines.append(
                f"| `{name}` | {quantity_cell} | {_cell(entry.oracle)} | {quantity_cases} | "
                f"{_fmt(max_abs)} | {_fmt(max_rel)} | {rtol} | {atol} | {entry.status} | "
                f"{finding} |"
            )
    return lines


def _findings_section(conformance_map: ConformanceMap) -> list[str]:
    """Render why each target that is not `verified` has the status it has."""
    lines = [
        "",
        "## Targets that are not verified",
        "",
        "A `defect` or `non-standard` target has a hand-written finding in "
        "`conformance/findings_src/`; building the report merges it with the run's evidence and "
        "an executed repro into the `finding` path of the table above, which is git-ignored.",
        "",
    ]
    not_verified = {
        name: entry
        for name, entry in sorted(conformance_map.targets.items())
        if entry.status != "verified"
    }
    if not not_verified:
        lines += ["Every target is `verified`.", ""]
        return lines
    for name, entry in not_verified.items():
        source = (
            f" ([`findings_src/{name}.md`](../conformance/findings_src/{name}.md))"
            if entry.finding is not None
            else ""
        )
        lines.append(f"- `{name}` (`{entry.status}`): {entry.reason}{source}")
    return lines


def render_conformance_md(
    conformance_map: ConformanceMap,
    cases: Sequence[CaseResult],
    details: Sequence[ReproDetail],
    counts: dict[str, tuple[dict[str, int], dict[str, int]]],
    oracle_rows: Sequence[OracleCrossCheckRow],
    oracle_examples: Sequence[OracleDisagreement],
    left_out: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Render the human-readable report written to `docs/CONFORMANCE.md`.

    Every number is either a count or an error magnitude rounded to 3 significant digits, rows
    are sorted by name, and nothing depends on the time or place of the build, so two builds of
    the same run produce the same text.

    Args:
        conformance_map: The map built from this run.
        cases: Every recorded `CaseResult` of the targets in the map.
        details: Every recorded `ReproDetail` of the same targets.
        counts: Per-target `(family_counts, bucket_counts)`, as `per_function_counts` returns.
        oracle_rows: Second-oracle cross-check rows, as `oracle_cross_check_rows` returns.
        oracle_examples: One example per disagreeing row, as `oracle_disagreement_examples`
            returns.
        left_out: Targets a build with `--allow-incomplete` left out of the map, each with the
            messages saying why; `None` or empty for a complete build.

    Returns:
        The report's full Markdown text, whose first line says it is generated and by which
        command.
    """
    status_counts = {
        status: sum(1 for entry in conformance_map.targets.values() if entry.status == status)
        for status in ("verified", "non-standard", "defect", "unverifiable")
    }
    lines = [
        f"_Generated by `{BUILD_COMMAND}` from a `{conformance_map.hypothesis_profile}`-profile "
        "run of the conformance suite. Do not edit by hand: change `conformance/` and regenerate "
        "(see `conformance/README.md`)._",
        "",
        "# Conformance report",
        "",
        "Every quantcore function the conformance suite covers is registered as a target and "
        "compared with an independent oracle on generated inputs. `verified` means quantcore was "
        "within the written tolerance of the oracle on every case and never raised or, for an "
        "iterative estimator, that the objective it attains is not systematically worse than the "
        "oracle's. `non-standard` means the result is sound but not the standard one: a "
        "documented alternative definition or, for an iterative estimator, an objective that is "
        "systematically but only slightly worse than the oracle's. `defect` means it is outside "
        "tolerance with no such explanation. `unverifiable` means there is no oracle, or "
        "agreement is recorded without being judged.",
        "",
        f"quantcore version: `{conformance_map.quantcore_version}`",
        "",
        "Library versions: "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(conformance_map.oracle_versions.items())),
        "",
        f"Hypothesis profile: `{conformance_map.hypothesis_profile}`",
        "",
        "| status | targets |",
        "|---|---|",
        *(f"| {status} | {count} |" for status, count in status_counts.items()),
        f"| total | {len(conformance_map.targets)} |",
    ]
    if left_out:
        lines += [
            "",
            "## Left out of this build",
            "",
            "This report was built with `--allow-incomplete`. The registered targets below are "
            "not in it, and a build without that flag fails until they are completed.",
            "",
        ]
        for name in sorted(left_out):
            lines += [f"- {_cell(message)}" for message in left_out[name]]
    declared = {
        name: coverage
        for name, coverage in DECLARED_COVERAGE.items()
        if name in conformance_map.targets
    }
    lines += _status_table(conformance_map)
    lines += _findings_section(conformance_map)
    lines += _coverage_section(declared, counts)
    lines += _iterative_estimator_section(cases, details)
    lines += _condition_aware_section(cases)
    lines += _render_oracle_cross_checks(oracle_rows, oracle_examples)
    lines.append("")
    return "\n".join(lines)


def _assert_no_extreme_magnitude(content: str, path: Path) -> None:
    """Guard against a near-zero-denominator relative error leaking into a generated file.

    `aggregate_errors` excludes cases whose oracle magnitude is at or below `_ZERO_FLOOR` from
    `max_rel_error`, specifically so that dividing by the comparator's `1e-300` floor never
    surfaces as a meaningless `~1e+296` in a report; this is the belt-and-suspenders check that
    the exclusion actually worked, run on every generated file before it is written.
    """
    if "e+296" in content:
        raise AssertionError(
            f"{path} contains the string 'e+296', which indicates a near-zero-denominator "
            "relative error leaked past aggregate_errors()'s exclusion; this must never happen"
        )


def _run_check(conformance_map: ConformanceMap) -> int:
    """`--check`: compare ``conformance_map`` (freshly built) to the committed file.

    Returns:
        `0` if they represent the same decisions (see `decision_diff`); `1`, printing every
        differing field, otherwise; `1` if the committed file does not exist at all.
    """
    if not CONFORMANCE_MAP_PATH.exists():
        print(
            f"{CONFORMANCE_MAP_PATH} does not exist; run without --check first",
            file=sys.stderr,
        )
        return 1
    committed_map = conformance_map_from_json(CONFORMANCE_MAP_PATH.read_text(encoding="utf-8"))
    diff = decision_diff(committed_map, conformance_map)
    if diff:
        print(
            f"{CONFORMANCE_MAP_PATH} is stale (decisions changed): regenerate with "
            f"`{BUILD_COMMAND}`",
            file=sys.stderr,
        )
        for line in diff:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"{CONFORMANCE_MAP_PATH.name} is up to date.")
    return 0


def incomplete_targets(
    configuration: Mapping[str, Sequence[str]], coverage_gaps: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """Merge configuration errors and coverage gaps into one list of messages per target.

    Pure and I/O-free.

    Args:
        configuration: Per-name messages, as `configuration_errors` returns.
        coverage_gaps: Per-target messages, as `missing_coverage_by_target` returns.

    Returns:
        Mapping from every name with at least one message to its messages, configuration errors
        first, sorted by name.
    """
    merged: dict[str, list[str]] = {}
    for name in sorted({*configuration, *coverage_gaps}):
        merged[name] = [*configuration.get(name, ()), *coverage_gaps.get(name, ())]
    return merged


def _print_messages(heading: str, messages: Iterable[str]) -> None:
    print(heading, file=sys.stderr)
    for message in messages:
        print(f"  {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Build (or, with `--check`, verify) the conformance map, the report and the findings.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code: `0` on success; non-zero if a registered target is incomplete (see
        `configuration_errors` and `missing_coverage`) and `--allow-incomplete` was not given, if
        too few of a target's cases are well-conditioned, if a `defect`/`non-standard` target has
        no `findings_src/<name>.md`, if `--check` finds the committed `conformance_map.json`
        stale, if a finding's repro fails to execute or does not reproduce, or -- under the
        `explore` profile -- if at least one mismatch was recorded for a target the committed map
        calls `verified`.

    Under the `explore` profile this never writes any file (not the map, not the report, not a
    finding): it only prints every recorded mismatch for a `verified` target and exits
    accordingly (see `explore_mismatches`). `--check` is meaningless combined with `explore` and
    is refused.
    """
    parser = argparse.ArgumentParser(
        prog="python -m conformance.build_report",
        description="Turn the results of a conformance suite run into the conformance map, "
        "the report and the findings.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory, write nothing, and fail if the decisions differ from the "
        "committed conformance_map.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="leave out, instead of failing on, every registered target that is misconfigured "
        "or has a declared family or bucket with no recorded case",
    )
    args = parser.parse_args(argv)

    cases = _load_cases()
    details = _load_details()
    hypothesis_profile = _load_recorded_profile()

    if hypothesis_profile == "explore":
        if args.check:
            print("--check is meaningless combined with the explore profile", file=sys.stderr)
            return 1
        return _run_explore_report(cases, details)

    targets = dict(registry())
    counts = per_function_counts(cases)
    incomplete = incomplete_targets(
        configuration_errors(targets, DECLARED_COVERAGE, TOLERANCES, {c.function for c in cases}),
        missing_coverage_by_target(DECLARED_COVERAGE, counts),
    )
    condition_gate_failures = ill_conditioned_gate_failures(
        ill_conditioned_counts(cases), min_well_conditioned_fraction=MIN_WELL_CONDITIONED_FRACTION
    )
    incomplete_messages = [message for messages in incomplete.values() for message in messages]
    if incomplete and not args.allow_incomplete:
        _print_messages(
            f"Cannot build the conformance map: {len(incomplete)} target(s) are incomplete.",
            incomplete_messages,
        )
        if condition_gate_failures:
            _print_messages(
                "Targets below the well-conditioned case threshold:", condition_gate_failures
            )
        print(
            "Complete them, or pass --allow-incomplete to build from the other targets.",
            file=sys.stderr,
        )
        return 1
    if condition_gate_failures:
        _print_messages(
            "Targets below the well-conditioned case threshold:", condition_gate_failures
        )
        return 1
    if incomplete:
        _print_messages(
            f"--allow-incomplete: leaving {len(incomplete)} incomplete target(s) out of the map.",
            incomplete_messages,
        )
        targets = {name: target for name, target in targets.items() if name not in incomplete}
        cases = [c for c in cases if c.function in targets]
        details = [d for d in details if d.function in targets]
        counts = per_function_counts(cases)

    conformance_map = build_conformance_map(cases, details, hypothesis_profile, targets)

    source_gaps = missing_finding_sources(conformance_map.targets)
    if source_gaps:
        _print_messages("defect/non-standard targets missing a findings_src/ file:", source_gaps)
        return 1
    unused_sources = unused_finding_sources(conformance_map.targets)
    if unused_sources:
        _print_messages("Note: finding sources no target needs:", unused_sources)

    if args.check:
        return _run_check(conformance_map)

    generated_json = conformance_map_to_json(conformance_map)
    _assert_no_extreme_magnitude(generated_json, CONFORMANCE_MAP_PATH)
    finding_errors = _write_findings(conformance_map, cases, details)
    if finding_errors:
        _print_messages("Could not produce every finding:", finding_errors)
        return 1

    remove_stale_findings(
        FINDINGS_DIR,
        (
            Path(entry.finding).name
            for entry in conformance_map.targets.values()
            if entry.finding is not None
        ),
    )

    report = render_conformance_md(
        conformance_map,
        cases,
        details,
        counts,
        oracle_cross_check_rows(cases, details),
        oracle_disagreement_examples(cases, details),
        left_out=incomplete,
    )
    _assert_no_extreme_magnitude(report, CONFORMANCE_MD_PATH)

    CONFORMANCE_MAP_PATH.write_text(generated_json, encoding="utf-8")
    CONFORMANCE_MD_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {CONFORMANCE_MAP_PATH}, {CONFORMANCE_MD_PATH} and {FINDINGS_DIR}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
