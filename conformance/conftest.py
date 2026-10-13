"""pytest plugin: registers hypothesis profiles and dumps recorded conformance cases to disk.

Selecting the hypothesis profile happens here, once, at collection time, from the environment
variable `QUANTCORE_CONFORMANCE_PROFILE`: `"ci"` (the default, 200 examples per test,
`derandomize=True`), `"full"` (5000 examples, `derandomize=True`), or `"explore"` (5000 examples,
*not* derandomized so every run draws fresh inputs, `print_blob=True` so a failing example can be
replayed with `@reproduce_failure`). All three use `deadline=None`. An unrecognised value raises
immediately, at import time, rather than silently falling back to `"ci"`.

The two derandomized profiles also stop hypothesis from drawing values out of this repository's
own source code. Hypothesis collects the numeric and string literals of every imported module
that is not in site-packages and substitutes one of them for about one draw in twenty, so that
tests meet the constants the code under test branches on. That makes the drawn cases depend on
which modules happen to be imported and on every literal anywhere in `src/quantcore` and
`conformance/`: a derandomized run would otherwise record different cases after an unrelated
edit, or when one test module is run alone. With the collection switched off, the cases a `ci` or
`full` run draws depend only on each test function's own source text and the hypothesis version.
`explore` keeps the collection, because finding inputs the fixed profiles never draw is its job.

Everything recorded during the session is written to disk at the end of the run:
`results/raw.json` (the `classify.CaseResult`s from every `harness.check()`/
`harness.check_second_oracle()` call, which `build_report.py` classifies), `results/details.json`
(the seeds and raw values `build_report.py` uses to write findings), `results/edge_cases.json`
(every `edge_cases.observe()` observation), and `results/meta.json` (the active hypothesis
profile, so a report built from these files knows which profile produced them from the results
themselves, not from an environment variable that could have changed between the test run and
the report build).

The files go to `conformance/results/` unless `QUANTCORE_CONFORMANCE_RESULTS_DIR` names another
directory, which lets several sessions (for example one per test module) run at the same time
without overwriting each other's results.
"""

import dataclasses
import json
import os
from pathlib import Path

from hypothesis import settings
from hypothesis.internal.conjecture import providers
from hypothesis.internal.constants_ast import Constants

from conformance import edge_cases, harness

RESULTS_DIR = Path(
    os.environ.get("QUANTCORE_CONFORMANCE_RESULTS_DIR", Path(__file__).parent / "results")
)
RAW_RESULTS_PATH = RESULTS_DIR / "raw.json"
DETAILS_PATH = RESULTS_DIR / "details.json"
EDGE_CASES_PATH = RESULTS_DIR / "edge_cases.json"
META_PATH = RESULTS_DIR / "meta.json"

_KNOWN_PROFILES = ("ci", "full", "explore")

settings.register_profile("ci", max_examples=200, deadline=None, derandomize=True)
settings.register_profile("full", max_examples=5000, deadline=None, derandomize=True)
settings.register_profile(
    "explore", max_examples=5000, deadline=None, derandomize=False, print_blob=True
)

_active_profile = os.environ.get("QUANTCORE_CONFORMANCE_PROFILE", "ci")
if _active_profile not in _KNOWN_PROFILES:
    raise SystemExit(
        f"QUANTCORE_CONFORMANCE_PROFILE must be one of {_KNOWN_PROFILES}, got {_active_profile!r}"
    )
settings.load_profile(_active_profile)

_DERANDOMIZED_PROFILES = ("ci", "full")


def _ignore_local_source_constants() -> None:
    """Make hypothesis draw no values from the literals of locally imported modules.

    Replaces the function hypothesis calls to collect those literals with one that returns an
    empty collection. The function is private to hypothesis, which is why the `oracle` extra pins
    the hypothesis version exactly.

    Raises:
        SystemExit: The installed hypothesis no longer has that function, so the collection
            cannot be switched off and a derandomized run would not be reproducible.
    """
    if not callable(getattr(providers, "_get_local_constants", None)):
        raise SystemExit(
            "hypothesis.internal.conjecture.providers._get_local_constants is missing: the "
            "installed hypothesis collects source literals some other way, and the derandomized "
            "conformance profiles cannot switch that off. Install the version pinned in the "
            "'oracle' extra, or update conformance/conftest.py for the new hypothesis."
        )
    no_constants = Constants()
    providers.CONSTANTS_CACHE.cache.clear()
    providers._get_local_constants = lambda: no_constants


if _active_profile in _DERANDOMIZED_PROFILES:
    _ignore_local_source_constants()


def pytest_sessionstart() -> None:
    """Clear any results left over from a previous run before this session records new ones."""
    harness.reset()
    edge_cases.reset()


def pytest_sessionfinish() -> None:
    """Write every recorded case, repro detail, edge-case observation, and the active profile."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = [dataclasses.asdict(case) for case in harness.results()]
    RAW_RESULTS_PATH.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    detail_records = [dataclasses.asdict(detail) for detail in harness.details()]
    DETAILS_PATH.write_text(json.dumps(detail_records, indent=2, sort_keys=True), encoding="utf-8")

    edge_case_records = [dataclasses.asdict(obs) for obs in edge_cases.observations()]
    EDGE_CASES_PATH.write_text(
        json.dumps(edge_case_records, indent=2, sort_keys=True), encoding="utf-8"
    )

    META_PATH.write_text(
        json.dumps({"hypothesis_profile": _active_profile}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
