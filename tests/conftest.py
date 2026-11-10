"""Fail the run when a test is skipped because a library could not be imported.

The oracle tests and the conformance builder tests import reference libraries from the `oracle`
extra through `pytest.importorskip`. Without this hook a missing or broken library would turn
their coverage into skips while the run still passed. Every such skip is listed at the end of the
run and the exit status is "tests failed". Without the oracle extra, run
`uv run pytest tests/unit --ignore=tests/unit/conformance`, where only the few tests that call a
reference library in their body fail.
"""

from __future__ import annotations

import pytest

_IMPORT_SKIP_MARK = "could not import"
_import_skips: list[str] = []


def _skip_reason(report: pytest.CollectReport | pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def _record_import_skip(report: pytest.CollectReport | pytest.TestReport) -> None:
    if report.skipped and _IMPORT_SKIP_MARK in _skip_reason(report):
        _import_skips.append(f"{report.nodeid}: {_skip_reason(report)}")


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _record_import_skip(report)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record_import_skip(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _import_skips and exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _import_skips:
        terminalreporter.section("skipped for a missing library (fails the run)", red=True)
        for line in _import_skips:
            terminalreporter.line(line)
