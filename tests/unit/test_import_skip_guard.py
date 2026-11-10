"""The run fails when a test is skipped because a library could not be imported (tests/conftest.py).

Each case runs pytest in a fresh process on a small planted test tree that carries a copy of the
repository's `tests/conftest.py`, and checks the exit status.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

_MODULE_LEVEL_IMPORT_SKIP = (
    'import pytest\npytest.importorskip("no_such_module_qc33")\n\ndef test_x():\n    pass\n'
)
_IN_TEST_IMPORT_SKIP = (
    'import pytest\n\ndef test_x():\n    pytest.importorskip("no_such_module_qc33")\n'
)
_OTHER_SKIP = (
    'import pytest\n\n@pytest.mark.skip(reason="not on this platform")\ndef test_x():\n    pass\n'
)
_PRESENT_LIBRARY = 'import pytest\npytest.importorskip("json")\n\ndef test_x():\n    pass\n'


def _run(tmp_path: Path, test_source: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "conftest.py").write_text(_CONFTEST.read_text())
    (tmp_path / "test_planted.py").write_text(test_source)
    (tmp_path / "test_other.py").write_text("def test_ok():\n    pass\n")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "source", [_MODULE_LEVEL_IMPORT_SKIP, _IN_TEST_IMPORT_SKIP], ids=["module level", "in a test"]
)
def test_a_skip_for_a_missing_library_fails_the_run(tmp_path: Path, source: str) -> None:
    result = _run(tmp_path, source)
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout
    assert "skipped for a missing library" in result.stdout
    assert "no_such_module_qc33" in result.stdout


@pytest.mark.parametrize("source", [_OTHER_SKIP, _PRESENT_LIBRARY], ids=["other skip", "present"])
def test_other_skips_and_present_libraries_pass(tmp_path: Path, source: str) -> None:
    result = _run(tmp_path, source)
    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert "skipped for a missing library" not in result.stdout


def test_a_run_whose_only_module_is_skipped_fails(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(_CONFTEST.read_text())
    (tmp_path / "test_planted.py").write_text(_MODULE_LEVEL_IMPORT_SKIP)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout


def test_a_failing_run_keeps_its_own_status(tmp_path: Path) -> None:
    (tmp_path / "test_fails.py").write_text("def test_z():\n    assert False\n")
    result = _run(tmp_path, _MODULE_LEVEL_IMPORT_SKIP)
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout
    assert "skipped for a missing library" in result.stdout
