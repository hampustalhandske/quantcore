"""Test for the public `quantcore.__version__` attribute."""

from __future__ import annotations

import re

import quantcore


def test_version_is_a_string_matching_semver_shape() -> None:
    assert isinstance(quantcore.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", quantcore.__version__)


def test_version_matches_pyproject_toml() -> None:
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    assert quantcore.__version__ == pyproject["project"]["version"]
