"""The committed API manifest matches the source, and the manifest builder sees every change."""

from __future__ import annotations

import copy
import importlib.resources
import json
from typing import Any

import pytest

import quantcore
from quantcore._api_manifest import (
    MANIFEST_PATH,
    _render,
    build_manifest,
    content_sha256,
    describe_module_source,
    describe_package_source,
    main,
)


def _committed() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest


def test_committed_manifest_matches_the_source() -> None:
    assert MANIFEST_PATH.read_text(encoding="utf-8") == _render(
        build_manifest(quantcore.__version__)
    ), "run `uv run python -m quantcore._api_manifest --write` and commit the manifest"


def test_check_command_passes_on_the_committed_manifest() -> None:
    assert main(["--check"]) == 0


def test_manifest_carries_the_package_version_and_its_own_hash() -> None:
    manifest = _committed()
    assert manifest["version"] == quantcore.__version__
    assert manifest["sha256"] == content_sha256(manifest)


def test_manifest_is_shipped_with_the_package() -> None:
    shipped = importlib.resources.files("quantcore").joinpath("api_manifest.json")
    assert json.loads(shipped.read_text(encoding="utf-8"))["sha256"] == _committed()["sha256"]


def test_every_export_resolves_to_a_name_its_module_defines() -> None:
    modules = _committed()["modules"]
    for package, entry in modules.items():
        for name, origin in entry.get("exports", {}).items():
            assert name in modules[origin]["names"], (package, name, origin)


def test_every_public_function_and_class_is_listed() -> None:
    modules = _committed()["modules"]
    for package_name in (m for m in modules if "exports" in m and m != "quantcore"):
        package = importlib.import_module(package_name)
        assert set(modules[package_name]["exports"]) == set(package.__all__)


def test_any_change_to_the_content_changes_the_hash() -> None:
    manifest = _committed()
    changed = copy.deepcopy(manifest)
    changed["modules"]["quantcore.statistics.multiple_testing"]["names"]["cscv_pbo"][
        "signature"
    ] += " "
    assert content_sha256(changed) != content_sha256(manifest)


_SOURCE = """
from dataclasses import dataclass
import numpy as np

Method = Literal["a", "b"]


@dataclass(frozen=True)
class Result:
    value: float
    n: int = 3


class FitError(RuntimeError):
    def __init__(self, message: str, *, params: tuple[float, ...]) -> None:
        super().__init__(message)

    def __reduce__(self):
        return (type(self), self.args)

    def _private(self) -> None:
        pass


def fit(series: np.ndarray, n_lags: int = 1, *, method: Method = "a") -> Result:
    return Result(0.0)


def _helper() -> None:
    pass
"""


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("n_lags: int = 1", "lags: int = 1"),
        ("n_lags: int = 1", "n_lags: int = 2"),
        ("n_lags: int = 1", "n_lags: float = 1"),
        (", *, method", ", method"),
        ("-> Result:", "-> float:"),
        ("    n: int = 3", "    n: float = 3"),
        ('Literal["a", "b"]', 'Literal["a", "c"]'),
        ("params: tuple[float, ...]", "params: tuple[float, ...] = ()"),
        ("class FitError(RuntimeError)", "class FitError(ValueError)"),
        ("@dataclass(frozen=True)", "@dataclass"),
    ],
)
def test_a_change_to_the_public_surface_changes_the_description(old: str, new: str) -> None:
    assert _SOURCE.count(old) == 1
    assert describe_module_source(_SOURCE.replace(old, new)) != describe_module_source(_SOURCE)


def test_private_names_and_bodies_do_not_enter_the_description() -> None:
    names = describe_module_source(_SOURCE)["names"]
    assert set(names) == {"Method", "Result", "FitError", "fit"}
    assert set(names["FitError"]["methods"]) == {"__init__", "__reduce__"}
    changed_body = _SOURCE.replace("return Result(0.0)", "return Result(1.0)")
    assert describe_module_source(changed_body) == describe_module_source(_SOURCE)


def test_package_exports_name_their_module() -> None:
    source = 'from quantcore.x.a import f, g as h\nfrom .b import K\n__all__ = ["f", "h", "K"]\n'
    assert describe_package_source(source, "quantcore.x") == {
        "exports": {"f": "quantcore.x.a", "h": "quantcore.x.a", "K": "quantcore.x.b"}
    }


def test_an_export_that_is_not_imported_from_a_module_is_refused() -> None:
    source = 'from quantcore.x.a import f\nTHRESHOLD = 1.0\n__all__ = ["f", "THRESHOLD"]\n'
    with pytest.raises(ValueError, match="THRESHOLD"):
        describe_package_source(source, "quantcore.x")


def test_check_command_refuses_unknown_arguments() -> None:
    assert main([]) == 2
    assert main(["--chek"]) == 2


@pytest.mark.parametrize(
    "path",
    sorted((MANIFEST_PATH.parents[2] / "docs" / "api_manifests").glob("*.json")),
    ids=lambda p: p.name,
)
def test_each_manifest_of_an_earlier_release_matches_its_hash(path: Any) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == content_sha256(manifest)
    assert path.name == f"quantcore-{manifest['version']}-{path.stem.rsplit('-', 1)[1]}.json"
