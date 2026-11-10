"""Census of module-level state: quantcore keeps no mutable object at module level.

A module-level dict, list, set, writable array or other mutable instance is state that outlives a
call. It can make a result depend on what ran before it in the process, and a consumer's
checkpoint does not capture it. Every value a quantcore module assigns at module level must be
immutable, or be listed in `ALLOWED` with the reason it is safe, and no function may rebind a
module-level name (`global` or `nonlocal`). Imported names, functions and classes are not state
of the module and are not checked.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import pkgutil
import types
from typing import Any

import numpy as np
import pytest

import quantcore

# (module, name) -> why the mutable object is safe.
ALLOWED: dict[tuple[str, str], str] = {}

# `__all__` only lists the names a package exports.
_NOT_STATE = frozenset({"__all__"})

_IMMUTABLE_SCALARS = (
    int,
    float,
    complex,
    str,
    bytes,
    bool,
    type(None),
    np.generic,
    range,
    pathlib.PurePath,
)


def _is_immutable(value: Any) -> bool:
    if isinstance(value, _IMMUTABLE_SCALARS):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_immutable(item) for item in value)
    if isinstance(value, types.MappingProxyType):
        return all(_is_immutable(k) and _is_immutable(v) for k, v in value.items())
    if isinstance(value, np.ndarray):
        return not value.flags.writeable
    # Type aliases and other typing constructs.
    return type(value).__module__ in ("typing", "typing_extensions")


def mutable_module_state(module: types.ModuleType, names: set[str]) -> list[str]:
    """The values `module` binds to `names` that are mutable and not in `ALLOWED`."""
    found = []
    for name in sorted(names - _NOT_STATE):
        if (module.__name__, name) in ALLOWED:
            continue
        value = getattr(module, name)
        if not _is_immutable(value):
            found.append(f"{module.__name__}.{name}: {type(value).__name__}")
    return found


def _assigned_names(tree: ast.Module) -> set[str]:
    """Names bound by an assignment statement at the top level of a module's source."""
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def _rebinding_statements(tree: ast.Module) -> list[int]:
    """Line numbers of `global` and `nonlocal` statements."""
    return [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Global, ast.Nonlocal))]


def _source_tree(module: types.ModuleType) -> ast.Module:
    return ast.parse(inspect.getsource(module))


def _quantcore_modules() -> list[types.ModuleType]:
    names = [quantcore.__name__] + [
        info.name for info in pkgutil.walk_packages(quantcore.__path__, prefix="quantcore.")
    ]
    return [importlib.import_module(name) for name in sorted(names)]


def test_every_quantcore_module_was_walked() -> None:
    walked = {module.__name__ for module in _quantcore_modules()}
    assert {"quantcore.risk.volatility", "quantcore.statistics.diagnostics"} <= walked
    assert len(walked) >= 38


def test_no_quantcore_module_holds_mutable_state() -> None:
    found = [
        entry
        for module in _quantcore_modules()
        for entry in mutable_module_state(module, _assigned_names(_source_tree(module)))
    ]
    assert found == []


def test_no_quantcore_function_rebinds_a_module_level_name() -> None:
    found = {
        module.__name__: lines
        for module in _quantcore_modules()
        if (lines := _rebinding_statements(_source_tree(module)))
    }
    assert found == {}


def test_every_allowed_entry_still_exists() -> None:
    for module_name, name in ALLOWED:
        assert hasattr(importlib.import_module(module_name), name), (module_name, name)


@pytest.mark.parametrize(
    "planted",
    [{}, [], set(), np.zeros(3), bytearray(b"x"), (1, []), types.SimpleNamespace(a=1)],
    ids=["dict", "list", "set", "writable array", "bytearray", "tuple holding a list", "instance"],
)
def test_census_reports_a_planted_mutable_object(planted: object) -> None:
    module = types.ModuleType("quantcore_planted")
    module._state = planted  # type: ignore[attr-defined]
    assert mutable_module_state(module, {"_state"}) == [
        f"quantcore_planted._state: {type(planted).__name__}"
    ]


def test_census_finds_the_names_a_module_assigns_and_its_rebinding_statements() -> None:
    tree = ast.parse(
        "import math\n"
        "from x import y\n"
        "_CACHE: dict[str, int] = {}\n"
        "_A, (_B, _C) = 1, (2, 3)\n"
        "_N = 0\n"
        "_N += 1\n"
        "def f():\n"
        "    global _N\n"
        "    _N = 2\n"
        "class K:\n"
        "    inner = []\n"
    )
    assert _assigned_names(tree) == {"_CACHE", "_A", "_B", "_C", "_N"}
    assert _rebinding_statements(tree) == [8]


def test_census_accepts_immutable_values() -> None:
    module = types.ModuleType("quantcore_planted")
    frozen = np.zeros(3)
    frozen.flags.writeable = False
    values = {
        "a": 1,
        "b": 1e-12,
        "c": ("c", "ct"),
        "d": types.MappingProxyType({"c": 1}),
        "e": frozen,
        "f": frozenset({1}),
        "g": np.float64(2.0),
    }
    for name, value in values.items():
        setattr(module, name, value)
    assert mutable_module_state(module, set(values)) == []
