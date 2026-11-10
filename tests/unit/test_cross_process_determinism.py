"""Every public function gives the same result in separate processes.

`tests/determinism_battery.py` calls each public function on fixed inputs and prints a digest of
each result. It runs here in three fresh processes:

- A: the default BLAS and OpenMP thread counts, and an empty Numba cache, so every kernel is
  compiled;
- B: the default thread counts again, with the same Numba cache, now warm;
- C: every BLAS and OpenMP thread count set to 1, with the warm cache.

A and B must agree on every digest: for a fixed BLAS thread configuration every result is
bit-identical, whether the kernels were compiled or loaded from the cache, and whatever an
earlier process did. A and C must agree on every digest except those of
`BLAS_THREAD_SENSITIVE`, the fits `arch` computes, whose floats must agree within
`_THREAD_RTOL`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import quantcore

_BATTERY = Path(__file__).resolve().parents[1] / "determinism_battery.py"
_SINGLE_THREAD = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
}
_THREAD_VARIABLES = frozenset(_SINGLE_THREAD)

# The `arch` fits stop when SLSQP's relative change in the objective falls below its default
# `ftol` of 1e-6. Near the optimum the objective is quadratic in the parameters, so two runs
# whose objectives differ only in the last bits can stop up to sqrt(1e-6) = 1e-3 apart,
# relative. Linux aarch64 with OpenBLAS, 1 thread against 8, measured at most 1.4e-6.
_THREAD_RTOL = 1e-3


def _load_battery() -> Any:
    spec = importlib.util.spec_from_file_location("determinism_battery", _BATTERY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLAS_THREAD_SENSITIVE: frozenset[str] = _load_battery().BLAS_THREAD_SENSITIVE


def _run_battery(numba_cache: Path, extra_env: dict[str, str]) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in _THREAD_VARIABLES}
    env.update({"NUMBA_CACHE_DIR": str(numba_cache), **extra_env})
    result = subprocess.run(
        [sys.executable, str(_BATTERY)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    output: dict[str, Any] = json.loads(result.stdout)
    return output


def _differences(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))


def _values_disagree(a: dict[str, list[str]], b: dict[str, list[str]], rtol: float) -> list[str]:
    """Names whose floats differ in count, or by more than `rtol` relative in any position."""
    disagree = []
    for name in sorted(a.keys() | b.keys()):
        left = [float.fromhex(v) for v in a.get(name, [])]
        right = [float.fromhex(v) for v in b.get(name, [])]
        if len(left) != len(right) or name not in a or name not in b:
            disagree.append(name)
        elif any(abs(x - y) > rtol * max(abs(x), abs(y)) for x, y in zip(left, right, strict=True)):
            disagree.append(name)
    return disagree


def _public_functions() -> set[str]:
    names = set()
    for info in pkgutil.iter_modules(quantcore.__path__, prefix="quantcore."):
        package = importlib.import_module(info.name)
        for name in getattr(package, "__all__", ()):
            if inspect.isfunction(getattr(package, name)):
                names.add(f"{info.name.removeprefix('quantcore.')}.{name}")
    return names


@pytest.fixture(scope="module")
def battery_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    cache = tmp_path_factory.mktemp("qc23") / "numba_cache"
    return {
        "cold_default": _run_battery(cache, {}),
        "warm_default": _run_battery(cache, {}),
        "warm_single": _run_battery(cache, _SINGLE_THREAD),
    }


def test_the_battery_covers_every_public_function_and_none_raises(
    battery_runs: dict[str, dict[str, Any]],
) -> None:
    digests = battery_runs["cold_default"]["digests"]
    assert set(digests) == _public_functions()
    assert not any(value.startswith("raised") for value in digests.values())
    assert BLAS_THREAD_SENSITIVE <= set(digests)
    assert set(battery_runs["cold_default"]["values"]) == BLAS_THREAD_SENSITIVE


def test_every_public_function_is_bit_identical_for_a_fixed_thread_configuration(
    battery_runs: dict[str, dict[str, Any]],
) -> None:
    cold, warm = battery_runs["cold_default"], battery_runs["warm_default"]
    assert _differences(cold["digests"], warm["digests"]) == []


def test_across_thread_counts_only_the_arch_fits_move_and_within_tolerance(
    battery_runs: dict[str, dict[str, Any]],
) -> None:
    default, single = battery_runs["cold_default"], battery_runs["warm_single"]
    moved = set(_differences(default["digests"], single["digests"]))
    assert moved <= BLAS_THREAD_SENSITIVE
    assert _values_disagree(default["values"], single["values"], _THREAD_RTOL) == []


def test_a_call_that_draws_unseeded_entropy_is_reported(tmp_path: Path) -> None:
    planted = {"QC23_ONLY": "planted_unseeded"}
    first = _run_battery(tmp_path, planted)
    second = _run_battery(tmp_path, planted)
    assert _differences(first["digests"], second["digests"]) == ["planted_unseeded"]


def test_a_value_beyond_the_tolerance_is_reported() -> None:
    base = {"risk.fit_garch_11": [(0.05).hex(), (0.08).hex(), (0.9).hex()]}
    within = {"risk.fit_garch_11": [(0.05 * (1 + 1e-6)).hex(), (0.08).hex(), (0.9).hex()]}
    beyond = {"risk.fit_garch_11": [(0.05 * (1 + 1e-2)).hex(), (0.08).hex(), (0.9).hex()]}
    assert _values_disagree(base, within, _THREAD_RTOL) == []
    assert _values_disagree(base, beyond, _THREAD_RTOL) == ["risk.fit_garch_11"]
    assert _values_disagree(base, {}, _THREAD_RTOL) == ["risk.fit_garch_11"]
