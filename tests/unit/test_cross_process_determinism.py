"""Every public function gives bit-identical results in separate processes.

`tests/determinism_battery.py` calls each public function on fixed inputs and prints a digest of
each result. It runs here in two fresh processes:

- A: the default BLAS and OpenMP thread counts, and an empty Numba cache, so every kernel is
  compiled;
- B: every BLAS and OpenMP thread count set to 1, and the same Numba cache, now warm.

The two must agree on every digest. A result that depended on the thread count, on compiled
against cached machine code, or on state left by an earlier call in the process would differ.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

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


def _run_battery(numba_cache: Path, extra_env: dict[str, str]) -> dict[str, str]:
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
    digests: dict[str, str] = json.loads(result.stdout)
    return digests


def _differences(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))


def _public_functions() -> set[str]:
    names = set()
    for info in pkgutil.iter_modules(quantcore.__path__, prefix="quantcore."):
        package = importlib.import_module(info.name)
        for name in getattr(package, "__all__", ()):
            if inspect.isfunction(getattr(package, name)):
                names.add(f"{info.name.removeprefix('quantcore.')}.{name}")
    return names


def test_every_public_function_is_bit_identical_across_processes(tmp_path: Path) -> None:
    cache = tmp_path / "numba_cache"
    cold_default_threads = _run_battery(cache, {})
    warm_single_thread = _run_battery(cache, _SINGLE_THREAD)
    assert set(cold_default_threads) == _public_functions()
    assert not any(value.startswith("raised") for value in cold_default_threads.values())
    assert _differences(cold_default_threads, warm_single_thread) == []


def test_a_call_that_draws_unseeded_entropy_is_reported(tmp_path: Path) -> None:
    planted = {"QC23_ONLY": "planted_unseeded"}
    first = _run_battery(tmp_path, planted)
    second = _run_battery(tmp_path, {**planted, **_SINGLE_THREAD})
    assert _differences(first, second) == ["planted_unseeded"]
