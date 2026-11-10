"""The public API of quantcore as a versioned, hashed JSON manifest.

The manifest lists every public module, and in each one every public name:
- a function, with its signature;
- a class, with its bases, decorators, annotated fields and public methods (plus `__init__` and
  `__reduce__`);
- a module-level value such as a type alias, with its source expression.

A package lists the names its `__all__` exports and the module each one comes from. Everything
is read from the source with `ast`, so the manifest is the same on every platform and under every
NumPy version.

`sha256` is the SHA-256 of the manifest without that field, serialized as canonical JSON
(sorted keys, no whitespace, UTF-8). Consumers vendor the manifest of the quantcore commit they
pin and test their imports against it (`docs/API_MANIFEST.md`).

Usage: `python -m quantcore._api_manifest --check` exits 1 when the committed
`api_manifest.json` differs from the source; `--write` rewrites it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "quantcore-api-manifest/1"
PACKAGE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_ROOT / "api_manifest.json"
_KEPT_DUNDER_METHODS = ("__init__", "__reduce__")


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({ast.unparse(node.args)}){returns}"


def _describe_class(node: ast.ClassDef) -> dict[str, Any]:
    fields: dict[str, str] = {}
    methods: dict[str, str] = {}
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            default = f" = {ast.unparse(item.value)}" if item.value is not None else ""
            fields[item.target.id] = f"{ast.unparse(item.annotation)}{default}"
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            _is_public(item.name) or item.name in _KEPT_DUNDER_METHODS
        ):
            methods[item.name] = _signature(item)
    return {
        "kind": "class",
        "bases": [ast.unparse(base) for base in node.bases],
        "decorators": [ast.unparse(d) for d in node.decorator_list],
        "fields": fields,
        "methods": methods,
    }


def describe_module_source(source: str) -> dict[str, Any]:
    """The public functions, classes and values a module's source defines at top level."""
    names: dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            names[node.name] = {"kind": "function", "signature": _signature(node)}
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            names[node.name] = _describe_class(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and _is_public(target.id):
                    names[target.id] = {"kind": "value", "value": ast.unparse(node.value)}
    return {"names": names}


def describe_package_source(source: str, package: str) -> dict[str, Any]:
    """The names a package's `__init__` lists in `__all__`, each with the module it comes from.

    The root package `quantcore` also lists `__version__`, a string it reads from the installed
    distribution's metadata; it is recorded under `names` as a value of kind `str`, beside the
    imported names under `exports`.

    Raises:
        ValueError: A name in `__all__` is not imported by a `from ... import` in the source.
    """
    tree = ast.parse(source)
    origin: dict[str, str] = {}
    exported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            module = node.module if node.level == 0 else f"{package}.{node.module}"
            for alias in node.names:
                origin[alias.asname or alias.name] = module
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = list(ast.literal_eval(node.value))
    values: dict[str, Any] = {}
    if package == "quantcore" and "__version__" in exported:
        exported.remove("__version__")
        values["__version__"] = {"kind": "value", "value": "str"}
    missing = sorted(set(exported) - set(origin))
    if missing:
        raise ValueError(f"{package}: __all__ names not imported from a module: {missing}")
    description: dict[str, Any] = {"exports": {name: origin[name] for name in sorted(exported)}}
    if values:
        description["names"] = values
    return description


def _module_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if all(_is_public(part) for part in parts):
            files[".".join(parts)] = path
    return files


def content_sha256(manifest: dict[str, Any]) -> str:
    """SHA-256 of the manifest without its `sha256` field, as canonical JSON."""
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(version: str) -> dict[str, Any]:
    """The manifest of the source next to this file, for the given package version."""
    modules: dict[str, Any] = {}
    for name, path in _module_files().items():
        source = path.read_text(encoding="utf-8")
        if path.name == "__init__.py":
            modules[name] = describe_package_source(source, name)
        else:
            modules[name] = describe_module_source(source)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "package": "quantcore",
        "version": version,
        "modules": modules,
    }
    manifest["sha256"] = content_sha256(manifest)
    return manifest


def _render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    from quantcore import __version__

    if argv not in (["--check"], ["--write"]):
        print("usage: python -m quantcore._api_manifest --check | --write", file=sys.stderr)
        return 2
    rendered = _render(build_manifest(__version__))
    if argv == ["--write"]:
        MANIFEST_PATH.write_text(rendered, encoding="utf-8")
        return 0
    committed = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else ""
    if committed != rendered:
        print(f"{MANIFEST_PATH.name} differs from the source; run with --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
