# The API manifest

`src/quantcore/api_manifest.json` lists quantcore's public API:
- every public module;
- in each module, every public function with its signature, every public class with its bases,
  decorators, fields and public methods, and every public module-level value such as a type
  alias;
- in each package, the names `__all__` exports and the module each one comes from.

It carries the package version and `sha256`. That hash is the SHA-256 of the manifest without
the `sha256` field, serialized as canonical JSON: sorted keys, separators `,` and `:`, UTF-8,
`ensure_ascii=False`. The manifest is built from the source with `ast`, so it is the same on
every platform. It ships inside the package (`importlib.resources.files("quantcore") /
"api_manifest.json"`).

A test fails when the committed manifest differs from the source. Regenerate it with
`uv run python -m quantcore._api_manifest --write`. A change to the manifest is a change to the
contract: it goes in `CHANGELOG.md`, and the consumers re-vendor at their next pin.

## For a consumer

1. Vendor the manifest of the quantcore commit you pin. From a commit that ships it, copy
   `src/quantcore/api_manifest.json`. For the two releases before it, use the files in
   `docs/api_manifests/`, built from those commits with the same builder:
   - `quantcore-0.3.1-599b8bd.json`;
   - `quantcore-0.3.2-59cfb2b.json`.

   The two differ only in `ConvergenceError.__reduce__`, which 0.3.2 adds.
2. In CI, with no quantcore checkout present, test that:
   - the vendored file's `sha256` matches its content;
   - every quantcore name your source and tests use resolves in the vendored manifest:
     `from quantcore.x import name`, and `name.attr` after `import quantcore.x` or
     `import quantcore.x as name`;
   - once your pin ships the manifest, the installed package's manifest has the vendored
     `sha256`. A re-pin without re-vendoring then fails.

A reference check:

```python
import ast, hashlib, json
from pathlib import Path


def manifest_sha256(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolves(modules: dict, module: str, name: str) -> bool:
    entry = modules.get(module, {})
    if name in entry.get("names", {}):
        return True
    origin = entry.get("exports", {}).get(name)
    if origin is not None:
        return name in modules[origin]["names"]
    return f"{module}.{name}" in modules  # a submodule


def _dotted(node: ast.expr) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def unresolved_quantcore_uses(manifest: dict, files: list[Path]) -> list[str]:
    """Every quantcore import or attribute use in `files` that the manifest does not list."""
    modules = manifest["modules"]
    found = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quantcore"):
                for alias in node.names:
                    if not _resolves(modules, node.module, alias.name):
                        found.append(f"{path}:{node.lineno} {node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("quantcore"):
                        if alias.name not in modules:
                            found.append(f"{path}:{node.lineno} {alias.name}")
                        # `import quantcore.x` binds the dotted path; `as y` binds `y`.
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            dotted = _dotted(node) if isinstance(node, ast.Attribute) else None
            prefixes = [a for a in aliases if dotted and dotted.startswith(a + ".")]
            if not prefixes:
                continue
            prefix = max(prefixes, key=len)
            name = dotted[len(prefix) + 1 :].split(".")[0]
            if not _resolves(modules, aliases[prefix], name):
                found.append(f"{path}:{node.lineno} {aliases[prefix]}.{name}")
    return sorted(set(found))
```

Give the check a negative control: a planted file that imports a name the manifest does not list
must be reported.
