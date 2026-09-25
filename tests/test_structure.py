"""Keep source files bounded and explicit package dependencies acyclic."""

import ast
from graphlib import CycleError, TopologicalSorter
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("kayak", "tests", "scripts", "benchmarks", "examples", "stubs")
MAX_SOURCE_LINES = 1000


def test_python_files_stay_within_the_line_limit() -> None:
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.pyi"))
    for directory in SOURCE_DIRECTORIES:
        paths.extend(
            path for path in (ROOT / directory).rglob("*") if path.suffix in {".py", ".pyi"}
        )
    violations: list[str] = []
    for path in sorted(paths):
        # Count physical lines, including comments, blanks, and the last line
        # without a newline. Readability limits must not discourage documentation.
        with path.open(encoding="utf-8") as source:
            lines = sum(1 for _ in source)
        if lines > MAX_SOURCE_LINES:
            violations.append(f"{path.relative_to(ROOT)}: {lines} lines")
    assert not violations, (
        f"Source files must have at most {MAX_SOURCE_LINES} lines; split by responsibility:\n"
        + "\n".join(violations)
    )


def test_kayak_imports_are_acyclic() -> None:
    modules: dict[str, Path] = {}
    for path in sorted((ROOT / "kayak").rglob("*.py")):
        name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        modules[name.removesuffix(".__init__")] = path

    dependencies: dict[str, set[str]] = {}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        imports: set[str] = set()
        source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Count explicit imports in every scope, including optional imports and
        # TYPE_CHECKING blocks. Moving an import must not conceal coupling.
        for node in ast.walk(source):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = resolve_name("." * node.level + (node.module or ""), package)
                for alias in node.names:
                    # `from . import decisions` imports a module; importing a
                    # symbol such as Choice depends on its containing module.
                    candidate = f"{imported}.{alias.name}"
                    imports.add(candidate if candidate in modules else imported)
        dependencies[name] = imports & modules.keys()

    try:
        tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc:
        pytest.fail(f"Kayak imports must be acyclic: {exc}")
