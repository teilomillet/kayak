from __future__ import annotations

"""Execute the main internal course notebook without notebook dependencies.

This test exists because the current environment does not ship `nbclient` or
Jupyter. The integrated course notebook is plain Python, so we can still keep
it honest by executing its code cells directly from the notebook JSON.
"""

import json
import os
import unittest
import warnings
from contextlib import contextmanager
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterator


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class CourseNotebookExecutionTests(unittest.TestCase):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    NOTEBOOK_DIR = Path("docs/internals/course/notebooks")
    DEBUG_SEQUENCE_NOTEBOOK = NOTEBOOK_DIR / "debug_your_broken_rag_with_kayak.ipynb"
    RAG_DEBUG_NOTEBOOK = NOTEBOOK_DIR / "rag_retrieval_debugging_with_kayak.ipynb"
    FAILURE_PATTERN_NOTEBOOK = (
        NOTEBOOK_DIR / "retrieval_failure_pattern_catalog.ipynb"
    )
    REAL_SLICE_NOTEBOOK = NOTEBOOK_DIR / "real_slice_proxy_diagnosis.ipynb"
    CHUNKING_NOTEBOOK = NOTEBOOK_DIR / "classic_chunking_and_one_vector.ipynb"
    CHUNK_COVERAGE_NOTEBOOK = (
        NOTEBOOK_DIR / "chunk_coverage_vs_joint_scoring.ipynb"
    )
    LIMIT_SMALL_TASK_PATH = Path(".cache/kayak/limit_small_real_subset/python_task.json")
    BRIGHT_STACKOVERFLOW_TASK_PATH = Path(
        ".cache/kayak/bright_stackoverflow_real_subset/python_task.json"
    )
    LEGAL_RAG_BENCH_TASK_PATH = Path(
        ".cache/kayak/legal_rag_bench_real_subset/python_task.json"
    )
    R2MED_BIOLOGY_TASK_PATH = Path(
        ".cache/kayak/r2med_biology_real_subset/python_task.json"
    )

    def _load_notebook(self, notebook_path: Path) -> dict[str, object]:
        return json.loads(notebook_path.read_text())

    def _code_cells(self, notebook_path: Path) -> list[tuple[int, str]]:
        notebook = self._load_notebook(notebook_path)
        cells: list[tuple[int, str]] = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            cells.append((index, source))
        return cells

    def _require_paths(self, paths: list[Path]) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            self.skipTest(
                "required cached task JSONs are missing: " + ", ".join(missing)
            )

    def _execute_notebook(self, notebook_path: Path) -> None:
        with pushd(self.REPO_ROOT):
            namespace: dict[str, object] = {"__name__": "__main__"}
            captured_stdout = StringIO()
            with redirect_stdout(captured_stdout):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="`torch.jit.script` is deprecated.*",
                        category=DeprecationWarning,
                    )
                    for index, source in self._code_cells(notebook_path):
                        code = compile(
                            source,
                            f"{notebook_path}#cell{index}",
                            "exec",
                        )
                        try:
                            exec(code, namespace)
                        except Exception as exc:  # pragma: no cover - failure path only
                            self.fail(
                                f"{notebook_path} failed in code cell {index}: "
                                f"{type(exc).__name__}: {exc}"
                            )

    def test_rag_debugging_notebook_executes_sequentially(self) -> None:
        self._execute_notebook(self.RAG_DEBUG_NOTEBOOK)

    def test_failure_pattern_notebook_executes_sequentially(self) -> None:
        self._execute_notebook(self.FAILURE_PATTERN_NOTEBOOK)

    def test_main_course_notebook_executes_sequentially(self) -> None:
        self._require_paths([self.LIMIT_SMALL_TASK_PATH])
        self._execute_notebook(self.DEBUG_SEQUENCE_NOTEBOOK)

    def test_real_slice_notebook_executes_sequentially(self) -> None:
        self._require_paths(
            [
                self.LIMIT_SMALL_TASK_PATH,
                self.BRIGHT_STACKOVERFLOW_TASK_PATH,
                self.LEGAL_RAG_BENCH_TASK_PATH,
                self.R2MED_BIOLOGY_TASK_PATH,
            ]
        )
        self._execute_notebook(self.REAL_SLICE_NOTEBOOK)

    def test_chunking_baseline_notebook_executes_sequentially(self) -> None:
        self._require_paths(
            [
                self.LIMIT_SMALL_TASK_PATH,
                self.BRIGHT_STACKOVERFLOW_TASK_PATH,
                self.LEGAL_RAG_BENCH_TASK_PATH,
                self.R2MED_BIOLOGY_TASK_PATH,
            ]
        )
        self._execute_notebook(self.CHUNKING_NOTEBOOK)

    def test_chunk_coverage_notebook_executes_sequentially(self) -> None:
        self._execute_notebook(self.CHUNK_COVERAGE_NOTEBOOK)


if __name__ == "__main__":
    unittest.main()
