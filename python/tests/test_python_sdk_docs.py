from __future__ import annotations

import tomllib
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


class PythonSdkDocsContractTests(unittest.TestCase):
    def test_project_metadata_uses_package_scoped_readme(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["readme"], "python/kayak/README.md")

    def test_python_sdk_docs_exist_and_are_linked(self) -> None:
        root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        package_readme = (REPO_ROOT / "python" / "kayak" / "README.md").read_text(
            encoding="utf-8"
        )
        sdk_doc = (REPO_ROOT / "docs" / "python_sdk.md").read_text(encoding="utf-8")
        charter_doc = (
            REPO_ROOT / "docs" / "python_sdk_charter.md"
        ).read_text(encoding="utf-8")
        roadmap_doc = (
            REPO_ROOT / "docs" / "python_sdk_roadmap.md"
        ).read_text(encoding="utf-8")

        self.assertIn("docs/python_sdk_charter.md", root_readme)
        self.assertIn("docs/python_sdk_roadmap.md", root_readme)
        self.assertIn("docs/python_sdk_charter.md", package_readme)
        self.assertIn("docs/python_sdk_roadmap.md", package_readme)
        self.assertIn("kayak.typing", package_readme)
        self.assertIn('kayak.help("typing")', package_readme)
        self.assertIn('kayak.help("TokenMatrixInput")', package_readme)
        self.assertIn("python_sdk_charter.md", sdk_doc)
        self.assertIn("python_sdk_roadmap.md", sdk_doc)

        self.assertIn("canonical open Python programming model", charter_doc)
        self.assertIn("Kayak Engine", charter_doc)
        self.assertIn("Build `kayak` as the local Python late-interaction SDK first", roadmap_doc)
        self.assertIn("Acceptance Checklist", roadmap_doc)


if __name__ == "__main__":
    unittest.main()
