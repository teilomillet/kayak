"""Keep the source documentation navigable from checkouts and source archives."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def heading_anchors(document: Path) -> set[str]:
    """Resolve the ATX headings used by these guides, including duplicate titles."""
    anchors: set[str] = set()
    fenced = False
    for line in document.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        if fenced:
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if heading is None:
            continue
        title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading[1])
        base = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchor = base
        suffix = 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{base}-{suffix}"
        anchors.add(anchor)
    return anchors


def test_documentation_links_and_example_commands_resolve() -> None:
    documents = [*ROOT.glob("*.md"), ROOT / "llms.txt"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    documents.extend((ROOT / "examples").rglob("*.md"))
    for document in documents:
        for module in re.findall(r"\bexamples\.([a-zA-Z_]\w*)", document.read_text()):
            assert (ROOT / "examples" / f"{module}.py").is_file(), (document, module)
        for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", document.read_text()):
            target = urlsplit(link)
            if target.scheme or target.netloc:
                continue
            path = document.parent / unquote(target.path) if target.path else document
            assert path.exists(), f"{document.name}: {link}"
            if target.fragment and path.suffix == ".md":
                assert unquote(target.fragment) in heading_anchors(path), f"{document.name}: {link}"


def test_every_runnable_example_is_linked_from_the_task_catalog() -> None:
    examples = ROOT / "examples"
    catalog = (examples / "README.md").read_text(encoding="utf-8")
    links = {urlsplit(link).path for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", catalog)}
    recipes = set(examples.glob("*.py")) - {examples / "__init__.py"}
    recipes.update(examples.glob("*.json"))
    for recipe in recipes:
        assert recipe.relative_to(examples).as_posix() in links, f"Recipe is unlisted: {recipe}"
