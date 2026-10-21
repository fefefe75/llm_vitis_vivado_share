"""Read files from disk into ``Document`` objects.

Two entry points:

* ``repo_documents``: the knowledge of this repository (docs, skills, prompts,
  templates, examples). ``training/`` is deliberately left out: it holds the
  evaluation bench and its answers, and indexing them would make any score
  meaningless.
* ``folder_documents``: any folder or file the user wants to add (datasheets,
  notes, another project). It stays on the user's machine, in the index only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .chunking import Document, kind_for

REPO_DIRS = ("docs", "skills", "prompts", "templates", "examples")
REPO_FILES = ("README.md", "AGENTS.md", "mcp/README.md")

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "sim_build", ".Xil",
    ".runs", ".sim", ".cache", "xsim.dir", ".mcp_runs", "_dbg", ".pytest_cache",
}
# recorded terminal sessions and their generator: no knowledge in them
REPO_SKIP = ("docs/demo/",)
MAX_TEXT_BYTES = 1_000_000
MAX_PDF_BYTES = 80_000_000


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _read_text(path: Path) -> str | None:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        return None                      # binary file
    return raw.decode("utf-8", errors="replace")


def _pdf_sections(path: Path) -> list[tuple[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("lecture des PDF : pip install -e 'rag[pdf]'") from exc
    reader = PdfReader(str(path))
    sections = []
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append((f"p. {number}", text))
    return sections


def load_file(path: Path, source: str, key: str, location: str) -> Document | None:
    """One file as a Document, or None if it is not something to index."""
    kind = kind_for(path.name, path.suffix)
    if kind is None:
        return None
    size = path.stat().st_size
    if size == 0 or size > (MAX_PDF_BYTES if kind == "pages" else MAX_TEXT_BYTES):
        return None

    if kind == "pages":
        sections = _pdf_sections(path)
        if not sections:
            return None
        return Document(source, key, path.stem, location,
                        "\n\n".join(t for _, t in sections), "pages", sections)

    text = _read_text(path)
    if not text or not text.strip():
        return None
    if kind == "html":
        from .amd import html_to_text

        text = html_to_text(text)
        kind = "markdown"
    title = (_first_heading(text) if kind == "markdown" else None) or key
    return Document(source, key, title, location, text, kind)


def _walk(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts[:-1]):
            continue
        if path.is_file() and not path.name.startswith("."):
            yield path


def _has_english_twin(path: Path) -> bool:
    """`x.fr.md` duplicates `x.md`: index the English one only."""
    name = path.name
    if ".fr." not in name:
        return False
    return (path.parent / name.replace(".fr.", ".")).exists()


def repo_documents(repo: Path) -> list[Document]:
    """The repository's knowledge files."""
    repo = repo.resolve()
    paths: list[Path] = [repo / f for f in REPO_FILES if (repo / f).is_file()]
    for folder in REPO_DIRS:
        if (repo / folder).is_dir():
            paths += list(_walk(repo / folder))
    docs = []
    for path in paths:
        if _has_english_twin(path):
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.startswith(REPO_SKIP):
            continue
        doc = load_file(path, "repo", rel, rel)
        if doc:
            docs.append(doc)
    return docs


def folder_documents(target: Path, source: str) -> list[Document]:
    """Every indexable file under ``target`` (or ``target`` itself)."""
    target = target.expanduser().resolve()
    if target.is_file():
        files, base = [target], target.parent
    else:
        files, base = list(_walk(target)), target
    docs = []
    for path in files:
        rel = path.relative_to(base).as_posix()
        doc = load_file(path, source, rel, f"{base.name}/{rel}")
        if doc:
            docs.append(doc)
    return docs
