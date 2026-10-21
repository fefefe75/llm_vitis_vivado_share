"""Cut documents into chunks that each talk about one thing.

Rules, in order:
1. Follow the author's structure: Markdown is cut at its headings (never inside
   a fenced code block, where ``# comment`` is not a heading), a PDF at its
   pages, an AMD topic is already one subject.
2. Merge a section that is too short with the next one of the same document,
   so that a lone heading or a two-line note does not become a chunk.
3. Split a section that is too long at paragraph boundaries, with an overlap,
   never in the middle of a line (code stays readable).

Each chunk keeps its breadcrumb (``document > section > subsection``): it is
embedded with the text, so the vector knows what the chunk is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import MAX_WORDS, MIN_WORDS, OVERLAP_WORDS

MARKDOWN_EXT = {".md", ".markdown", ".rst", ".txt"}
CODE_EXT = {
    ".vhd", ".vhdl", ".v", ".sv", ".svh", ".tcl", ".xdc", ".sdc", ".py", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".sh", ".mk", ".cmake", ".yaml", ".yml", ".toml", ".json",
}
PDF_EXT = {".pdf"}
HTML_EXT = {".html", ".htm"}
MAKEFILE_NAMES = {"Makefile", "makefile", "GNUmakefile"}

# Bump when the cutting rules change: it is part of each document's fingerprint,
# so the next indexing re-cuts and re-embeds everything instead of keeping
# chunks made by the old rules.
CHUNKER_VERSION = "2"

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Document:
    """One unit of the corpus: a file, a PDF, an AMD topic."""

    source: str          # "repo", "amd:UG901", "local:datasheets"...
    key: str             # unique inside the source (relative path, topic id)
    title: str           # shown to the user and to the model
    location: str        # relative path or URL, used for citations
    text: str
    kind: str = "markdown"   # markdown | code | text | pages
    sections: list[tuple[str, str]] | None = None   # pre-cut (heading, text)


@dataclass
class Chunk:
    heading: str
    text: str
    ord: int = 0
    context: str = field(default="")

    def embedding_text(self) -> str:
        return f"{self.context}\n\n{self.text}" if self.context else self.text


def _words(text: str) -> int:
    return len(text.split())


def split_markdown(text: str) -> list[tuple[str, str]]:
    """[(breadcrumb, body)] cut at headings of level 1 to 4, fences respected."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((" > ".join(path), body))

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            current = []
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
        current.append(line)
    flush()
    return sections


def _paragraphs(text: str) -> list[str]:
    """Blocks separated by blank lines; a fenced code block stays one block."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_long_block(block: str, max_words: int) -> list[str]:
    """A single block above the limit is cut at line boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for line in block.splitlines():
        n = _words(line)
        if current and count + n > max_words:
            pieces.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        pieces.append("\n".join(current))
    # a single line longer than the limit (minified file...) is cut by words
    out: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= max_words:
            out.append(piece)
        else:
            out += [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return out


def split_long(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Pack paragraphs into windows of at most ``max_words``, with overlap."""
    if _words(text) <= max_words:
        return [text]
    blocks: list[str] = []
    for block in _paragraphs(text):
        blocks += _split_long_block(block, max_words) if _words(block) > max_words else [block]

    windows: list[str] = []
    current: list[str] = []
    count = 0
    for block in blocks:
        n = _words(block)
        if current and count + n > max_words:
            windows.append("\n\n".join(current))
            # carry the last paragraphs (up to `overlap` words) into the next window
            carried: list[str] = []
            kept = 0
            for prev in reversed(current):
                if kept + _words(prev) > overlap:
                    break
                carried.insert(0, prev)
                kept += _words(prev)
            current, count = carried, kept
        current.append(block)
        count += n
    if current:
        windows.append("\n\n".join(current))
    return windows


def merge_short(sections: list[tuple[str, str]], min_words: int = MIN_WORDS,
                max_words: int = MAX_WORDS) -> list[tuple[str, str]]:
    """Glue short sections to the following ones, without exceeding max_words."""
    merged: list[tuple[str, str]] = []
    heads: list[str] = []
    bodies: list[str] = []
    count = 0
    for head, body in sections:
        n = _words(body)
        if bodies and count >= min_words:
            merged.append((_join_heads(heads), "\n\n".join(bodies)))
            heads, bodies, count = [], [], 0
        if bodies and count + n > max_words:
            merged.append((_join_heads(heads), "\n\n".join(bodies)))
            heads, bodies, count = [], [], 0
        heads.append(head)
        bodies.append(body)
        count += n
    if bodies:
        merged.append((_join_heads(heads), "\n\n".join(bodies)))
    return merged


def _join_heads(heads: list[str]) -> str:
    seen: list[str] = []
    for head in heads:
        if head and head not in seen:
            seen.append(head)
    if len(seen) <= 2:
        return " / ".join(seen)
    return f"{seen[0]} / {seen[1]} / ..."


def chunk_document(doc: Document) -> list[Chunk]:
    """The chunks of a document, in reading order."""
    if doc.sections is not None:
        sections = doc.sections
    elif doc.kind == "markdown":
        sections = split_markdown(doc.text)
    else:
        sections = [("", doc.text)]

    # the document title is shown on its own: drop it from the breadcrumbs
    prefix = f"{doc.title} > "
    sections = [
        ("" if h == doc.title else h[len(prefix):] if h.startswith(prefix) else h, b)
        for h, b in sections
    ]
    chunks: list[Chunk] = []
    for heading, body in merge_short([(h, b) for h, b in sections if b.strip()]):
        parts = split_long(body)
        for i, part in enumerate(parts):
            label = heading if len(parts) == 1 else f"{heading} (part {i + 1})".strip()
            context = " > ".join(x for x in (doc.title, label) if x)
            chunks.append(Chunk(heading=label, text=part.strip(), context=context))
    for i, chunk in enumerate(chunks):
        chunk.ord = i
    return chunks


def kind_for(path_name: str, suffix: str) -> str | None:
    """Document kind for a file name, or None if the file is not indexed."""
    suffix = suffix.lower()
    if path_name in MAKEFILE_NAMES:
        return "code"
    if suffix in MARKDOWN_EXT:
        return "markdown"
    if suffix in CODE_EXT:
        return "code"
    if suffix in PDF_EXT:
        return "pages"
    if suffix in HTML_EXT:
        return "html"
    return None
