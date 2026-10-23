"""Settings of the RAG, all overridable by environment variables.

Nothing here is a secret and nothing points inside the git repository by
default: the index and the downloaded documentation live in the user's data
directory, so that a `git add -A` can never pick them up. This matters because
people will index their own documents (datasheets, internal notes) with it.

Environment variables:

    FPGA_RAG_HOME          data directory (default: $XDG_DATA_HOME/fpga-rag,
                           i.e. ~/.local/share/fpga-rag)
    FPGA_RAG_DB            index file (default: $FPGA_RAG_HOME/index.db)
    FPGA_RAG_BACKEND       fastembed | openai | hash          (default: fastembed)
    FPGA_RAG_MODEL         embedding model name
                           (default: nomic-ai/nomic-embed-text-v1.5-Q for fastembed,
                            text-embedding-nomic-embed-text-v1.5 for openai)
    FPGA_RAG_BASE_URL      OpenAI-compatible server for the openai backend
                           (default: http://localhost:1234, LM Studio)
    FPGA_RAG_API_KEY       bearer token for that server, if it needs one
    FPGA_RAG_QUERY_PREFIX  / FPGA_RAG_DOC_PREFIX
                           task prefixes; guessed from the model name if unset
    FPGA_RAG_USER_AGENT    User-Agent used to download the AMD documentation
    FPGA_RAG_EMBED_BATCH   chunks embedded at once by fastembed (default: 4);
                           lower it if indexing is killed for lack of memory
    FPGA_RAG_EMBED_THREADS ONNX threads used by fastembed (default: all cores)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = {
    "fastembed": "nomic-ai/nomic-embed-text-v1.5-Q",
    "openai": "text-embedding-nomic-embed-text-v1.5",
    "hash": "hash-512",
}

# Task prefixes expected by the common retrieval models. Using the wrong one (or
# none) silently costs retrieval quality, so they are applied automatically.
PREFIXES = (
    ("nomic", "search_query: ", "search_document: "),
    ("e5", "query: ", "passage: "),
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
)

# Chunking: below MIN_WORDS a section is merged with the next one of the same
# file, above MAX_WORDS it is split at paragraph boundaries with OVERLAP words.
# Same sizes as the pipeline measured in training/RAPPORT_RAG.md (k=6).
MIN_WORDS = 120
MAX_WORDS = 700
OVERLAP_WORDS = 60


def _data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "fpga-rag"


def private_dir(path: Path) -> Path:
    """Create ``path`` with a ``.gitignore`` that ignores everything in it.

    Even if FPGA_RAG_HOME is pointed inside a git repository by mistake, the
    index, the model cache and the downloaded manuals cannot be committed.
    """
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".gitignore"
    if not marker.exists():
        marker.write_text("# fpga-rag : donnees locales, jamais versionnees\n*\n", encoding="utf-8")
    return path


def guess_prefixes(model: str) -> tuple[str, str]:
    """(query prefix, document prefix) for a model name."""
    low = model.lower()
    for key, query, doc in PREFIXES:
        if key in low:
            return query, doc
    return "", ""


@dataclass(frozen=True)
class Settings:
    home: Path
    db_path: Path
    backend: str
    model: str
    base_url: str
    api_key: str
    query_prefix: str
    doc_prefix: str
    user_agent: str

    @property
    def amd_dir(self) -> Path:
        """Where the downloaded AMD documentation is kept (never in git)."""
        return self.home / "amd"

    @classmethod
    def from_env(cls, **overrides: str) -> "Settings":
        env = {**os.environ, **{k: v for k, v in overrides.items() if v is not None}}
        home = Path(env.get("FPGA_RAG_HOME") or _data_home()).expanduser()
        db = Path(env.get("FPGA_RAG_DB") or home / "index.db").expanduser()
        backend = (env.get("FPGA_RAG_BACKEND") or "fastembed").strip().lower()
        if backend not in DEFAULT_MODELS:
            raise ValueError(
                f"FPGA_RAG_BACKEND={backend!r} inconnu : fastembed, openai ou hash"
            )
        model = env.get("FPGA_RAG_MODEL") or DEFAULT_MODELS[backend]
        query_prefix, doc_prefix = guess_prefixes(model)
        return cls(
            home=home,
            db_path=db,
            backend=backend,
            model=model,
            base_url=(env.get("FPGA_RAG_BASE_URL") or "http://localhost:1234").rstrip("/"),
            api_key=env.get("FPGA_RAG_API_KEY", ""),
            query_prefix=env.get("FPGA_RAG_QUERY_PREFIX", query_prefix),
            doc_prefix=env.get("FPGA_RAG_DOC_PREFIX", doc_prefix),
            user_agent=env.get("FPGA_RAG_USER_AGENT")
            or (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
