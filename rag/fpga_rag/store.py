"""The index: one SQLite file holding the text, a full-text index and the vectors.

Tables:

    meta        key/value: embedding backend, model, dimension, schema version
    sources     one row per indexed source ("repo", "amd:UG901", "local:notes")
    docs        one row per file or topic, with the sha256 of its text, so that
                re-indexing only re-embeds what changed
    chunks      the chunks: heading, text, breadcrumb
    chunks_fts  FTS5 full-text index (BM25), for exact identifiers such as
                "Synth 8-327" or XPAR_GPIO_0_DEVICE_ID that vectors blur
    chunks_vec  sqlite-vec table, one cosine vector per chunk

The embedding model is recorded at creation: searching with another model would
compare vectors from two different spaces and return noise without any error,
so the store refuses it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunking import CHUNKER_VERSION, Chunk, Document, chunk_document
from .config import private_dir

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    location TEXT NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL REFERENCES sources(name),
    key TEXT NOT NULL,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    UNIQUE (source, key)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(id),
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL,
    context TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text, tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"
);
"""


class RagError(RuntimeError):
    """Raised when the index cannot be used as asked (missing, other model...)."""


@dataclass
class IndexReport:
    source: str
    docs_total: int
    docs_embedded: int
    docs_unchanged: int
    docs_removed: int
    chunks_added: int
    seconds: float

    def line(self) -> str:
        return (f"[{self.source}] {self.docs_total} documents : {self.docs_embedded} indexes, "
                f"{self.docs_unchanged} inchanges, {self.docs_removed} retires, "
                f"{self.chunks_added} morceaux ajoutes ({self.seconds:.1f} s)")


def _sha(doc: Document) -> str:
    h = hashlib.sha256(CHUNKER_VERSION.encode())
    for part in (doc.title, doc.location, doc.kind, doc.text):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the index with sqlite-vec loaded."""
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RagError("sqlite-vec n'est pas installe : pip install -e rag") from exc
    private_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except AttributeError as exc:
        raise RagError(
            "ce Python est compile sans support des extensions SQLite ; "
            "utiliser le Python du systeme, de conda ou de python.org"
        ) from exc
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect(self.db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ meta
    def meta(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT key, value FROM meta"))

    def check_model(self, model_name: str, dim: int, create: bool) -> None:
        """Refuse to mix two embedding models in one index."""
        meta = self.meta()
        if "model" not in meta:
            if not create:
                raise RagError(
                    f"index vide ({self.db_path}) : lancer d'abord "
                    "`python -m fpga_rag index-repo`"
                )
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [("model", model_name), ("dim", str(dim)), ("schema", SCHEMA_VERSION)],
                )
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
                    f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
                )
            return
        if meta["model"] != model_name or int(meta["dim"]) != dim:
            raise RagError(
                f"l'index a ete construit avec '{meta['model']}' ({meta['dim']} dimensions), "
                f"le modele courant est '{model_name}' ({dim}). Revenir au meme modele "
                "(FPGA_RAG_BACKEND / FPGA_RAG_MODEL) ou reconstruire : "
                "`python -m fpga_rag reset` puis reindexer."
            )

    # ------------------------------------------------------------ write
    def _delete_docs(self, doc_ids: Iterable[int]) -> None:
        for doc_id in doc_ids:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            if ids:
                marks = ",".join("?" * len(ids))
                self.conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({marks})", ids)
                self.conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", ids)
                self.conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)
            self.conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))

    def index_source(self, source: str, kind: str, location: str, docs: list[Document],
                     embedder, progress=None) -> IndexReport:
        """Bring one source in line with ``docs``: add, update, remove.

        Unchanged documents (same sha256) are not re-embedded. Each document is
        committed on its own, so an interrupted run keeps what it has done.
        """
        start = time.time()
        self.check_model(embedder.name, embedder.dim, create=True)
        with self.conn:
            self.conn.execute(
                "INSERT INTO sources(name, kind, location, indexed_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, "
                "location = excluded.location",
                (source, kind, location, time.time()),
            )
        existing = {key: (doc_id, sha) for doc_id, key, sha in self.conn.execute(
            "SELECT id, key, sha256 FROM docs WHERE source = ?", (source,))}
        wanted = {doc.key: doc for doc in docs}

        removed = [existing[k][0] for k in existing if k not in wanted]
        with self.conn:
            self._delete_docs(removed)

        embedded = unchanged = chunks_added = 0
        for number, doc in enumerate(docs, 1):
            sha = _sha(doc)
            if doc.key in existing and existing[doc.key][1] == sha:
                unchanged += 1
                continue
            chunks: list[Chunk] = chunk_document(doc)
            vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
            with self.conn:
                if doc.key in existing:
                    self._delete_docs([existing[doc.key][0]])
                cur = self.conn.execute(
                    "INSERT INTO docs(source, key, title, location, sha256) VALUES (?, ?, ?, ?, ?)",
                    (source, doc.key, doc.title, doc.location, sha),
                )
                doc_id = cur.lastrowid
                for chunk, vector in zip(chunks, vectors):
                    cur = self.conn.execute(
                        "INSERT INTO chunks(doc_id, ord, heading, context, text) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, chunk.ord, chunk.heading, chunk.context, chunk.text),
                    )
                    chunk_id = cur.lastrowid
                    self.conn.execute(
                        "INSERT INTO chunks_fts(rowid, heading, text) VALUES (?, ?, ?)",
                        (chunk_id, chunk.context, chunk.text),
                    )
                    self.conn.execute(
                        "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, np.asarray(vector, dtype=np.float32).tobytes()),
                    )
            embedded += 1
            chunks_added += len(chunks)
            if progress:
                progress(number, len(docs), doc.key)

        with self.conn:
            self.conn.execute("UPDATE sources SET indexed_at = ? WHERE name = ?", (time.time(), source))
        return IndexReport(source, len(docs), embedded, unchanged, len(removed),
                           chunks_added, time.time() - start)

    def remove_source(self, source: str) -> int:
        ids = [r[0] for r in self.conn.execute("SELECT id FROM docs WHERE source = ?", (source,))]
        with self.conn:
            self._delete_docs(ids)
            self.conn.execute("DELETE FROM sources WHERE name = ?", (source,))
        return len(ids)

    # ------------------------------------------------------------ read
    def status(self) -> dict:
        meta = self.meta()
        rows = self.conn.execute(
            "SELECT s.name, s.kind, s.location, s.indexed_at, "
            "COUNT(DISTINCT d.id), COUNT(c.id) FROM sources s "
            "LEFT JOIN docs d ON d.source = s.name LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY s.name ORDER BY s.name"
        ).fetchall()
        return {
            "db": str(self.db_path),
            "model": meta.get("model"),
            "dim": int(meta["dim"]) if "dim" in meta else None,
            "sources": [
                {"name": n, "kind": k, "location": loc,
                 "indexed_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(t)),
                 "docs": d, "chunks": c}
                for n, k, loc, t, d, c in rows
            ],
            "chunks": sum(r[5] for r in rows),
        }

    def source_names(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT name FROM sources ORDER BY name")]

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """[(chunk_id, cosine similarity)] best first."""
        if "model" not in self.meta():
            return []
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (np.asarray(query_vec, dtype=np.float32).tobytes(), k),
        ).fetchall()
        return [(cid, 1.0 - dist) for cid, dist in rows]

    def keyword_search(self, fts_query: str, k: int) -> list[tuple[int, float]]:
        """[(chunk_id, bm25)] best first (bm25 is negative: lower is better)."""
        if not fts_query:
            return []
        return self.conn.execute(
            "SELECT rowid, bm25(chunks_fts, 0.5, 1.0) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (fts_query, k),
        ).fetchall()

    def chunks_by_id(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT c.id, c.heading, c.context, c.text, d.source, d.title, d.location, d.key "
            f"FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id IN ({marks})",
            ids,
        ).fetchall()
        return {
            r[0]: {"id": r[0], "heading": r[1], "context": r[2], "text": r[3],
                   "source": r[4], "title": r[5], "location": r[6], "key": r[7]}
            for r in rows
        }
