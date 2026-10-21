"""Hybrid search: meaning (vectors) and exact words (BM25), fused by rank.

Why both: a vector finds "how do I rerun a synthesis that is already done" even
when the text says "reset_run synth_1", but it blurs exact identifiers
("Synth 8-327", XPAR_GPIO_0_DEVICE_ID). BM25 is the opposite. Reciprocal Rank
Fusion adds 1 / (60 + rank) from each list, which needs no score calibration
between the two.

There is no relevance threshold on purpose: on this corpus no cosine threshold
separates relevant from off-topic chunks. The model is told instead to answer
only from the excerpts and to say "the corpus does not say" otherwise
(prompts/quickref.md, "Contract").

Evidence diversity: the chunks of one file often repeat each other, and letting
them fill every slot costs breadth. `max_per_doc` caps how many chunks one
document may contribute. Measured on the repo bench (30 questions, k=6,
`text-embedding-nomic-embed-text-v1.5` through LM Studio): 93.2 % of the
required strings retrieved without the cap, 94.9 % with 2 chunks per document,
and every task except `agent_workflow` then reaches 100 %. The cap is on by
default; `--par-doc 0` restores the old behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .embedders import Embedder, make_embedder
from .store import RagError, Store

RRF_K = 60
MODES = ("hybrid", "vector", "keyword")
# two chunks per file by default: see the measurement in the module docstring
DEFAULT_MAX_PER_DOC = 2


@dataclass
class Hit:
    rank: int
    score: float          # fused score (hybrid) or the single-list score
    similarity: float | None
    bm25: float | None
    source: str
    title: str
    heading: str
    location: str
    text: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def fts_query(text: str, max_terms: int = 32) -> str:
    """A safe FTS5 query: each word quoted, joined by OR (BM25 weighs them)."""
    terms: list[str] = []
    for tok in re.findall(r"\w+", text.lower()):
        if tok not in terms:
            terms.append(tok)
    return " OR ".join(f'"{t}"' for t in terms[:max_terms])


def _match_source(name: str, wanted: list[str] | None) -> bool:
    """`amd` matches every `amd:*` source; an exact name matches itself."""
    if not wanted:
        return True
    return any(name == w or name.startswith(w.rstrip(":") + ":") for w in wanted)


class Searcher:
    """Keeps the embedder loaded between queries (loading it is the slow part)."""

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self.settings)
        return self._embedder

    def search(self, query: str, k: int = 6, sources: list[str] | None = None,
               mode: str = "hybrid", max_per_doc: int = DEFAULT_MAX_PER_DOC) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode inconnu : {mode} ({', '.join(MODES)})")
        if not query.strip():
            return []
        db: Path = self.settings.db_path
        if not db.exists():
            raise RagError(
                f"aucun index ({db}). Le construire : `python -m fpga_rag index-repo`"
            )
        with Store(db) as store:
            # sqlite-vec accepts at most k = 4096 neighbours per query
            pool = min(4096, max(50, k * 10) if sources else max(30, k * 5))
            vec: list[tuple[int, float]] = []
            kw: list[tuple[int, float]] = []
            if mode in ("hybrid", "vector"):
                emb = self.embedder
                store.check_model(emb.name, emb.dim, create=False)
                vec = store.vector_search(emb.embed_query(query), pool)
            if mode in ("hybrid", "keyword"):
                kw = store.keyword_search(fts_query(query), pool)

            info = store.chunks_by_id(list({cid for cid, _ in vec} | {cid for cid, _ in kw}))
            sim = dict(vec)
            bm = dict(kw)
            fused: dict[int, float] = {}
            for results in (vec, kw):
                rank = 0
                for cid, _ in results:
                    if cid not in info or not _match_source(info[cid]["source"], sources):
                        continue
                    rank += 1
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

            # best first, at most `max_per_doc` chunks of the same file
            per_doc: dict[tuple[str, str], int] = {}
            best: list[int] = []
            for cid in sorted(fused, key=lambda c: -fused[c]):
                row = info[cid]
                doc = (row["source"], row["location"])
                if max_per_doc and per_doc.get(doc, 0) >= max_per_doc:
                    continue
                per_doc[doc] = per_doc.get(doc, 0) + 1
                best.append(cid)
                if len(best) >= k:
                    break
            hits = []
            for rank, cid in enumerate(best, 1):
                row = info[cid]
                hits.append(Hit(
                    rank=rank,
                    score=round(fused[cid], 5),
                    similarity=round(sim[cid], 4) if cid in sim else None,
                    bm25=round(-bm[cid], 3) if cid in bm else None,
                    source=row["source"], title=row["title"], heading=row["heading"],
                    location=row["location"], text=row["text"],
                ))
            return hits


def format_hits(hits: list[Hit], max_chars: int = 1500, query: str | None = None) -> str:
    """Excerpts ready to be pasted in a prompt, each with its citation."""
    if not hits:
        return ("Aucun extrait trouve. Repondre : « le corpus ne le dit pas » "
                "(ne pas completer de memoire).")
    lines = []
    if query:
        lines.append(f"Requete : {query}")
    lines.append(
        "Extraits de la documentation indexee. Repondre UNIQUEMENT depuis ces extraits, "
        "en citant la source [n] de chaque fait ; sinon dire « le corpus ne le dit pas »."
    )
    for hit in hits:
        text = hit.text if len(hit.text) <= max_chars else hit.text[:max_chars].rstrip() + " [...]"
        where = f"{hit.title} > {hit.heading}" if hit.heading else hit.title
        lines.append(f"\n[{hit.rank}] {where}\n    source : {hit.source} | {hit.location}\n{text}")
    return "\n".join(lines)
