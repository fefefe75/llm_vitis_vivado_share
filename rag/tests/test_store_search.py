from __future__ import annotations

from pathlib import Path

import pytest

from fpga_rag.chunking import Document
from fpga_rag.config import Settings
from fpga_rag.embedders import HashEmbedder
from fpga_rag.search import Searcher, fts_query, format_hits
from fpga_rag.store import RagError, Store


def _doc(key: str, text: str, source: str = "local:t") -> Document:
    return Document(source, key, key, key, text)


DOCS = [
    _doc("build.md", "# Build\n\nAvant launch_runs, reset_run synth_1 si le run est termine "
         "(ERROR: [Common 17-69])."),
    _doc("vitis.md", "# Vitis\n\nLe GPIO se configure avec XPAR_GPIO_0_DEVICE_ID dans xparameters.h."),
    _doc("xdc.md", "# XDC\n\ncreate_clock -period 10.000 [get_ports clk] pour 100 MHz."),
]


@pytest.fixture()
def indexed(rag_home: Path) -> Settings:
    settings = Settings.from_env()
    with Store(settings.db_path) as store:
        store.index_source("local:t", "local", "/tmp/t", DOCS, HashEmbedder())
        store.index_source("amd:UG901", "amd", "x", [
            _doc("t1", "Synthesis attribute KEEP_HIERARCHY keeps the hierarchy.", "amd:UG901")],
            HashEmbedder())
    return settings


def test_incremental_reindex(rag_home):
    settings = Settings.from_env()
    with Store(settings.db_path) as store:
        first = store.index_source("local:t", "local", "x", DOCS, HashEmbedder())
        assert (first.docs_embedded, first.chunks_added) == (3, 3)
        again = store.index_source("local:t", "local", "x", DOCS, HashEmbedder())
        assert (again.docs_embedded, again.docs_unchanged) == (0, 3)
        changed = [DOCS[0], _doc("vitis.md", "# Vitis\n\nNouveau texte.")]
        third = store.index_source("local:t", "local", "x", changed, HashEmbedder())
        assert (third.docs_embedded, third.docs_unchanged, third.docs_removed) == (1, 1, 1)
        assert store.status()["chunks"] == 2
        # nothing orphaned in the full-text and vector tables
        for table, col in (("chunks_fts", "rowid"), ("chunks_vec", "chunk_id")):
            orphans = store.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} NOT IN (SELECT id FROM chunks)"
            ).fetchone()[0]
            assert orphans == 0


def test_other_model_is_refused(indexed):
    with Store(indexed.db_path) as store:
        with pytest.raises(RagError, match="hash-512"):
            store.index_source("local:t", "local", "x", DOCS, HashEmbedder(dim=256))


def test_exact_identifier_found_by_keyword_search(indexed):
    hits = Searcher(indexed).search("XPAR_GPIO_0_DEVICE_ID", k=1, mode="keyword")
    assert hits[0].location == "vitis.md"


def test_hybrid_search_and_citation(indexed):
    hits = Searcher(indexed).search("Common 17-69 reset_run", k=2)
    assert hits[0].location == "build.md"
    assert hits[0].similarity is not None and hits[0].bm25 is not None
    text = format_hits(hits)
    assert "[1]" in text and "build.md" in text and "le corpus ne le dit pas" in text


def test_source_filter_prefix(indexed):
    hits = Searcher(indexed).search("hierarchy synthesis", k=5, sources=["amd"])
    assert {h.source for h in hits} == {"amd:UG901"}
    assert Searcher(indexed).search("hierarchy", k=5, sources=["amd:UG999"]) == []


def test_remove_source(indexed):
    with Store(indexed.db_path) as store:
        assert store.remove_source("amd:UG901") == 1
        assert store.source_names() == ["local:t"]


def test_fts_query_is_safe():
    assert fts_query('a "b" AND (c) NEAR d-e') == '"a" OR "b" OR "and" OR "c" OR "near" OR "d" OR "e"'
    assert fts_query("!!!") == ""


def test_search_without_index_explains(rag_home):
    with pytest.raises(RagError, match="index-repo"):
        Searcher(Settings.from_env()).search("x")
