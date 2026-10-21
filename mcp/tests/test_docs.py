"""Outils search_docs / docs_index_status, sur un VRAI index SQLite.

Le moteur `hash` (sans modele, sans reseau) suffit a prouver le chemin complet :
indexation, recherche hybride, mise en forme avec citations, messages d'erreur.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parents[2] / "rag"
sys.path.insert(0, str(RAG_DIR))

pytest.importorskip("sqlite_vec")

from fpga_rag.__main__ import main as rag_main  # noqa: E402

from vivado_mcp.config import Settings  # noqa: E402
from vivado_mcp.tools import FpgaTools  # noqa: E402


@pytest.fixture()
def rag_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FPGA_RAG_HOME", str(tmp_path / "rag_home"))
    monkeypatch.setenv("FPGA_RAG_BACKEND", "hash")
    docs = tmp_path / "notes"
    docs.mkdir()
    (docs / "tcl.md").write_text(
        "# Relancer un build\n\n"
        "Si Vivado affiche `ERROR: [Common 17-69] Run 'synth_1' needs to be reset`, "
        "tester `get_property PROGRESS [get_runs synth_1]` puis appeler "
        "`reset_run synth_1` avant `launch_runs`.\n",
        encoding="utf-8",
    )
    (docs / "xdc.md").write_text(
        "# Contraintes\n\nUne horloge de 100 MHz se declare avec "
        "`create_clock -period 10.000 [get_ports clk]`.\n",
        encoding="utf-8",
    )
    return docs


def _tools(tmp_path: Path) -> FpgaTools:
    return FpgaTools(Settings.from_env(workspace=str(tmp_path)))


def test_search_docs_without_index_explains_what_to_do(tmp_path, rag_env):
    out = _tools(tmp_path).search_docs("reset_run")
    assert "indisponible" in out and "index-repo" in out


def test_search_docs_returns_cited_excerpts(tmp_path, rag_env):
    assert rag_main(["add", str(rag_env)]) == 0
    out = _tools(tmp_path).search_docs("Common 17-69 synth_1 needs to be reset", k=2)
    assert "[1]" in out and "notes/tcl.md" in out and "reset_run synth_1" in out
    assert "le corpus ne le dit pas" in out          # consigne d'ancrage


def test_search_docs_source_filter(tmp_path, rag_env):
    assert rag_main(["add", str(rag_env), "--name", "notes"]) == 0
    out = _tools(tmp_path).search_docs("horloge create_clock", sources=["repo"])
    assert "Aucun extrait" in out


def test_docs_index_status(tmp_path, rag_env):
    assert rag_main(["add", str(rag_env), "--name", "notes"]) == 0
    status = json.loads(_tools(tmp_path).docs_index_status())
    assert status["model"] == "hash-512"
    assert status["sources"][0]["name"] == "local:notes"
    assert status["sources"][0]["docs"] == 2
