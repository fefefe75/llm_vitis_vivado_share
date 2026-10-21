from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from conftest import tiny_pdf
from fpga_rag.__main__ import main
from fpga_rag.config import Settings, guess_prefixes
from fpga_rag.embedders import HashEmbedder, OpenAIEmbedder

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ openai backend
class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        if body["model"] != "nomic-test":
            self.send_response(404)
            self.end_headers()
            return
        vecs = HashEmbedder(64).embed_documents(body["input"])
        data = [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)][::-1]
        payload = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_openai_backend_prefixes_order_and_batches(server):
    _Handler.seen.clear()
    emb = OpenAIEmbedder(server, "nomic-test", "search_query: ", "search_document: ", batch=2)
    assert emb.dim == 64
    vecs = emb.embed_documents(["a", "b", "c"])
    assert vecs.shape == (3, 64)
    # results reordered by "index" even though the server answered in reverse
    assert np.allclose(vecs[0], HashEmbedder(64).embed_documents(["search_document: a"])[0])
    assert _Handler.seen[1]["input"] == ["search_document: a", "search_document: b"]
    assert _Handler.seen[2]["input"] == ["search_document: c"]


def test_openai_backend_unknown_model_is_explicit(server):
    with pytest.raises(RuntimeError, match="charge"):
        OpenAIEmbedder(server, "absent", "", "")


def test_prefixes_guessed_from_model_name():
    assert guess_prefixes("nomic-ai/nomic-embed-text-v1.5-Q") == ("search_query: ", "search_document: ")
    assert guess_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert guess_prefixes("BAAI/bge-small-en-v1.5") == ("", "")


def test_default_data_dir_is_outside_the_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("FPGA_RAG_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    settings = Settings.from_env()
    assert settings.db_path == tmp_path / "fpga-rag" / "index.db"
    assert REPO not in settings.db_path.parents


# ------------------------------------------------------------ CLI
def test_cli_add_search_status_remove(rag_home, tmp_path, capsys):
    pytest.importorskip("pypdf")
    folder = tmp_path / "datasheets"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "pinout.pdf").write_bytes(tiny_pdf("GTY quad 224 reference clock pins"))
    (folder / "notes.txt").write_text("Le bloc MMCM genere 200 MHz depuis 100 MHz.", encoding="utf-8")
    (folder / "image.png").write_bytes(b"\x89PNG\x00\x00")
    (folder / ".cache").mkdir()
    (folder / ".cache" / "secret.md").write_text("ne pas indexer", encoding="utf-8")

    assert main(["add", str(folder)]) == 0
    assert main(["status", "--json"]) == 0
    out = capsys.readouterr().out
    status = json.loads(out[out.index("{"):])
    source = status["sources"][0]
    assert (source["name"], source["docs"]) == ("local:datasheets", 2)   # png and .cache/ skipped

    assert main(["search", "reference clock GTY", "-k", "1", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["location"] == "datasheets/sub/pinout.pdf"
    assert hits[0]["heading"] == "p. 1"

    assert main(["remove", "local:datasheets"]) == 0
    assert main(["remove", "local:datasheets"]) == 2


def test_cli_index_repo_and_eval(rag_home, capsys):
    assert main(["index-repo", "--repo", str(REPO)]) == 0
    capsys.readouterr()
    assert main(["eval", "--bench", str(REPO / "training/eval/bench.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "couverture de la recherche" in out
    # training/ (bench answers, fine-tuning data) is never part of the repo index
    from fpga_rag.store import Store
    with Store(Settings.from_env().db_path) as store:
        keys = [k for (k,) in store.conn.execute("SELECT key FROM docs")]
    assert keys and not any(k.startswith(("training/", "rag/", "mcp/tests")) for k in keys)
    assert "README.md" in keys and not any(k.endswith(".fr.md") and k.replace(".fr.", ".") in keys
                                           for k in keys)


def test_cli_reset_needs_confirmation(rag_home):
    assert main(["reset"]) == 2
    assert main(["reset", "--yes"]) == 0


def test_data_dir_inside_a_git_repo_is_never_committed(tmp_path, monkeypatch):
    import subprocess

    repo = tmp_path / "depot"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("FPGA_RAG_HOME", str(repo / "donnees_rag"))
    monkeypatch.setenv("FPGA_RAG_BACKEND", "hash")
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\ntexte interne", encoding="utf-8")
    assert main(["add", str(notes)]) == 0
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--ignored"],
                            capture_output=True, text=True, check=True).stdout
    untracked = [l for l in status.splitlines() if l.startswith("??")]
    assert untracked == [], untracked


def test_fastembed_backend_prefixes_and_normalisation(monkeypatch, tmp_path):
    """The real model is not downloadable here: fastembed's class is replaced by a
    stand-in that records what it receives, to check the wrapper around it."""
    import sys
    import types

    seen: dict = {}

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir):
            seen["init"] = (model_name, cache_dir)

        def embed(self, docs, batch_size=16):
            seen.setdefault("inputs", []).append(list(docs))
            for d in docs:
                yield np.full(8, float(len(d)), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setenv("FPGA_RAG_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("FPGA_RAG_BACKEND", "fastembed")
    monkeypatch.delenv("FPGA_RAG_MODEL", raising=False)
    from fpga_rag.embedders import make_embedder

    emb = make_embedder(Settings.from_env())
    assert seen["init"] == ("nomic-ai/nomic-embed-text-v1.5-Q", str(tmp_path / "h" / "models"))
    assert (tmp_path / "h" / ".gitignore").exists()
    assert emb.dim == 8
    vecs = emb.embed_documents(["abc"])
    assert seen["inputs"][-1] == ["search_document: abc"]
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)
    emb.embed_query("q")
    assert seen["inputs"][-1] == ["search_query: q"]


def test_very_large_k_is_clamped(rag_home, tmp_path):
    notes = tmp_path / "n"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nreset_run synth_1", encoding="utf-8")
    assert main(["add", str(notes)]) == 0
    assert main(["search", "reset_run", "-k", "5000", "--json"]) == 0
