"""Le service OpenAI-compatible : modeles, recherche, ancrage, flux SSE.

Un faux serveur de modeles tient le role de LM Studio : les tests n'ont besoin ni
de GPU, ni de modele, ni de reseau. Ce qu'ils verifient est ce qui compte :
l'index est bien interroge, les extraits partent dans le prompt, la reponse porte
ses citations, et un refus ne s'accompagne pas d'une liste de sources.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fpga_rag.config import Settings
from fpga_rag.embedders import make_embedder
from fpga_rag.search import Searcher
from fpga_rag.serve import Config, Serveur
from fpga_rag.sources import folder_documents
from fpga_rag.store import Store

CORPUS = {
    "xdc.md": "# Contraintes XDC\n\n## Horloge\n\nPour une horloge de 100 MHz, le Tcl est "
              "`create_clock -period 10.000 -name i_clk [get_ports i_clk]`.\n",
    "cocotb.md": "# Testbench cocotb\n\n## Mesure\n\nNe pas faire de `wait for` en boucle : "
                 "mesurer avec un compteur de cycles, puis comparer au log.\n",
}


@pytest.fixture()
def service(rag_home: Path, monkeypatch: pytest.MonkeyPatch):
    """A real index (hash backend), a fake upstream, and the service in a thread."""
    corpus = rag_home / "corpus"
    corpus.mkdir(parents=True)
    for name, text in CORPUS.items():
        (corpus / name).write_text(text, encoding="utf-8")

    settings = Settings.from_env()
    with Store(settings.db_path) as store:
        store.index_source("local:test", "local", str(corpus),
                           folder_documents(corpus, "local:test"), make_embedder(settings))

    amont = ThreadingHTTPServer(("127.0.0.1", 0), _Amont)
    threading.Thread(target=amont.serve_forever, daemon=True).start()

    monkeypatch.setenv("FPGA_RAG_CHAT_URL",
                       f"http://127.0.0.1:{amont.server_address[1]}/v1/chat/completions")
    monkeypatch.setenv("FPGA_RAG_CHAT_MODEL", "faux-modele")
    monkeypatch.setenv("FPGA_RAG_REASONING_EFFORT", "none")
    config = Config(settings)
    Serveur.config = config
    Serveur.searcher = Searcher(settings)

    serveur = ThreadingHTTPServer(("127.0.0.1", 0), Serveur)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{serveur.server_address[1]}"
    _Amont.recu.clear()
    yield base
    serveur.shutdown()
    amont.shutdown()


class _Amont(BaseHTTPRequestHandler):
    """Le serveur de modeles : repond en SSE comme LM Studio, et retient la demande."""

    recu: dict = {}

    def log_message(self, format: str, *args) -> None:   # noqa: A002
        pass

    def do_POST(self) -> None:                       # noqa: N802 (http.server API)
        taille = int(self.headers.get("Content-Length") or 0)
        demande = json.loads(self.rfile.read(taille) or b"{}")
        _Amont.recu.update(demande)
        texte = ("le corpus ne le dit pas" if "hors corpus" in json.dumps(demande)
                 else "Utilise `create_clock` avec -period 10.000 [1].")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        for delta in ({"role": "assistant", "content": ""},
                      {"reasoning_content": "je cherche", "reasoning": "je cherche"}):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
        for mot in texte.split(" "):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': mot + ' '}}]})}\n\n".encode())
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _post(base: str, chemin: str, corps: dict) -> dict:
    requete = urllib.request.Request(base + chemin, data=json.dumps(corps).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        return json.loads(reponse.read())


def _chat(question: str, modele: str = "fpga-rag", **extra) -> dict:
    return {"model": modele, "messages": [{"role": "user", "content": question}], **extra}


def test_modeles_servis(service: str) -> None:
    with urllib.request.urlopen(service + "/v1/models", timeout=10) as reponse:
        modeles = [m["id"] for m in json.loads(reponse.read())["data"]]
    assert modeles == ["fpga-rag", "fpga-rag-keyword", "fpga-rag-direct"]


def test_sante_decrit_l_index(service: str) -> None:
    with urllib.request.urlopen(service + "/health", timeout=10) as reponse:
        sante = json.loads(reponse.read())
    assert sante["etat"] == "ok"
    assert sante["chunks"] >= 2 and "local:test" in sante["sources"]


def test_recherche_sans_llm(service: str) -> None:
    resultat = _post(service, "/v1/rag/search", {"query": "create_clock 100 MHz", "k": 2})
    assert resultat["hits"], "aucun extrait : l'index n'est pas interroge"
    assert resultat["hits"][0]["location"].endswith("xdc.md")
    assert "create_clock" in resultat["extraits"]


def test_reponse_ancree_avec_citations(service: str) -> None:
    reponse = _post(service, "/v1/chat/completions", _chat("horloge 100 MHz en XDC"))
    contenu = reponse["choices"][0]["message"]["content"]
    assert "[1]" in contenu, "la reponse ne cite aucun extrait"
    assert "Sources :" in contenu
    # les extraits ont bien ete lus dans la base avant de partir au modele
    utilisateur = [m["content"] for m in _Amont.recu["messages"] if m["role"] == "user"]
    assert any("<extraits>" in c and "create_clock" in c for c in utilisateur)
    assert reponse["sources"][0]["location"].endswith("xdc.md")


def test_refus_sans_liste_de_sources(service: str) -> None:
    reponse = _post(service, "/v1/chat/completions", _chat("prix d'une licence hors corpus ?"))
    contenu = reponse["choices"][0]["message"]["content"]
    assert "le corpus ne le dit pas" in contenu
    assert "Sources :" not in contenu


def test_modele_direct_ne_lit_pas_l_index(service: str) -> None:
    reponse = _post(service, "/v1/chat/completions", _chat("horloge 100 MHz", modele="fpga-rag-direct"))
    assert reponse["sources"] == []
    utilisateur = [m["content"] for m in _Amont.recu["messages"] if m["role"] == "user"]
    assert not any("<extraits>" in c for c in utilisateur), "le mode direct a injecte des extraits"


def test_modele_inconnu_refuse(service: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as erreur:
        _post(service, "/v1/chat/completions", _chat("horloge", modele="gpt-imaginaire"))
    assert erreur.value.code == 404


def test_flux_sse_transmet_le_raisonnement_et_les_sources(service: str) -> None:
    requete = urllib.request.Request(
        service + "/v1/chat/completions",
        data=json.dumps(_chat("horloge 100 MHz", stream=True)).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        brut = reponse.read().decode()
    assert brut.rstrip().endswith("data: [DONE]")
    lignes = [ligne[5:].strip() for ligne in brut.splitlines() if ligne.startswith("data:")]
    paquets = [json.loads(ligne) for ligne in lignes if ligne != "[DONE]"]

    contenu, reflexion = [], []
    for paquet in paquets:
        if "choices" in paquet:
            delta = paquet["choices"][0]["delta"]
            contenu.append(delta.get("content") or "")
            reflexion.append(delta.get("reasoning_content") or "")
    texte = "".join(contenu)
    assert "create_clock" in texte and "Sources :" in texte
    # le raisonnement ne doit jamais se melanger a la reponse
    assert "je cherche" not in texte
    assert "je cherche" in "".join(reflexion)
    assert paquets[-1]["sources"][0]["location"].endswith("xdc.md")
