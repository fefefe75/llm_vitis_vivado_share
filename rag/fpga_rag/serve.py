"""Serve the index as an OpenAI-compatible model, so a chat client uses THE DATABASE.

Why: an index that only exists in a terminal serves one place. Here the same
chain (hybrid search + grounded generation) becomes a model any
OpenAI-compatible client on the machine can select — Open WebUI, LiteLLM, a
coding agent, a script — and every answer is built from excerpts read in
SQLite, with the citation of the file it came from.

    GET  /v1/models            the models served, and what they do
    POST /v1/chat/completions  grounded answer (RAG), or the bare model
    POST /v1/rag/search        the excerpts alone, without any LLM
    GET  /health               index path, chunks, model

Models served:

    fpga-rag           hybrid search (vectors + BM25) on the index
    fpga-rag-keyword   exact words only (BM25), for an identifier you know
    fpga-rag-direct    the model alone, no corpus (to compare, never to trust)

Settings, all by environment variable:

    FPGA_RAG_CHAT_MODEL   model ran by the upstream server (default gpt-oss-20b)
    FPGA_RAG_CHAT_URL     its /v1/chat/completions (default: FPGA_RAG_BASE_URL)
    FPGA_RAG_PORT         listening port (default 8098)
    FPGA_RAG_K           excerpts injected per answer (default 8)
    FPGA_RAG_MAX_PER_DOC excerpts from the same file (default 2)
    FPGA_RAG_TEMPERATURE default 0.2
    FPGA_RAG_MAX_TOKENS  default 4096
    FPGA_RAG_REASONING_EFFORT  "none" (default), "low", "high"...

An answer that cites nothing is a bug, and so is a service that hides the
reasoning it received: the reasoning travels in its own channel
(`reasoning_content` and `reasoning`), never inside `content`, exactly like LM
Studio sends it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Settings
from .search import Hit, Searcher, format_hits
from .store import RagError, Store

MODELS = {
    "fpga-rag": ("hybrid", "RAG sur l'index (vecteurs + BM25), extraits cites"),
    "fpga-rag-keyword": ("keyword", "RAG mots exacts (BM25), pour un identifiant connu"),
    "fpga-rag-direct": (None, "le modele seul, sans corpus (pour comparer)"),
}

# The contract is deliberately in front of everything else in the system prompt:
# placed at the end, the model answered "the corpus does not say" and then made
# a number up. It also forbids citing an excerpt it did not use.
CONTRAT = """Tu reponds UNIQUEMENT depuis les extraits fournis dans le message de \
l'utilisateur, entre <extraits> et </extraits>.

Ancrage : tout chiffre, prix, duree, date, version, nom de fichier ou option d'outil doit se \
trouver dans ces extraits ET porter sa citation [n]. Si la reponse n'y est pas, ecris exactement \
« le corpus ne le dit pas » puis ARRETE-TOI : aucun ordre de grandeur, aucune fourchette, aucun \
« en general », aucun exemple de memoire. Un chiffre plausible non cite est une faute, meme si tu \
le crois exact. Ne cite jamais un extrait que tu n'as pas utilise.

Tu es un assistant FPGA : VHDL, testbenches cocotb, contraintes XDC, flows Tcl Vivado, timing \
closure, Vitis embarque. Tu n'affirmes JAMAIS qu'un design fonctionne sans un log de simulation ou \
de build a l'appui. Reponds dans la langue de la question."""

MAX_HISTORY = 8
AUCUN_EXTRAIT = ("Aucun extrait de l'index ne correspond a cette question. "
                 "`le corpus ne le dit pas`.")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


class Config:
    """What the service needs to know, read once from the environment."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chat_model = os.environ.get("FPGA_RAG_CHAT_MODEL") or "gpt-oss-20b"
        self.chat_url = (os.environ.get("FPGA_RAG_CHAT_URL")
                         or settings.base_url.rstrip("/") + "/v1/chat/completions")
        self.port = _env_int("FPGA_RAG_PORT", 8098)
        self.host = os.environ.get("FPGA_RAG_HOST") or "127.0.0.1"
        self.k = _env_int("FPGA_RAG_K", 8)
        self.max_per_doc = _env_int("FPGA_RAG_MAX_PER_DOC", 2)
        self.temperature = _env_float("FPGA_RAG_TEMPERATURE", 0.2)
        self.max_tokens = _env_int("FPGA_RAG_MAX_TOKENS", 4096)
        self.reasoning = (os.environ.get("FPGA_RAG_REASONING_EFFORT") or "none").strip()


def dechiffrer(messages: list[dict]) -> tuple[str, list[dict]]:
    """Last user message = the question; the rest = history (system prompts dropped).

    Our own system prompt makes the rules, so a client's prompt is not forwarded:
    two contradictory prompts in one request is how grounding gets lost.
    """
    question, history = "", []
    for message in messages or []:
        role, content = message.get("role"), message.get("content") or ""
        if isinstance(content, list):        # some clients send content blocks
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        if role == "system":
            continue
        if role == "user":
            question = content
            history.append({"role": "user", "content": content})
        elif role == "assistant":
            history.append({"role": "assistant", "content": content})
    while history and history[-1]["role"] == "user":
        history.pop()
    return question, history[-MAX_HISTORY:]


def _extraits(hits: list[Hit]) -> str:
    return format_hits(hits, max_chars=1500)


def _cite(rank: int, texte: str) -> bool:
    return re.search(rf"\[{rank}\](?!\d)", texte) is not None


def bloc_sources(hits: list[Hit], texte: str) -> str:
    """Only the excerpts the answer actually cites; all of them if it cites none.

    A refusal gets no source list: printing eight files under "the corpus does
    not say" reads as if one of them had answered.
    """
    if not hits or "le corpus ne le dit pas" in texte.lower():
        return ""
    used = [h for h in hits if _cite(h.rank, texte)]
    lignes = ["", "---", "Sources :"]
    for hit in used or hits:
        where = f"{hit.title} > {hit.heading}" if hit.heading else hit.title
        lignes.append(f"[{hit.rank}] {where} — {hit.location}")
    return "\n".join(lignes)


def corps_sources(hits: list[Hit]) -> list[dict]:
    return [{"rank": h.rank, "source": h.source, "title": h.title, "heading": h.heading,
             "location": h.location, "score": h.score} for h in hits]


class Amont(RuntimeError):
    """The upstream model server refused or dropped the call."""


def appeler(config: Config, messages: list[dict], sans_reflexion: bool = False):
    """Yield ("content"|"reasoning"|"finish", text) from the upstream server."""
    charge: dict = {
        "model": config.chat_model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": True,          # LM Studio streams anyway: ask for it and read it
    }
    effort = "none" if sans_reflexion else config.reasoning
    if effort:
        charge["reasoning_effort"] = effort
    requete = urllib.request.Request(
        config.chat_url, data=json.dumps(charge).encode(),
        headers={"Content-Type": "application/json"})
    try:
        reponse = urllib.request.urlopen(requete, timeout=900)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise Amont(f"{config.chat_url} a repondu HTTP {exc.code} : {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise Amont(f"serveur de modeles injoignable ({config.chat_url}) : {exc}") from exc

    with reponse:
        for brut in reponse:
            ligne = brut.decode("utf-8", "replace").strip()
            if not ligne.startswith("data:"):
                continue
            donnee = ligne[5:].strip()
            if donnee == "[DONE]":
                break
            try:
                paquet = json.loads(donnee)
            except json.JSONDecodeError:
                continue
            choix = (paquet.get("choices") or [{}])[0]
            delta = choix.get("delta") or {}
            raisonnement = delta.get("reasoning_content") or delta.get("reasoning")
            if raisonnement:
                yield "reasoning", raisonnement
            if delta.get("content"):
                yield "content", delta["content"]
            if choix.get("finish_reason"):
                yield "finish", choix["finish_reason"]


def _messages(mode: str | None, question: str, history: list[dict], hits: list[Hit]) -> list[dict]:
    messages = [{"role": "system", "content": CONTRAT}]
    if mode is None:                                  # ...-direct: no corpus at all
        return messages + history + [{"role": "user", "content": question}]
    return messages + history + [{
        "role": "user",
        "content": f"<extraits>\n{_extraits(hits)}\n</extraits>\n\n{question}"}]


def _nom_modele(demande: str) -> str | None:
    """Exact name, or a prefixed one (a client may send `connection/fpga-rag`)."""
    return next((n for n in MODELS if demande == n or demande.endswith("/" + n)), None)


class Serveur(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    config: Config
    searcher: Searcher

    def log_message(self, format: str, *args) -> None:   # noqa: A002 (base signature)
        sys.stderr.write("  %s %s\n" % (self.address_string(), format % args))

    # ------------------------------------------------------------------ helpers
    def _json(self, code: int, obj: dict) -> None:
        corps = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _erreur(self, code: int, message: str) -> None:
        self._json(code, {"error": {"message": message, "type": "fpga_rag"}})

    def _lire(self) -> dict | None:
        taille = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(taille) or b"{}")
        except json.JSONDecodeError:
            return None

    # --------------------------------------------------------------------- GET
    def do_GET(self) -> None:                        # noqa: N802 (http.server API)
        chemin = self.path.rstrip("/")
        if chemin.endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "fpga-rag",
                 "description": desc} for name, (_, desc) in MODELS.items()]})
            return
        if chemin in ("", "/health", "/sante"):
            status = {"etat": "ok", "configure": str(self.config.settings.db_path),
                      "chat_model": self.config.chat_model, "k": self.config.k,
                      "max_per_doc": self.config.max_per_doc}
            try:
                with Store(self.config.settings.db_path) as store:
                    info = store.status()
                status.update({"chunks": info["chunks"], "model": info["model"],
                               "dim": info["dim"],
                               "sources": [s["name"] for s in info["sources"]]})
            except RagError as exc:
                status.update({"etat": "sans index", "erreur": str(exc)})
            self._json(200, status)
            return
        self._erreur(404, f"chemin inconnu : {self.path}")

    # -------------------------------------------------------------------- POST
    def do_POST(self) -> None:                       # noqa: N802 (http.server API)
        chemin = self.path.rstrip("/")
        demande = self._lire()
        if demande is None:
            self._erreur(400, "JSON invalide")
            return
        if chemin.endswith("/rag/search"):
            self._recherche(demande)
            return
        if chemin.endswith("/chat/completions"):
            self._chat(demande)
            return
        self._erreur(404, f"chemin inconnu : {self.path}")

    def _recherche(self, demande: dict) -> None:
        query = (demande.get("query") or "").strip()
        if not query:
            self._erreur(400, "champ 'query' manquant")
            return
        mode = demande.get("mode") or "hybrid"
        try:
            hits = self.searcher.search(query, k=int(demande.get("k") or self.config.k),
                                        sources=demande.get("sources"), mode=mode,
                                        max_per_doc=self.config.max_per_doc)
        except (RagError, ValueError) as exc:
            self._erreur(500, str(exc))
            return
        self._json(200, {"query": query, "mode": mode, "hits": corps_sources(hits),
                         "extraits": _extraits(hits)})

    def _chat(self, demande: dict) -> None:
        nom = _nom_modele(demande.get("model") or "fpga-rag")
        if nom is None:
            self._erreur(404, f"modele inconnu : {demande.get('model')} "
                              f"(servis : {', '.join(MODELS)})")
            return
        mode = MODELS[nom][0]
        question, history = dechiffrer(demande.get("messages") or [])
        if not question.strip():
            self._erreur(400, "aucun message utilisateur")
            return
        k = int(demande.get("k") or self.config.k)

        hits: list[Hit] = []
        if mode is not None:
            try:
                hits = self.searcher.search(question, k=k, mode=mode,
                                            max_per_doc=self.config.max_per_doc)
            except RagError as exc:
                self._erreur(500, str(exc))
                return

        ident = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        cree = int(time.time())
        if demande.get("stream"):
            self._flux(ident, cree, nom, mode, question, history, hits)
        else:
            self._reponse(ident, cree, nom, mode, question, history, hits)

    def _reponse(self, ident: str, cree: int, modele: str, mode: str | None, question: str,
                 history: list[dict], hits: list[Hit]) -> None:
        if not hits and mode is not None:
            texte, fin = AUCUN_EXTRAIT, "stop"
        else:
            texte, fin = "", "stop"
            for sans_reflexion in (False, True):
                morceaux: list[str] = []
                try:
                    for effet, morceau in appeler(self.config, _messages(
                            mode, question, history, hits), sans_reflexion=sans_reflexion):
                        if effet == "content":
                            morceaux.append(morceau)
                        elif effet == "finish":
                            fin = morceau
                except Amont as exc:
                    self._erreur(502, str(exc))
                    return
                texte = "".join(morceaux).strip()
                # an empty answer means the reasoning ate the whole token budget
                if texte or not sans_reflexion:
                    break
            if fin == "length":
                texte += "\n\n[reponse tronquee : budget de jetons atteint]"
            if not texte:
                texte = "[le modele n'a rendu aucun texte]"
        if mode is not None:
            texte += bloc_sources(hits, texte)
        self._json(200, {
            "id": ident, "object": "chat.completion", "created": cree, "model": modele,
            "choices": [{"index": 0, "finish_reason": fin,
                         "message": {"role": "assistant", "content": texte}}],
            "sources": corps_sources(hits),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    def _flux(self, ident: str, cree: int, modele: str, mode: str | None, question: str,
              history: list[dict], hits: list[Hit]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def envoyer(delta: dict, motif: str | None = None) -> None:
            charge = {"id": ident, "object": "chat.completion.chunk", "created": cree,
                      "model": modele,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": motif}]}
            self.wfile.write(f"data: {json.dumps(charge, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        contenu: list[str] = []
        fin = "stop"

        def ajouter(morceau: str) -> None:
            contenu.append(morceau)
            envoyer({"content": morceau})

        try:
            envoyer({"role": "assistant", "content": ""})
            if not hits and mode is not None:
                ajouter(AUCUN_EXTRAIT)
            else:
                messages = _messages(mode, question, history, hits)
                for sans_reflexion in (False, True):
                    for effet, morceau in appeler(self.config, messages,
                                                  sans_reflexion=sans_reflexion):
                        if effet == "content":
                            ajouter(morceau)
                        elif effet == "reasoning":
                            # its own channel: never inside `content`
                            envoyer({"reasoning_content": morceau, "reasoning": morceau})
                        elif effet == "finish":
                            fin = morceau
                    # a retry only happens when the reasoning ate every token
                    if "".join(contenu).strip() or not sans_reflexion:
                        break
                if fin == "length":
                    ajouter("\n\n[reponse tronquee : budget de jetons atteint]")
                if not "".join(contenu).strip():
                    ajouter("[le modele n'a rendu aucun texte ; voir la reflexion ci-dessus]")
            if mode is not None:
                bloc = bloc_sources(hits, "".join(contenu))
                if bloc:
                    envoyer({"content": bloc})
            envoyer({}, motif=fin)
            final = json.dumps({"id": ident, "sources": corps_sources(hits)}, ensure_ascii=False)
            self.wfile.write(f"data: {final}\n\n".encode())
        except Amont as exc:
            envoyer({"content": f"\n[erreur amont : {exc}]"}, motif="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main(settings: Settings | None = None, host: str | None = None, port: int | None = None) -> int:
    settings = settings or Settings.from_env()
    config = Config(settings)
    if host:
        config.host = host
    if port:
        config.port = port
    if config.max_per_doc < 0:
        config.max_per_doc = 0

    try:
        with Store(settings.db_path) as store:
            info = store.status()
    except RagError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1
    if not info["chunks"]:
        print(f"index vide ({settings.db_path}) : `python -m fpga_rag index-repo`",
              file=sys.stderr)
        return 1

    searcher = Searcher(settings)
    if settings.backend != "keyword":
        searcher.embedder.embed_query("prechauffage")     # pay the model load once
    Serveur.config = config
    Serveur.searcher = searcher

    print("fpga-rag — API OpenAI-compatible")
    print(f"  index    : {info['chunks']} morceaux, {info['model']} ({info['dim']} dim.)")
    print(f"  modele   : {config.chat_model} via {config.chat_url}")
    print(f"  extraits : k={config.k}, {config.max_per_doc} max par fichier")
    print(f"  adresse  : http://{config.host}:{config.port}/v1")
    print(f"  modeles  : {', '.join(MODELS)}")
    print(f"  essai    : curl http://127.0.0.1:{config.port}/v1/models")
    print("  arret    : Ctrl+C\n")

    serveur = ThreadingHTTPServer((config.host, config.port), Serveur)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\n  arret.")
    finally:
        serveur.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
