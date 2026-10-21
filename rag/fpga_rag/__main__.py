"""Command line: ``python -m fpga_rag <command>`` (or ``fpga-rag <command>``).

    index-repo            index this repository's docs, skills, prompts, templates
    add PATH [--name N]   index a local folder or file (md, txt, pdf, html, code)
    remove NAME           remove a source from the index
    search "question"     search the index (hybrid by default)
    serve                 serve the index as an OpenAI-compatible model (RAG + citations)
    status                what is indexed, with which model
    reset --yes           delete the index (to change model, for instance)
    amd catalog|select|measure|fetch|index
                          download public AMD manuals locally, then index them
    eval                  retrieval coverage on training/eval/bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .config import Settings
from .embedders import make_embedder
from .search import DEFAULT_MAX_PER_DOC, MODES, Searcher, format_hits
from .sources import folder_documents, repo_documents
from .store import RagError, Store


def _repo_root() -> Path:
    env = os.environ.get("FPGA_RAG_REPO")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[2]
    return here if (here / "AGENTS.md").exists() else Path.cwd()


def _progress(number: int, total: int, key: str) -> None:
    if number == total or number % 25 == 0:
        print(f"    {number}/{total}  {key[:70]}", flush=True)


def _index(settings: Settings, source: str, kind: str, location: str, docs) -> int:
    if not docs:
        print(f"[{source}] aucun fichier indexable dans {location}")
        return 1
    embedder = make_embedder(settings)
    print(f"[{source}] {len(docs)} documents, modele {embedder.name} ({embedder.dim} dim.)")
    with Store(settings.db_path) as store:
        report = store.index_source(source, kind, location, docs, embedder, _progress)
    print(report.line())
    print(f"index : {settings.db_path}")
    return 0


def cmd_index_repo(args, settings: Settings) -> int:
    repo = Path(args.repo).resolve() if args.repo else _repo_root()
    return _index(settings, "repo", "repo", str(repo), repo_documents(repo))


def cmd_add(args, settings: Settings) -> int:
    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        print(f"introuvable : {target}", file=sys.stderr)
        return 2
    name = args.name or (target.stem if target.is_file() else target.name)
    if not re.fullmatch(r"[\w.-]+", name):
        print(f"nom de source invalide : {name!r} (lettres, chiffres, . _ -)", file=sys.stderr)
        return 2
    source = f"local:{name}"
    return _index(settings, source, "local", str(target), folder_documents(target, source))


def cmd_remove(args, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        if args.name not in store.source_names():
            print(f"source inconnue : {args.name} (voir `status`)", file=sys.stderr)
            return 2
        n = store.remove_source(args.name)
    print(f"{args.name} retire ({n} documents)")
    return 0


def cmd_search(args, settings: Settings) -> int:
    hits = Searcher(settings).search(args.query, args.k, args.source, args.mode,
                                     max_per_doc=args.par_doc)
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=1))
    else:
        print(format_hits(hits, args.max_chars))
        if hits:
            print("\nscores : " + "  ".join(
                f"[{h.rank}] cos={h.similarity if h.similarity is not None else '-'} "
                f"bm25={h.bm25 if h.bm25 is not None else '-'}" for h in hits))
    return 0


def cmd_status(args, settings: Settings) -> int:
    if not settings.db_path.exists():
        print(f"aucun index ({settings.db_path}) : lancer `python -m fpga_rag index-repo`")
        return 1
    with Store(settings.db_path) as store:
        status = store.status()
    status["configured_backend"] = settings.backend
    status["configured_model"] = settings.model
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=1))
        return 0
    print(f"index   : {status['db']}")
    print(f"modele  : {status['model']} ({status['dim']} dim.)  "
          f"| configure : {settings.backend} / {settings.model}")
    print(f"morceaux: {status['chunks']}")
    for src in status["sources"]:
        print(f"  {src['name']:<24} {src['docs']:>6} docs {src['chunks']:>7} morceaux  "
              f"{src['indexed_at']}  {src['location']}")
    return 0


def cmd_reset(args, settings: Settings) -> int:
    if not args.yes:
        print(f"supprimerait {settings.db_path} : relancer avec --yes", file=sys.stderr)
        return 2
    for suffix in ("", "-wal", "-shm"):
        Path(str(settings.db_path) + suffix).unlink(missing_ok=True)
    print(f"index supprime : {settings.db_path}")
    return 0


def cmd_amd(args, settings: Settings) -> int:
    from .amd import DEFAULT_CODES, AmdDocs

    amd = AmdDocs(settings)
    if args.amd_cmd == "catalog":
        print(f"{len(amd.catalog(refresh=args.refresh))} manuels au catalogue")
    elif args.amd_cmd == "select":
        amd.select(args.codes or DEFAULT_CODES, add=args.add)
    elif args.amd_cmd == "add":
        codes = amd.add_links(args.urls, args.name)
        if not codes:
            print("rien de collecte", file=sys.stderr)
            return 1
        rc = 0
        for code in codes:
            docs = amd.pdf_documents(code) if code in amd.collected_pdfs() else amd.documents(code)
            rc |= _index(settings, f"amd:{code}", "amd", f"docs.amd.com {code}", docs)
        return rc
    elif args.amd_cmd == "measure":
        amd.measure()
    elif args.amd_cmd == "fetch":
        amd.fetch(args.only, args.limit, args.refresh)
    elif args.amd_cmd == "index":
        pdfs = amd.collected_pdfs()
        codes = [c.upper() for c in args.only] if args.only else amd.collected() + pdfs
        if not codes:
            print("rien a indexer : lancer d'abord `amd fetch` ou `amd add`", file=sys.stderr)
            return 1
        rc = 0
        for code in codes:
            docs = amd.pdf_documents(code) if code in pdfs else amd.documents(code)
            rc |= _index(settings, f"amd:{code}", "amd", f"docs.amd.com {code}", docs)
        return rc
    return 0


def _norm(text: str) -> str:
    # same normalisation as training/eval/run_eval.py
    t = text.lower().replace("*", "").replace("`", "")
    return re.sub(r"[\s ]+", " ", t)


def cmd_eval(args, settings: Settings) -> int:
    """Share of the bench's required strings present in the retrieved excerpts.

    It is the ceiling of what a model can cite with this retrieval, measured
    without any LLM. The bench and its answers live in training/, which is
    never indexed; the command checks it.
    """
    bench = Path(args.bench) if args.bench else _repo_root() / "training" / "eval" / "bench.jsonl"
    items = [json.loads(line) for line in bench.open(encoding="utf-8") if line.strip()]
    with Store(settings.db_path) as store:
        leaked = store.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE key LIKE 'training/%' OR key LIKE '%bench%.jsonl'"
        ).fetchone()[0]
    if leaked:
        print(f"ATTENTION : {leaked} fichier(s) du banc ou de training/ sont indexes : "
              "le score ne vaut rien.", file=sys.stderr)
        return 3
    searcher = Searcher(settings)
    per_task: dict[str, list[float]] = {}
    rows = []
    got_total = max_total = 0.0
    for item in items:
        hits = searcher.search(item["question"], args.k, args.source, args.mode,
                               max_per_doc=args.par_doc)
        context = _norm("\n".join(f"{h.heading}\n{h.text}" for h in hits))
        got = sum(c["weight"] for c in item["checks"] if _norm(c["must_contain"]) in context)
        top = sum(c["weight"] for c in item["checks"])
        got_total += got
        max_total += top
        per_task.setdefault(item["task"], [0.0, 0.0])
        per_task[item["task"]][0] += got
        per_task[item["task"]][1] += top
        rows.append({"id": item["id"], "task": item["task"], "got": got, "max": top,
                     "sources": [f"{h.source}:{h.location}" for h in hits]})
        if args.verbose:
            print(f"  {item['id']}  {got:>3}/{top:<3} {item['task']}")
    print(f"\ncouverture de la recherche (k={args.k}, mode {args.mode}) : "
          f"{got_total:.0f}/{max_total:.0f} = {100 * got_total / max_total:.1f} %")
    for task, (got, top) in sorted(per_task.items()):
        print(f"  {task:<16} {100 * got / top:5.1f} %")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "k": args.k, "mode": args.mode, "model": settings.model, "score": got_total,
            "max": max_total, "items": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    from .serve import main as servir

    return servir(settings, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fpga-rag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index-repo", help="indexer ce depot")
    p.add_argument("--repo", help="racine du depot (defaut : celui de ce paquet)")
    p.set_defaults(func=cmd_index_repo)

    p = sub.add_parser("add", help="indexer un dossier ou un fichier local")
    p.add_argument("path")
    p.add_argument("--name", help="nom de la source (defaut : nom du dossier)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("remove", help="retirer une source de l'index")
    p.add_argument("name", help="ex. local:notes, amd:UG901")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("search", help="chercher dans l'index")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--source", nargs="*", help="restreindre (repo, amd, amd:UG901, local:x)")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--par-doc", type=int, default=DEFAULT_MAX_PER_DOC,
                   help="morceaux max par fichier (0 = sans limite)")
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("serve", help="servir l'index comme modele OpenAI-compatible")
    p.add_argument("--host", default=None, help="defaut : 127.0.0.1 (ne pas ouvrir sur le reseau)")
    p.add_argument("--port", type=int, default=None, help="defaut : FPGA_RAG_PORT ou 8098")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="contenu de l'index")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("reset", help="supprimer l'index")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("amd", help="documentation AMD publique, en local")
    amd_sub = p.add_subparsers(dest="amd_cmd", required=True)
    q = amd_sub.add_parser("catalog", help="telecharger le catalogue (16 Mo)")
    q.add_argument("--refresh", action="store_true")
    q = amd_sub.add_parser("select", help="choisir les manuels par code (UG901...)")
    q.add_argument("--codes", nargs="*", help="defaut : guides Vivado/Vitis generaux")
    q.add_argument("--add", action="store_true", help="ajouter a la selection au lieu de la remplacer")
    q = amd_sub.add_parser("add", help="collecter et indexer des liens docs.amd.com (/r/... ou PDF /v/u/...)")
    q.add_argument("urls", nargs="+")
    q.add_argument("--name", help="code a utiliser si le lien n'en contient pas")
    amd_sub.add_parser("measure", help="compter les sujets AVANT de telecharger")
    q = amd_sub.add_parser("fetch", help="telecharger les manuels choisis")
    q.add_argument("--only", nargs="*")
    q.add_argument("--limit", type=int, help="sujets max par manuel (essai)")
    q.add_argument("--refresh", action="store_true")
    q = amd_sub.add_parser("index", help="indexer les manuels telecharges")
    q.add_argument("--only", nargs="*")
    p.set_defaults(func=cmd_amd)

    p = sub.add_parser("eval", help="couverture de la recherche sur le banc")
    p.add_argument("--bench")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--source", nargs="*")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--par-doc", type=int, default=DEFAULT_MAX_PER_DOC)
    p.add_argument("--out")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        return args.func(args, settings)
    except (RagError, RuntimeError, ValueError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrompu : ce qui a ete indexe est conserve", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
