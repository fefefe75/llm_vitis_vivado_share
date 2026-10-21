"""Download public AMD documentation from docs.amd.com, topic by topic.

Everything lands in ``$FPGA_RAG_HOME/amd`` (outside the repository): the
manuals belong to AMD and must not be redistributed, not even in a private
repository. Anyone can rebuild the same corpus with these commands.

Endpoints (public, no account, allowed by docs.amd.com/robots.txt):

    full catalogue          GET /api/khub/maps
    table of contents       GET /api/khub/maps/<id>/pages
    content of a topic      GET /api/khub/maps/<id>/topics/<contentId>/content

A topic is one DITA fragment written by AMD on one subject, so the corpus is
already cut along the manual's own structure.

Workflow:  catalog -> select -> measure (read it!) -> fetch -> index
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .chunking import Document
from .config import MIN_WORDS, Settings, private_dir

BASE = "https://docs.amd.com"
DELAY_S = 0.25        # politeness: at most ~4 requests per second
TIMEOUT_S = 90
RETRIES = 3

# Default selection: the Vivado / Vitis design-flow guides, independent of any
# device family. Add device manuals (TRM, transceivers, IP product guides) with
# `amd select --codes ...` according to your own hardware.
DEFAULT_CODES = (
    "UG835",   # Vivado Tcl Command Reference
    "UG894",   # Using Tcl Scripting
    "UG892",   # Design Flows Overview
    "UG900",   # Logic Simulation
    "UG901",   # Synthesis
    "UG903",   # Using Constraints
    "UG904",   # Implementation
    "UG906",   # Design Analysis and Closure Techniques
    "UG908",   # Programming and Debugging
    "UG949",   # UltraFast Design Methodology Guide
    "UG896",   # Designing with IP
    "UG994",   # Designing IP Subsystems Using IP Integrator
    "UG1118",  # Creating and Packaging Custom IP
    "UG1400",  # Vitis Embedded Software Development
)
SKIP_TITLE_WORDS = ("release note", "errata", "addendum", "quick start", "installation")
BIG_MANUAL = 900


# ------------------------------------------------------------------ network
def http_get(url: str, user_agent: str, timeout: int = TIMEOUT_S) -> tuple[int | None, bytes]:
    """GET with retries. Returns (HTTP status or None, body)."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                return exc.code, b""
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    return None, str(last).encode()


# ------------------------------------------------------------------ HTML -> text
def _table_to_markdown(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        vals = [" ".join(c.get_text(" ", strip=True).split()) for c in tr.find_all(["td", "th"])]
        if any(vals):
            rows.append([v.replace("|", "\\|") for v in vals])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def html_to_text(fragment: str) -> str:
    """Readable Markdown-ish text from an HTML/DITA fragment, tables included.

    Tables are rendered to Markdown BEFORE being replaced by a marker, otherwise
    their values (pin names, register fields, option lists) would be lost.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("lecture HTML : pip install -e 'rag[amd]'") from exc

    soup = BeautifulSoup(fragment, "lxml")
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    tables = soup.find_all("table")
    rendered = [_table_to_markdown(t) for t in tables]
    for i, table in enumerate(tables):
        table.replace_with(soup.new_string(f"\n@@TABLE{i}@@\n"))

    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd",
                             "pre", "figcaption"]):
        if el.find_parent("pre") is not None:
            continue
        if el.name == "pre":
            parts.append("```\n" + el.get_text("\n").strip() + "\n```")
            continue
        if el.name in ("p", "li", "dd", "dt") and el.find_parent(["li", "dd"]) is not None:
            continue          # already included through the enclosing list item
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if re.fullmatch(r"h[1-6]", el.name):
            parts.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    body = "\n\n".join(parts)
    for i, md in enumerate(rendered):
        marker = f"@@TABLE{i}@@"
        body = body.replace(marker, md) if marker in body else body + ("\n\n" + md if md else "")
    body = re.sub(r"\n{3,}", "\n\n", body)
    return re.sub(r"[ \t]{2,}", " ", body).strip()


# ------------------------------------------------------------------ catalogue
class AmdDocs:
    def __init__(self, settings: Settings, log: Callable[[str], None] = print) -> None:
        self.settings = settings
        self.dir = settings.amd_dir
        self.raw = self.dir / "raw"
        self.log = log

    @property
    def catalog_path(self) -> Path:
        return self.dir / "catalog.json"

    @property
    def targets_path(self) -> Path:
        return self.dir / "targets.json"

    def _get(self, path: str) -> tuple[int | None, bytes]:
        return http_get(BASE + path, self.settings.user_agent)

    def catalog(self, refresh: bool = False) -> list[dict]:
        if refresh or not self.catalog_path.exists():
            status, raw = self._get("/api/khub/maps")
            if status != 200:
                raise RuntimeError(
                    f"catalogue AMD illisible (HTTP {status}) : {raw[:200]!r}. "
                    "Acces a docs.amd.com bloque par un proxy ?"
                )
            private_dir(self.dir)
            self.catalog_path.write_bytes(raw)
            self.log(f"catalogue telecharge : {len(raw) / 1e6:.1f} Mo -> {self.catalog_path}")
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("maps", [])

    @staticmethod
    def _meta(entry: dict, key: str) -> str:
        for item in entry.get("metadata") or []:
            if item.get("key") == key:
                values = item.get("values") or []
                return values[0] if values else ""
        return ""

    def select(self, codes: tuple[str, ...] | list[str] = DEFAULT_CODES,
               add: bool = False) -> dict[str, dict]:
        """Pick one English manual per code; the code must be in the title.

        ``add=True`` keeps the manuals already selected instead of replacing them.
        """
        docs = self.catalog()
        found: dict[str, dict] = dict(self.targets()) if add and self.targets_path.exists() else {}
        for code in codes:
            code = code.upper().strip()
            hits = []
            for entry in docs:
                title = entry.get("title") or ""
                if not re.search(r"\(\s*" + re.escape(code) + r"\s*\)", title, re.I):
                    continue
                if self._meta(entry, "ft:locale") not in ("en-US", "en", ""):
                    continue
                if any(w in title.lower() for w in SKIP_TITLE_WORDS):
                    continue
                hits.append(entry)
            if not hits:
                self.log(f"  {code:<8} absent du catalogue (aucun titre anglais contenant ({code}))")
                continue
            best = min(hits, key=lambda e: len(e.get("title") or ""))
            found[code] = {
                "id": best["id"],
                "title": best.get("title", code),
                "url": self._meta(best, "url"),
                "revision": self._meta(best, "Revision"),
            }
            extra = f"  ({len(hits)} candidats, verifier la version)" if len(hits) > 1 else ""
            self.log(f"  {code:<8} rev {found[code]['revision'] or '?':<6} "
                     f"{found[code]['title'][:60]}{extra}")
        private_dir(self.dir)
        self.targets_path.write_text(json.dumps(found, indent=1, ensure_ascii=False), encoding="utf-8")
        self.log(f"{len(found)} manuels retenus -> {self.targets_path}")
        return found

    def targets(self) -> dict[str, dict]:
        if not self.targets_path.exists():
            raise RuntimeError("aucune selection : lancer d'abord `python -m fpga_rag amd select`")
        return json.loads(self.targets_path.read_text(encoding="utf-8"))

    def _toc(self, map_id: str) -> list[dict] | None:
        status, raw = self._get(f"/api/khub/maps/{map_id}/pages")
        if status != 200 or not raw:
            return None
        toc = json.loads(raw.decode("utf-8"))
        page = (toc.get("paginatedToc") or [{}])[0]
        return walk_toc(page.get("pageToc"))

    def measure(self) -> dict[str, int]:
        """Number of topics per selected manual, BEFORE downloading anything."""
        sizes: dict[str, int] = {}
        for code, info in self.targets().items():
            topics = self._toc(info["id"])
            sizes[code] = len(topics) if topics is not None else -1
            self.log(f"  {code:<8} {sizes[code]:>6} sujets   {info['title'][:52]}")
            time.sleep(DELAY_S)
        total = sum(n for n in sizes.values() if n > 0)
        self.log(f"\nTOTAL : {total} sujets, ~{total / 2 / 60:.0f} min de collecte a 2 requetes/s")
        big = {c: n for c, n in sizes.items() if n > BIG_MANUAL}
        if big:
            self.log("ATTENTION, tres gros manuels : " + ", ".join(f"{c} ({n})" for c, n in big.items()))
            self.log("  une reference de registres (milliers de tableaux de bits) noie la recherche :")
            self.log("  la retirer de la selection sauf besoin precis.")
        return sizes

    def fetch(self, only: list[str] | None = None, limit: int | None = None,
              refresh: bool = False) -> int:
        """Download every selected manual (or ``only`` those). Idempotent."""
        total = 0
        for code, info in self.targets().items():
            if only and code not in {c.upper() for c in only}:
                continue
            total += self._fetch_manual(code, info, limit, refresh)
        self.log(f"TOTAL : {total} sujets dans {self.raw}")
        return total

    def _fetch_manual(self, code: str, info: dict, limit: int | None, refresh: bool) -> int:
        dest = self.raw / f"{code}.jsonl"
        done = self.raw / f"{code}.done"
        # The .done marker separates a finished manual from an interrupted one,
        # which would otherwise look complete and never be resumed.
        if done.exists() and dest.exists() and not refresh and not limit:
            n = sum(1 for _ in dest.open(encoding="utf-8"))
            self.log(f"  [{code}] deja collecte : {n} sujets")
            return n
        done.unlink(missing_ok=True)
        topics = self._toc(info["id"])
        if topics is None:
            self.log(f"  [{code}] table des matieres illisible : ignore")
            return 0
        if limit:
            topics = topics[:limit]
        private_dir(self.raw)
        self.log(f"  [{code}] {len(topics)} sujets : {info['title'][:56]}")
        written = empty = failed = 0
        start = time.time()
        with dest.open("w", encoding="utf-8") as out:
            for number, topic in enumerate(topics, 1):
                status, raw = self._get(
                    f"/api/khub/maps/{info['id']}/topics/{topic['contentId']}/content")
                time.sleep(DELAY_S)
                if status != 200 or not raw:
                    empty += status == 404
                    failed += status != 404
                    continue
                try:
                    text = html_to_text(raw.decode("utf-8", "replace"))
                except Exception:   # noqa: BLE001 - one bad topic must not stop a manual
                    failed += 1
                    continue
                if len(text) < 40:
                    empty += 1
                    continue
                pretty = topic.get("prettyUrl") or ""
                out.write(json.dumps({
                    "doc": code,
                    "doc_title": info["title"],
                    "topic_id": topic["contentId"],
                    "section": topic["title"],
                    "section_path": " > ".join(topic["path"]),
                    "url": BASE + pretty if pretty.startswith("/") else pretty,
                    "text": text,
                }, ensure_ascii=False) + "\n")
                written += 1
                if number % 50 == 0:
                    self.log(f"      {number}/{len(topics)}  ecrits={written}")
        if not limit:
            done.write_text(f"{written} sujets, {time.time() - start:.0f} s\n", encoding="utf-8")
        self.log(f"  [{code}] {written} sujets, {empty} vides, {failed} echecs, "
                 f"{time.time() - start:.0f} s")
        return written

    def collected(self) -> list[str]:
        return sorted(p.stem for p in self.raw.glob("*.jsonl")) if self.raw.exists() else []

    # -------------------------------------------------------------- links
    @property
    def pdf_dir(self) -> Path:
        return self.dir / "pdf"

    def collected_pdfs(self) -> list[str]:
        return sorted(p.stem for p in self.pdf_dir.glob("*.pdf")) if self.pdf_dir.exists() else []

    def fetch_pdf(self, url: str, code: str | None = None) -> str:
        """Download a PDF-only document (``/v/u/...`` page or direct content link).

        The ``/v/u/`` page is only a landing page: the PDF itself is behind
        ``/api/khub/documents/<id>/content``, whose link is read from the page.
        """
        code = (code or code_from_url(url) or "").upper()
        if not re.fullmatch(r"[A-Z0-9_.-]+", code):
            raise RuntimeError(f"code introuvable dans {url} : le donner avec --name")
        title = code
        content_path = _document_content_path(url)
        if content_path is None:
            status, page = http_get(url, self.settings.user_agent)
            if status != 200:
                raise RuntimeError(f"{url} : HTTP {status}")
            html = page.decode("utf-8", "replace")
            content_path = _document_content_path(html)
            match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
            if match:
                title = " ".join(match.group(1).split())
            if content_path is None:
                raise RuntimeError(f"{url} : aucun lien /api/khub/documents/.../content dans la page")
        status, pdf = self._get(content_path)
        if status != 200 or not pdf.startswith(b"%PDF"):
            raise RuntimeError(f"{BASE}{content_path} : pas un PDF (HTTP {status})")
        private_dir(self.pdf_dir)
        (self.pdf_dir / f"{code}.pdf").write_bytes(pdf)
        (self.pdf_dir / f"{code}.json").write_text(json.dumps(
            {"title": title, "url": url, "content": BASE + content_path}, ensure_ascii=False),
            encoding="utf-8")
        self.log(f"  [{code}] PDF {len(pdf) / 1e6:.1f} Mo : {title[:60]}")
        return code

    def pdf_documents(self, code: str) -> list[Document]:
        from .sources import _pdf_sections

        path = self.pdf_dir / f"{code}.pdf"
        meta_path = self.pdf_dir / f"{code}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        sections = _pdf_sections(path)
        if not sections:
            self.log(f"  [{code}] PDF sans texte extractible (scan ?) : ignore")
            return []
        return [Document(f"amd:{code}", code, meta.get("title", code), meta.get("url", str(path)),
                         "\n\n".join(t for _, t in sections), "pages", sections)]

    def add_links(self, urls: list[str], name: str | None = None) -> list[str]:
        """Collect what docs.amd.com links point to; returns the codes to index."""
        codes: list[str] = []
        for url in urls:
            if "/v/u/" in url or _document_content_path(url):
                codes.append(self.fetch_pdf(url, name))
                continue
            code = code_from_url(url)
            if not code:
                raise RuntimeError(f"lien non reconnu : {url} (attendu docs.amd.com/r/... ou /v/u/...)")
            if code not in self.select([code], add=True):
                continue
            self.fetch(only=[code])
            if (self.raw / f"{code}.jsonl").exists():
                codes.append(code)
        return codes

    def documents(self, code: str) -> list[Document]:
        """The topics of one downloaded manual, ready to index.

        Consecutive small topics under the same parent are grouped (a topic of
        three lines makes a poor chunk); the group is cited with the URL of its
        first topic.
        """
        path = self.raw / f"{code}.jsonl"
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        docs: list[Document] = []
        i = 0
        while i < len(records):
            group = [records[i]]
            words = len(records[i]["text"].split())
            while (i + len(group) < len(records)
                   and records[i + len(group)]["section_path"] == records[i]["section_path"]
                   and words < MIN_WORDS):
                words += len(records[i + len(group)]["text"].split())
                group.append(records[i + len(group)])
            first = group[0]
            sections = [(" > ".join(x for x in (r["section_path"], r["section"]) if x), r["text"])
                        for r in group]
            docs.append(Document(
                source=f"amd:{code}",
                key=first.get("topic_id") or first.get("url") or f"{code}-{i}",
                title=f"{first['doc_title']} ({code})" if code not in first["doc_title"] else first["doc_title"],
                location=first.get("url", ""),
                text="\n\n".join(t for _, t in sections),
                kind="markdown",
                sections=sections,
            ))
            i += len(group)
        return docs


def code_from_url(url: str) -> str | None:
    """Manual code of a docs.amd.com link: /r/en-US/ug901-vivado-synthesis/... -> UG901,
    /v/u/en-US/some_spec_sp999 -> SP999."""
    match = re.search(r"/r/[^/]+/([a-z]{2,3}\d{3,5})(?:[-/]|$)", url, re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"/v/u/[^/]+/(?:[^/?#]*[_-])?([a-z]{2,3}\d{3,5})(?:[?#]|$)", url, re.I)
    return match.group(1).upper() if match else None


def _document_content_path(text: str) -> str | None:
    match = re.search(r"/api/khub/documents/([A-Za-z0-9_~.-]+)/content", text)
    return match.group(0) if match else None


def walk_toc(nodes, path: tuple[str, ...] = ()) -> list[dict]:
    """Flatten the table of contents into topics with their breadcrumb."""
    out: list[dict] = []
    for node in nodes or []:
        title = (node.get("title") or "").strip()
        here = path + (title,) if title else path
        if node.get("contentId"):
            out.append({"title": title, "contentId": node["contentId"],
                        "prettyUrl": node.get("prettyUrl"), "path": list(path)})
        out.extend(walk_toc(node.get("children"), here))
    return out
