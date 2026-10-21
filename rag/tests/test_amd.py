"""AMD collection without network: docs.amd.com answers are simulated.

The JSON shapes (catalogue entries with `metadata`, `paginatedToc[0].pageToc`
with recursive `children`) are those recorded in the original pipeline that
downloaded 3 411 topics; they are reproduced here, not invented for the test.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("bs4")

from fpga_rag import amd as amd_mod  # noqa: E402
from fpga_rag.amd import AmdDocs, html_to_text, walk_toc  # noqa: E402
from fpga_rag.config import Settings  # noqa: E402

CATALOG = [
    {"id": "map-901", "title": "Vivado Design Suite User Guide: Synthesis (UG901)",
     "metadata": [{"key": "ft:locale", "values": ["en-US"]}, {"key": "Revision", "values": ["2025.2"]}]},
    {"id": "map-901-ja", "title": "Vivado Design Suite User Guide: Synthesis (UG901)",
     "metadata": [{"key": "ft:locale", "values": ["ja-JP"]}]},
    {"id": "map-901-rn", "title": "Release Notes for Synthesis (UG901)",
     "metadata": [{"key": "ft:locale", "values": ["en-US"]}]},
]
TOC = {"paginatedToc": [{"pageToc": [
    {"title": "Synthesis", "children": [
        {"title": "Attributes", "contentId": "c1", "prettyUrl": "/r/en-US/ug901/attributes"},
        {"title": "KEEP", "contentId": "c2", "prettyUrl": "/r/en-US/ug901/keep",
         "children": [{"title": "KEEP Example", "contentId": "c3", "prettyUrl": "/r/en-US/ug901/keep-ex"}]},
    ]},
]}]}
TOPICS = {
    "c1": "<h1>Attributes</h1><p>Attributes are set in RTL or XDC.</p>",
    "c2": "<h1>KEEP</h1><p>Use KEEP to prevent optimisation.</p>"
          "<table><tr><th>Value</th><th>Effect</th></tr><tr><td>TRUE</td><td>keep</td></tr></table>",
    "c3": "<h1>KEEP Example</h1><pre>(* keep = \"true\" *) signal s : std_logic;</pre>",
}


def fake_http_get(url: str, user_agent: str, timeout: int = 90):
    path = url.removeprefix(amd_mod.BASE)
    if path == "/api/khub/maps":
        return 200, json.dumps(CATALOG).encode()
    if path == "/api/khub/maps/map-901/pages":
        return 200, json.dumps(TOC).encode()
    for cid, html in TOPICS.items():
        if path == f"/api/khub/maps/map-901/topics/{cid}/content":
            return 200, html.encode()
    return 404, b""


@pytest.fixture()
def amd(rag_home, monkeypatch):
    monkeypatch.setattr(amd_mod, "http_get", fake_http_get)
    monkeypatch.setattr(amd_mod, "DELAY_S", 0)
    return AmdDocs(Settings.from_env(), log=lambda *_: None)


def test_html_to_text_keeps_tables_lists_and_code():
    html = ("<h2>Title</h2><p>Intro</p><ul><li><p>item one</p></li><li>item two</li></ul>"
            "<div><table><tr><td>A|B</td><td>1</td></tr></table></div>"
            "<pre>reset_run synth_1\nlaunch_runs synth_1</pre>")
    text = html_to_text(html)
    assert "## Title" in text
    assert text.count("item one") == 1          # nested <p> not duplicated
    assert "| A\\|B | 1 |" in text              # table outside any paragraph is kept
    assert "```\nreset_run synth_1\nlaunch_runs synth_1\n```" in text


def test_walk_toc_breadcrumbs():
    topics = walk_toc(TOC["paginatedToc"][0]["pageToc"])
    assert [(t["contentId"], t["path"]) for t in topics] == [
        ("c1", ["Synthesis"]), ("c2", ["Synthesis"]), ("c3", ["Synthesis", "KEEP"])]


def test_select_keeps_the_english_user_guide(amd):
    found = amd.select(["UG901", "UG999"])
    assert list(found) == ["UG901"]
    assert found["UG901"]["id"] == "map-901"
    assert found["UG901"]["revision"] == "2025.2"


def test_fetch_then_index_documents(amd):
    amd.select(["UG901"])
    assert amd.measure() == {"UG901": 3}
    assert amd.fetch() == 3
    assert (amd.raw / "UG901.done").exists()
    assert amd.fetch() == 3                      # second run: already collected, no download
    records = [json.loads(line) for line in (amd.raw / "UG901.jsonl").open()]
    assert records[1]["url"] == "https://docs.amd.com/r/en-US/ug901/keep"
    assert "| TRUE | keep |" in records[1]["text"]
    docs = amd.documents("UG901")
    # the two small topics under "Synthesis" are grouped, the nested one stays apart
    assert [d.key for d in docs] == ["c1", "c3"]
    assert docs[0].source == "amd:UG901"
    assert docs[0].location.endswith("/attributes")
    assert [h for h, _ in docs[0].sections] == ["Synthesis > Attributes", "Synthesis > KEEP"]


def test_fetch_with_limit_does_not_mark_done(amd):
    amd.select(["UG901"])
    assert amd.fetch(limit=1) == 1
    assert not (amd.raw / "UG901.done").exists()


def test_amd_data_lives_outside_the_repository(amd, rag_home):
    assert amd.dir == rag_home / "amd"


# ------------------------------------------------------------ direct links
from conftest import tiny_pdf  # noqa: E402
from fpga_rag.__main__ import main  # noqa: E402
from fpga_rag.amd import code_from_url  # noqa: E402

SPEC_PAGE = ("<html><head><title>Example Link Protocol Specification (SP999)</title></head>"
             "<body><a href=\"/api/khub/documents/AbC_12-x/content\">PDF</a></body></html>")


def fake_with_pdf(url: str, user_agent: str, timeout: int = 90):
    path = url.removeprefix(amd_mod.BASE)
    if path == "/v/u/en-US/example_link_protocol_spec_sp999":
        return 200, SPEC_PAGE.encode()
    if path == "/api/khub/documents/AbC_12-x/content":
        return 200, tiny_pdf("Lane alignment uses a dedicated control block")
    return fake_http_get(url, user_agent, timeout)


def test_code_from_url():
    assert code_from_url("https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Attributes") == "UG901"
    assert code_from_url("https://docs.amd.com/r/en-US/pg1234-some-ip/Product-Specification") == "PG1234"
    assert code_from_url("https://docs.amd.com/v/u/en-US/example_link_protocol_spec_sp999") == "SP999"
    assert code_from_url("https://docs.amd.com/v/u/en-US/ug999") == "UG999"
    assert code_from_url("https://example.com/whatever") is None


def test_select_add_keeps_previous_selection(amd):
    CATALOG.append({"id": "map-903", "title": "Vivado Design Suite User Guide: Using Constraints (UG903)",
                    "metadata": [{"key": "ft:locale", "values": ["en-US"]}]})
    try:
        amd.select(["UG901"])
        assert list(amd.select(["UG903"], add=True)) == ["UG901", "UG903"]
        assert list(amd.select(["UG903"])) == ["UG903"]
    finally:
        CATALOG.pop()


def test_cli_amd_add_structured_and_pdf_links(rag_home, monkeypatch, capsys):
    pytest.importorskip("pypdf")
    monkeypatch.setattr(amd_mod, "http_get", fake_with_pdf)
    monkeypatch.setattr(amd_mod, "DELAY_S", 0)
    rc = main(["amd", "add",
               "https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Attributes",
               "https://docs.amd.com/v/u/en-US/example_link_protocol_spec_sp999"])
    assert rc == 0
    capsys.readouterr()
    assert main(["status", "--json"]) == 0
    out = capsys.readouterr().out
    names = {s["name"]: s for s in json.loads(out[out.index("{"):])["sources"]}
    assert set(names) == {"amd:UG901", "amd:SP999"}
    assert main(["search", "lane alignment control block", "-k", "1", "--json"]) == 0
    hit = json.loads(capsys.readouterr().out)[0]
    assert (hit["source"], hit["heading"]) == ("amd:SP999", "p. 1")
    assert hit["title"] == "Example Link Protocol Specification (SP999)"
    assert hit["location"].endswith("/v/u/en-US/example_link_protocol_spec_sp999")
    # `amd index` rebuilds both kinds from the local copies, without network
    monkeypatch.setattr(amd_mod, "http_get", lambda *a, **k: (None, b"offline"))
    assert main(["amd", "index"]) == 0


def test_pdf_link_without_pdf_is_explicit(amd, monkeypatch):
    monkeypatch.setattr(amd_mod, "http_get", lambda url, ua, timeout=90: (200, b"<html>no link</html>"))
    with pytest.raises(RuntimeError, match="aucun lien"):
        amd.fetch_pdf("https://docs.amd.com/v/u/en-US/x_sp999")
