from __future__ import annotations

from fpga_rag.chunking import (
    Document, chunk_document, merge_short, split_long, split_markdown,
)


def test_heading_inside_code_fence_is_not_a_heading():
    text = "# Titre\n\nIntro.\n\n```tcl\n# commentaire Tcl\nreset_run synth_1\n```\n\n## Suite\n\nFin."
    sections = split_markdown(text)
    assert [h for h, _ in sections] == ["Titre", "Titre > Suite"]
    assert "# commentaire Tcl" in sections[0][1]


def test_breadcrumb_follows_levels():
    text = "# A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n\n## D\n\nw"
    assert [h for h, _ in split_markdown(text)] == ["A", "A > B", "A > B > C", "A > D"]


def test_short_sections_are_merged_without_exceeding_the_limit():
    sections = [("a", "mot " * 30), ("b", "mot " * 30), ("c", "mot " * 200), ("d", "mot " * 10)]
    merged = merge_short(sections, min_words=100, max_words=300)
    assert [h for h, _ in merged] == ["a / b / ...", "d"]
    assert all(len(b.split()) <= 300 for _, b in merged)


def test_long_text_split_at_paragraphs_with_overlap():
    paras = [f"paragraphe{i} " + "mot " * 40 for i in range(10)]
    windows = split_long("\n\n".join(paras), max_words=100, overlap=45)
    assert len(windows) > 1
    assert all(len(w.split()) <= 100 for w in windows)
    # overlap: the last paragraph of a window opens the next one
    for first, second in zip(windows, windows[1:]):
        assert first.split("\n\n")[-1] == second.split("\n\n")[0]


def test_code_block_kept_whole_when_it_fits_and_cut_at_lines_otherwise():
    code = "```vhdl\n" + "\n".join(f"signal s{i} : std_logic;" for i in range(200)) + "\n```"
    windows = split_long("intro\n\n" + code, max_words=120, overlap=0)
    assert all(len(w.split()) <= 120 for w in windows)
    assert all(line.endswith(";") or line.startswith("```") or line == "intro" or not line
               for w in windows for line in w.splitlines())


def test_document_title_not_repeated_in_heading():
    doc = Document("repo", "docs/x.md", "Guide Tcl", "docs/x.md",
                   "# Guide Tcl\n\n" + "intro " * 150 + "\n\n## Pieges\n\n" + "piege " * 150)
    chunks = chunk_document(doc)
    assert [c.heading for c in chunks] == ["", "Pieges"]
    assert chunks[1].context == "Guide Tcl > Pieges"
    assert chunks[1].embedding_text().startswith("Guide Tcl > Pieges\n\n")
    assert [c.ord for c in chunks] == [0, 1]
