from __future__ import annotations

from flash_vlm.assembler import MarkdownAssembler
from flash_vlm.schemas import PageResult


def test_assemble_with_frontmatter_and_markers():
    pages = [
        PageResult(page=1, markdown="# Title\n\ntext", ok=True),
        PageResult(page=2, markdown="", ok=False, error="timeout"),
    ]
    assembler = MarkdownAssembler(page_separator="\n\n", page_markers=True, frontmatter=True)
    markdown = assembler.assemble(pages, "doc.pdf", "test-model")

    assert markdown.startswith("---\n")
    assert "source: doc.pdf" in markdown
    assert "model: test-model" in markdown
    assert "<!-- Page 1 -->" in markdown
    assert "Страница 2 не распознана: timeout" in markdown


def test_assemble_without_frontmatter():
    pages = [PageResult(page=1, markdown="hello", ok=True)]
    assembler = MarkdownAssembler(page_separator="\n\n", page_markers=False, frontmatter=False)
    markdown = assembler.assemble(pages, "doc.pdf")
    assert markdown == "hello\n"
