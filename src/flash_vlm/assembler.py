"""Сборщик итогового Markdown из результатов страниц."""

from __future__ import annotations

from .schemas import PageResult

__all__ = ["MarkdownAssembler"]


class MarkdownAssembler:
    """Склеивает Markdown-блоки страниц в единый документ."""

    def __init__(
        self,
        page_separator: str = "\n\n",
        page_markers: bool = False,
        frontmatter: bool = True,
    ) -> None:
        self.page_separator = page_separator
        self.page_markers = page_markers
        self.frontmatter = frontmatter

    def assemble(self, pages: list[PageResult], source: str, model: str | None = None) -> str:
        """Собирает итоговый Markdown из упорядоченных результатов страниц."""
        blocks: list[str] = []
        for result in pages:
            text = (result.markdown or "").strip()
            if result.ok and text:
                block = text
                if self.page_markers:
                    block = f"<!-- Page {result.page} -->\n\n{block}"
                blocks.append(block)
            elif not result.ok:
                error = result.error or "неизвестная ошибка"
                blocks.append(f"> ⚠️ Страница {result.page} не распознана: {error}")

        body = self.page_separator.join(blocks).strip()
        if self.frontmatter:
            front = self._frontmatter(source, pages, model)
            return f"{front}\n\n{body}\n" if body else f"{front}\n"
        return f"{body}\n" if body else ""

    @staticmethod
    def _frontmatter(source: str, pages: list[PageResult], model: str | None) -> str:
        ok_count = sum(1 for p in pages if p.ok)
        lines = [
            "---",
            f"source: {source}",
            f"pages: {len(pages)}",
            f"pages_ok: {ok_count}",
        ]
        if model:
            lines.append(f"model: {model}")
        lines.extend(["generator: Flash-VLM DocParser", "---"])
        return "\n".join(lines)
