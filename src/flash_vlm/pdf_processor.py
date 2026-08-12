"""PDF-процессор: рендеринг страниц в изображения + «умный» ресайз.

Поддерживаются два бэкенда:

- ``pymupdf`` (по умолчанию) — не требует системных зависимостей;
- ``pdf2image`` — требует установленного Poppler (``poppler-utils``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from .utils import image_to_data_url, parse_pages

__all__ = ["PageImage", "PdfProcessor", "resize_to_width"]


@dataclass
class PageImage:
    """Изображение одной страницы документа."""

    page_number: int  # 1-based
    width: int
    height: int
    image: Image.Image

    def to_data_url(self, image_format: str = "PNG") -> str:
        return image_to_data_url(self.image, image_format)


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    """Изменяет ширину изображения, сохраняя соотношение сторон (LANCZOS)."""
    if width <= 0 or image.width == width:
        return image
    ratio = width / image.width
    new_height = max(1, round(image.height * ratio))
    return image.resize((width, new_height), Image.Resampling.LANCZOS)


class PdfProcessor:
    """Конвертирует страницы PDF в список ``PageImage`` с ресайзом."""

    def __init__(self, dpi: int = 200, width: int = 512, backend: str = "pymupdf") -> None:
        self.dpi = max(1, dpi)
        self.width = width
        self.backend = (backend or "pymupdf").lower()
        if self.backend not in {"pymupdf", "pdf2image"}:
            raise ValueError(f"Неизвестный бэкенд рендеринга: {backend!r}")

    def page_count(self, pdf_path: str | Path) -> int:
        """Количество страниц в документе."""
        with pymupdf.open(str(pdf_path)) as doc:  # noqa: SIM117
            return doc.page_count

    def render(self, pdf_path: str | Path, pages: object = None) -> list[PageImage]:
        """Рендерит заданные страницы (по умолчанию все) с ресайзом до ``width``."""
        page_numbers = parse_pages(pages, self.page_count(pdf_path))
        if self.backend == "pdf2image":
            return self._render_pdf2image(pdf_path, page_numbers)
        return self._render_pymupdf(pdf_path, page_numbers)

    # ------------------------------------------------------------------
    def _render_pymupdf(self, pdf_path: str | Path, pages: list[int]) -> list[PageImage]:
        result: list[PageImage] = []
        with pymupdf.open(str(pdf_path)) as doc:
            for page_number in pages:
                page = doc[page_number - 1]
                matrix = pymupdf.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                image = resize_to_width(image, self.width)
                result.append(PageImage(page_number, image.width, image.height, image))
        return result

    def _render_pdf2image(self, pdf_path: str | Path, pages: list[int]) -> list[PageImage]:
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Бэкенд 'pdf2image' требует установки пакета: pip install pdf2image"
            ) from exc

        try:
            first, last = min(pages), max(pages)
            images = convert_from_path(str(pdf_path), dpi=self.dpi, first_page=first, last_page=last)
        except Exception as exc:
            raise RuntimeError(
                "Не удалось отрендерить PDF через pdf2image. "
                "Убедитесь, что установлен Poppler (poppler-utils)."
            ) from exc

        by_page = {first + offset: image for offset, image in enumerate(images)}
        result: list[PageImage] = []
        for page_number in pages:
            image = by_page[page_number].convert("RGB")
            image = resize_to_width(image, self.width)
            result.append(PageImage(page_number, image.width, image.height, image))
        return result
