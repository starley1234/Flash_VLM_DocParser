from __future__ import annotations

import pytest

from flash_vlm.pdf_processor import PdfProcessor, resize_to_width


def test_page_count(sample_pdf):
    processor = PdfProcessor()
    assert processor.page_count(sample_pdf) == 3


def test_render_resizes_to_width(sample_pdf):
    processor = PdfProcessor(dpi=100, width=512)
    images = processor.render(sample_pdf, pages="1-2")
    assert len(images) == 2
    assert [im.page_number for im in images] == [1, 2]
    for image in images:
        assert image.width == 512
        assert image.height > 0


def test_render_all_pages(sample_pdf):
    processor = PdfProcessor(dpi=72, width=256)
    images = processor.render(sample_pdf)
    assert len(images) == 3


def test_resize_keeps_aspect_ratio():
    from PIL import Image

    image = Image.new("RGB", (1000, 500))
    resized = resize_to_width(image, 500)
    assert resized.size == (500, 250)


def test_invalid_backend():
    with pytest.raises(ValueError):
        PdfProcessor(backend="nope")
