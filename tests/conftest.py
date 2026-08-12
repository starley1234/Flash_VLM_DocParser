from __future__ import annotations

import pymupdf
import pytest

from flash_vlm.config import Settings


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    """Создаёт небольшой тестовый PDF из трёх страниц."""
    path = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    doc = pymupdf.open()
    for index in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 90), f"Document title page {index + 1}", fontsize=16)
        page.insert_text((72, 130), "Some paragraph of text.", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def fake_settings(tmp_path):
    """Настройки в демо-режиме (без LM Studio)."""
    return Settings(
        fake_vlm=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        page_markers=True,
    )
