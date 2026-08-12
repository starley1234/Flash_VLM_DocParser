from __future__ import annotations

import asyncio
from pathlib import Path

from flash_vlm.config import Settings
from flash_vlm.factory import build_pipeline


def test_convert_end_to_end(sample_pdf, tmp_path):
    settings = Settings(
        fake_vlm=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        page_markers=True,
        add_frontmatter=True,
    )
    pipeline = build_pipeline(settings)
    progress: list[tuple[int, int, int]] = []

    result = asyncio.run(
        pipeline.convert(sample_pdf, on_progress=lambda done, total, cur: progress.append((done, total, cur)))
    )

    assert result.pages_total == 3
    assert result.pages_ok == 3
    assert result.pages_failed == 0
    assert result.cached_pages == 0
    assert result.model == "fake-vision-model"
    assert "<!-- Page 1 -->" in result.markdown
    assert result.markdown.startswith("---\n")
    assert Path(result.output_path).exists()
    assert len(progress) == 3
    assert progress[-1] == (3, 3, 3)


def test_convert_uses_cache_on_second_run(sample_pdf, tmp_path):
    settings = Settings(
        fake_vlm=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        page_markers=False,
    )
    pipeline = build_pipeline(settings)

    first = asyncio.run(pipeline.convert(sample_pdf))
    second = asyncio.run(pipeline.convert(sample_pdf))

    assert first.cached_pages == 0
    assert second.cached_pages == 3
    assert second.markdown == first.markdown


def test_convert_partial_pages(sample_pdf, tmp_path):
    settings = Settings(fake_vlm=True, cache_dir=tmp_path / "cache", output_dir=tmp_path / "output")
    pipeline = build_pipeline(settings)
    result = asyncio.run(pipeline.convert(sample_pdf, pages="1,3"))
    assert result.pages_total == 2
    assert [p.page for p in result.pages] == [1, 3]


def test_convert_uses_settings_model_by_default(sample_pdf, tmp_path):
    """Модель из FLASH_VLM_MODEL (settings.model) должна использоваться,
    если явная модель не передана."""
    settings = Settings(
        fake_vlm=True,
        model="my-configured-model",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )
    pipeline = build_pipeline(settings)
    result = asyncio.run(pipeline.convert(sample_pdf))
    assert result.model == "my-configured-model"


def test_convert_explicit_model_overrides_settings(sample_pdf, tmp_path):
    settings = Settings(
        fake_vlm=True,
        model="my-configured-model",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )
    pipeline = build_pipeline(settings)
    result = asyncio.run(pipeline.convert(sample_pdf, model="explicit-model"))
    assert result.model == "explicit-model"


def test_convert_missing_file(tmp_path):
    settings = Settings(fake_vlm=True, cache_dir=tmp_path / "cache", output_dir=tmp_path / "output")
    pipeline = build_pipeline(settings)
    try:
        asyncio.run(pipeline.convert(tmp_path / "nope.pdf"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
