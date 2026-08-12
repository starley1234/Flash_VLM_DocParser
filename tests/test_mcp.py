from __future__ import annotations

import asyncio
import base64
import io

from PIL import Image

from flash_vlm.config import Settings
from flash_vlm.mcp.server import create_mcp_server


def _make_server(tmp_path):
    settings = Settings(
        fake_vlm=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        page_markers=True,
    )
    return create_mcp_server(settings)


def test_tools_are_registered(tmp_path):
    server = _make_server(tmp_path)
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert {"check_health", "list_models", "ocr_image", "convert_pdf_to_markdown"} <= names


def test_list_models(tmp_path):
    server = _make_server(tmp_path)
    result = asyncio.run(server.call_tool("list_models", {}))
    assert result.structured_content["result"] == ["fake-vision-model"]


def test_check_health(tmp_path):
    server = _make_server(tmp_path)
    result = asyncio.run(server.call_tool("check_health", {}))
    text = result.structured_content["result"]
    assert "Доступен: да" in text


def test_ocr_image(tmp_path):
    server = _make_server(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (128, 64), "white").save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = asyncio.run(server.call_tool("ocr_image", {"image_base64": b64}))
    markdown = result.structured_content["result"]
    assert "128×64" in markdown  # размер изображения из заглушки


def test_convert_pdf(tmp_path, sample_pdf):
    server = _make_server(tmp_path)
    result = asyncio.run(
        server.call_tool("convert_pdf_to_markdown", {"pdf_path": str(sample_pdf), "page_markers": True})
    )
    markdown = result.structured_content["result"]
    assert "Сохранено в:" in markdown
    assert "<!-- Page 1 -->" in markdown
