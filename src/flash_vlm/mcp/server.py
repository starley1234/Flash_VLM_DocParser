"""MCP-сервер Flash-VLM DocParser.

Экспонирует инструменты для OCR и конвертации PDF→Markdown через локальную VLM
(LM Studio). Запускается как stdio-сервер (для Claude Desktop, Cursor и др.):

    flash-vlm-mcp

Также доступен поверх HTTP — монтируется в FastAPI-приложение по адресу ``/mcp``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context

from .. import __version__
from ..config import Settings, get_settings
from ..factory import build_pipeline
from ..pipeline import Pipeline
from ..utils import decode_base64_image

__all__ = ["create_mcp_server", "main"]

SERVER_INSTRUCTIONS = (
    "Сервер Flash-VLM DocParser выполняет OCR изображений и конвертацию PDF "
    "в Markdown через локальные визуально-языковые модели (LM Studio)."
)


async def _safe_report_progress(ctx: Context | None, done: int, total: int, current_page: int) -> None:
    """Сообщает прогресс, если контекст доступен (вне запроса — тихо пропускает)."""
    if ctx is None:
        return
    try:
        await ctx.report_progress(done, total, f"Обработка страницы {current_page} из {total}")
    except Exception:  # noqa: BLE001 — прогресс не должен ронять инструмент
        pass


def create_mcp_server(settings: Settings | None = None, pipeline: Pipeline | None = None) -> MCPServer:
    """Создаёт MCP-сервер с инструментами OCR и конвертации PDF."""
    settings = settings or get_settings()
    pipeline = pipeline or build_pipeline(settings)

    server = MCPServer(
        name="flash-vlm-docparser",
        version=__version__,
        title="Flash-VLM DocParser",
        instructions=SERVER_INSTRUCTIONS,
    )

    @server.tool(
        description=(
            "Проверяет доступность LM Studio и возвращает список загруженных моделей "
            "с их идентификаторами."
        )
    )
    async def check_health() -> str:
        info = await pipeline.client.check_health()
        info["base_url"] = settings.lmstudio_base_url
        lines = [f"LM Studio: {info['base_url']}", f"Доступен: {'да' if info['ok'] else 'нет'}"]
        if info["models"]:
            lines.append("Модели:")
            lines.extend(f"- {m}" for m in info["models"])
        if info.get("error"):
            lines.append(f"Ошибка: {info['error']}")
        return "\n".join(lines)

    @server.tool(description="Возвращает список моделей, доступных в LM Studio.")
    async def list_models() -> list[str]:
        return await pipeline.client.list_models()

    @server.tool(
        description=(
            "Распознаёт текст на изображении (base64 или data URL) и возвращает его "
            "в формате Markdown. Необязательно можно указать свой системный промпт и модель."
        )
    )
    async def ocr_image(
        image_base64: str,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        image = decode_base64_image(image_base64)
        from .. import prompts as prompt_lib

        chosen_model = await pipeline.client.resolve_model(model)
        return await pipeline.client.ocr_image(
            image,
            prompt or prompt_lib.DEFAULT_SYSTEM_PROMPT,
            chosen_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    @server.tool(
        description=(
            "Конвертирует PDF-файл (по пути на диске) в Markdown через VLM и сохраняет "
            "результат в .md рядом с исходным файлом (или по указанному пути). Возвращает "
            "итоговый Markdown с путём сохранения."
        )
    )
    async def convert_pdf_to_markdown(
        pdf_path: str,
        pages: str | None = None,
        model: str | None = None,
        output_path: str | None = None,
        page_markers: bool = False,
        ctx: Context | None = None,
    ) -> str:
        def on_progress(done: int, total: int, current_page: int) -> None:
            asyncio.get_running_loop().create_task(
                _safe_report_progress(ctx, done, total, current_page)
            )

        result = await pipeline.convert(
            Path(pdf_path),
            pages=pages,
            model=model,
            output_path=Path(output_path) if output_path else None,
            page_markers=page_markers,
            on_progress=on_progress,
        )
        header = (
            f"✅ Конвертация завершена. Сохранено в: {result.output_path}\n"
            f"Модель: {result.model}\n"
            f"Страниц: {result.pages_total} (ok={result.pages_ok}, ошибок={result.pages_failed})\n\n"
        )
        return header + result.markdown

    return server


def main() -> int:
    """Запускает MCP-сервер в режиме stdio."""
    server = create_mcp_server()
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
