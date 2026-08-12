"""Консольный интерфейс (CLI) Flash-VLM DocParser.

Команды:
- ``flash-vlm convert PDF`` — конвертировать PDF в Markdown;
- ``flash-vlm health`` / ``flash-vlm models`` — диагностика LM Studio;
- ``flash-vlm serve`` — запустить HTTP API;
- ``flash-vlm mcp`` — запустить MCP-сервер (stdio);
- ``flash-vlm ui`` — запустить Streamlit UI.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .config import get_settings
from .factory import build_client, build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flash-vlm",
        description="Конвертация PDF в Markdown через локальные VLM (LM Studio).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # convert -----------------------------------------------------------
    p_convert = sub.add_parser("convert", help="Конвертировать PDF в Markdown")
    p_convert.add_argument("pdf", type=Path, help="Путь к PDF-файлу")
    p_convert.add_argument("-o", "--output", type=Path, default=None, help="Путь к .md (или каталог)")
    p_convert.add_argument("--pages", default=None, help="Страницы, например '1-5,8' (по умолчанию все)")
    p_convert.add_argument(
        "--model",
        default=None,
        help="Модель VLM (по умолчанию берётся из FLASH_VLM_MODEL, иначе автоопределение)",
    )
    p_convert.add_argument("--width", type=int, default=None, help="Ширина изображения (по умолчанию 512)")
    p_convert.add_argument("--prompt-file", type=Path, default=None, help="Файл с системным промптом")
    p_convert.add_argument("--page-markers", action="store_true", help="Добавить комментарии <!-- Page N -->")
    p_convert.add_argument("--no-cache", action="store_true", help="Отключить кеширование страниц")
    p_convert.add_argument("--no-frontmatter", action="store_true", help="Не добавлять frontmatter")
    p_convert.add_argument("--fake", action="store_true", help="Использовать заглушку VLM (без LM Studio)")

    # serve -------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="Запустить HTTP API")
    p_serve.add_argument("--host", default=None, help="Хост API (по умолчанию 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None, help="Порт API (по умолчанию 8000)")

    # mcp ---------------------------------------------------------------
    sub.add_parser("mcp", help="Запустить MCP-сервер (stdio)")

    # ui ----------------------------------------------------------------
    p_ui = sub.add_parser("ui", help="Запустить Streamlit UI")
    p_ui.add_argument("--host", default="0.0.0.0", help="Хост UI")
    p_ui.add_argument("--port", type=int, default=8501, help="Порт UI")

    # health / models -----------------------------------------------------
    sub.add_parser("health", help="Проверить доступность LM Studio")
    sub.add_parser("models", help="Показать список моделей LM Studio")

    return parser


async def _convert(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.fake:
        settings = settings.model_copy(update={"fake_vlm": True})
    if args.width is not None:
        settings = settings.model_copy(update={"image_width": args.width})
    if args.no_cache:
        settings = settings.model_copy(update={"use_cache": False})
    if args.no_frontmatter:
        settings = settings.model_copy(update={"add_frontmatter": False})

    prompt = None
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    pipeline = build_pipeline(settings)

    def on_progress(done: int, total: int, current_page: int) -> None:
        sys.stdout.write(f"\rОбработка страницы {current_page} из {total}... ")
        sys.stdout.flush()

    try:
        result = await pipeline.convert(
            args.pdf,
            pages=args.pages,
            model=args.model,
            prompt=prompt,
            output_path=args.output,
            page_markers=args.page_markers,
            on_progress=on_progress,
        )
    finally:
        await pipeline.client.close()

    sys.stdout.write("\n")
    print(f"✅ Готово: {result.output_path}")
    print(f"   Страниц: {result.pages_total} (ok={result.pages_ok}, ошибок={result.pages_failed}, "
          f"из кеша={result.cached_pages})")
    print(f"   Модель: {result.model} · Время: {result.duration_ms / 1000:.1f} c")
    return 0


async def _diagnose(command: str) -> int:
    settings = get_settings()
    client = build_client(settings)
    try:
        if command == "health":
            info = await client.check_health()
            info["base_url"] = settings.lmstudio_base_url
            print(f"LM Studio: {settings.lmstudio_base_url}")
            print(f"Доступен: {'✅ да' if info['ok'] else '❌ нет'}")
            if info["models"]:
                print("Модели:")
                for m in info["models"]:
                    print(f"  - {m}")
            if info.get("error"):
                print(f"Ошибка: {info['error']}")
            return 0 if info["ok"] else 1
        if command == "models":
            models = await client.list_models()
            for m in models:
                print(m)
            if not models:
                print("(модели не найдены)")
            return 0
    finally:
        await client.close()
    return 1


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port)
    return 0


def _run_mcp() -> int:
    from .mcp.server import main as mcp_main

    return mcp_main()


def _run_ui(args: argparse.Namespace) -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Установите Streamlit: pip install 'flash-vlm-docparser[ui]'") from exc

    module = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(module),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    stcli.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "convert":
            return asyncio.run(_convert(args))
        if args.command in {"health", "models"}:
            return asyncio.run(_diagnose(args.command))
        if args.command == "serve":
            return _serve(args)
        if args.command == "mcp":
            return _run_mcp()
        if args.command == "ui":
            return _run_ui(args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
