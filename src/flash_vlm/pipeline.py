"""Оркестратор конвейера: PDF → страницы → очередь → VLM → Markdown."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from . import prompts
from .assembler import MarkdownAssembler
from .cache import PageCache
from .config import Settings
from .pdf_processor import PageImage, PdfProcessor
from .queue_manager import TaskQueue
from .schemas import ConversionResult, PageResult
from .utils import ensure_dir, now_ms, pages_dir_for, parse_pages, sha1_hex
from .vlm import VisionClient

logger = logging.getLogger(__name__)

# (done, total, page_result) — вызывается после каждой обработанной страницы.
# Может быть как синхронной, так и асинхронной функцией.
ProgressCallback = Callable[[int, int, PageResult], Awaitable[None] | None]

__all__ = ["Pipeline", "ProgressCallback"]


class Pipeline:
    """Конвейер преобразования PDF в Markdown."""

    def __init__(self, settings: Settings, client: VisionClient) -> None:
        self.settings = settings
        self.client = client
        self.pdf = PdfProcessor(
            dpi=settings.render_dpi,
            width=settings.image_width,
            backend=settings.render_backend,
        )

    async def convert(
        self,
        pdf_path: str | Path,
        *,
        pages: object = None,
        model: str | None = None,
        prompt: str | None = None,
        output_path: str | Path | None = None,
        page_markers: bool | None = None,
        use_cache: bool | None = None,
        job_id: str | None = None,
        source_name: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        """Конвертирует PDF в Markdown.

        Args:
            pdf_path: путь к PDF-файлу.
            pages: спецификация страниц (``None``/``"1-5,8"``/список).
            model: идентификатор модели. Приоритет: явный параметр → настройка
                ``FLASH_VLM_MODEL`` → автоопределение первой доступной модели.
            prompt: системный промпт (``None`` = промпт по умолчанию).
            output_path: путь к итоговому ``.md`` (или каталог для него).
            page_markers: добавлять комментарии ``<!-- Page N -->``.
            use_cache: использовать кеш страниц (``None`` = из настроек).
            job_id: идентификатор задачи для кеша (``None`` = вычислить).
            source_name: имя исходного файла для frontmatter (``None`` = имя pdf_path).
            on_progress: колбэк ``(done, total, page_result)`` после каждой страницы.
        """
        started_at = now_ms()
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        prompt = prompt or prompts.DEFAULT_SYSTEM_PROMPT
        # Приоритет модели: явный параметр -> настройка FLASH_VLM_MODEL -> автоопределение.
        model = await self.client.resolve_model(model or self.settings.model or None)
        logger.info("Модель для распознавания: %s", model)

        out_path = self._resolve_output_path(pdf_path, output_path)

        # Уменьшенные изображения страниц сохраняются на диск, чтобы можно было
        # проверить, что ресайз действительно применился (см. /jobs/{id}/page/{n}).
        pages_dir: Path | None = None
        if self.settings.save_page_images:
            pages_dir = pages_dir_for(out_path)
            ensure_dir(pages_dir)

        total_pages = self.pdf.page_count(pdf_path)
        page_numbers = parse_pages(pages, total_pages)
        images = self.pdf.render(pdf_path, pages=page_numbers)

        if pages_dir is not None:
            for image in images:
                image.image.save(pages_dir / f"page_{image.page_number:04d}.png")

        job_id = job_id or self._compute_job_id(pdf_path, model, prompt, page_numbers)
        cache = PageCache(
            self.settings.cache_dir,
            enabled=self.settings.use_cache if use_cache is None else use_cache,
        )

        results: dict[int, PageResult] = {}
        done = 0

        async def process_page(image: PageImage) -> None:
            nonlocal done

            cached = cache.get(job_id, image.page_number)
            if cached is not None:
                result = PageResult(**cached)
                result.cached = True
            else:
                page_started = now_ms()
                try:
                    text = await self._ocr_with_retries(image, prompt, model)
                    result = PageResult(
                        page=image.page_number,
                        markdown=text,
                        ok=True,
                        duration_ms=now_ms() - page_started,
                        model=model,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = PageResult(
                        page=image.page_number,
                        markdown="",
                        ok=False,
                        error=str(exc),
                        duration_ms=now_ms() - page_started,
                        model=model,
                    )
                cache.set(job_id, image.page_number, result.model_dump())

            # Привязываем информацию об уменьшенном изображении страницы.
            result.image_width = image.width
            result.image_height = image.height
            if pages_dir is not None:
                result.image_path = str(pages_dir / f"page_{image.page_number:04d}.png")

            results[image.page_number] = result
            done += 1
            if on_progress is not None:
                await self._maybe_await(on_progress(done, len(images), result))

        queue = TaskQueue(process_page, concurrency=self.settings.concurrency, maxsize=self.settings.queue_size)
        async with queue:
            for image in images:
                await queue.put(image)
            await queue.join()

        ordered = [results[image.page_number] for image in images]
        assembler = MarkdownAssembler(
            page_separator=self.settings.page_separator,
            page_markers=self.settings.page_markers if page_markers is None else page_markers,
            frontmatter=self.settings.add_frontmatter,
        )
        markdown = assembler.assemble(ordered, source=source_name or pdf_path.name, model=model)

        ensure_dir(out_path.parent)
        out_path.write_text(markdown, encoding="utf-8")

        return ConversionResult(
            markdown=markdown,
            output_path=str(out_path),
            pages_dir=str(pages_dir) if pages_dir is not None else None,
            pages_total=len(images),
            pages_ok=sum(1 for p in ordered if p.ok),
            pages_failed=sum(1 for p in ordered if not p.ok),
            cached_pages=sum(1 for p in ordered if p.cached),
            duration_ms=now_ms() - started_at,
            model=model,
            pages=ordered,
        )

    # ------------------------------------------------------------------
    async def _ocr_with_retries(self, image: PageImage, prompt: str, model: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                return await self.client.ocr_image(
                    image.image,
                    prompt,
                    model,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                await asyncio.sleep(self.settings.retry_backoff ** attempt)
        raise last_error  # type: ignore[misc]

    def _resolve_output_path(self, pdf_path: Path, output_path: str | Path | None) -> Path:
        if output_path:
            out = Path(output_path)
            if out.suffix.lower() != ".md":
                out = out / f"{pdf_path.stem}.md"
        else:
            out = pdf_path.with_suffix(".md")
        return out

    def _compute_job_id(self, pdf_path: Path, model: str, prompt: str, pages: list[int]) -> str:
        stat = pdf_path.stat()
        return sha1_hex(
            pdf_path.resolve(),
            stat.st_mtime_ns,
            stat.st_size,
            self.settings.image_width,
            self.settings.render_dpi,
            self.settings.render_backend,
            model,
            prompt,
            pages,
        )

    @staticmethod
    async def _maybe_await(value: Awaitable[None] | None) -> None:
        if value is not None and inspect.isawaitable(value):
            await value
