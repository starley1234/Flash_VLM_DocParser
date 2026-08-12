"""Очередь задач на базе ``asyncio.Queue``.

Позволяет загружать сразу несколько документов и последовательно отправлять
страницы на VLM с ограниченным числом параллельных воркеров — чтобы не
переполнить видеопамять (для 16 ГБ VRAM по умолчанию ``concurrency=1``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")

__all__ = ["TaskQueue"]


class TaskQueue:
    """Простой менеджер очереди задач с фиксированным пулом воркеров."""

    def __init__(
        self,
        handler: Callable[[ItemT], Awaitable[Any]],
        concurrency: int = 1,
        maxsize: int = 0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency должен быть >= 1")
        self._queue: asyncio.Queue[ItemT] = asyncio.Queue(maxsize=maxsize)
        self._handler = handler
        self._concurrency = concurrency
        self._workers: list[asyncio.Task[None]] = []

    @property
    def concurrency(self) -> int:
        return self._concurrency

    async def put(self, item: ItemT) -> None:
        """Добавляет задачу в очередь (await, если очередь заполнена)."""
        await self._queue.put(item)

    async def start(self) -> None:
        """Запускает пул воркеров."""
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._run_worker(), name=f"vlm-worker-{index}")
            for index in range(self._concurrency)
        ]

    async def join(self) -> None:
        """Дожидается завершения всех поставленных задач."""
        await self._queue.join()

    async def stop(self) -> None:
        """Останавливает воркеров."""
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def _run_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._handler(item)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # Обработчик должен ловить собственные ошибки; здесь — страховка,
                # чтобы одна упавшая задача не остановила остальные.
                logger.exception("Необработанная ошибка при обработке задачи")
            finally:
                self._queue.task_done()

    async def __aenter__(self) -> "TaskQueue":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
