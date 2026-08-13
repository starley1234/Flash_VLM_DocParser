"""Pydantic-модели: результаты распознавания, задачи API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PageResult(BaseModel):
    """Результат распознавания одной страницы."""

    page: int
    markdown: str = ""
    ok: bool = True
    error: str | None = None
    cached: bool = False
    duration_ms: int = 0
    model: str | None = None
    # Изображение страницы (после ресайза), сохранённое на диск для контроля.
    image_path: str | None = Field(default=None, exclude=True)
    image_width: int | None = None
    image_height: int | None = None


class ConversionResult(BaseModel):
    """Итоговый результат конвертации документа."""

    markdown: str = ""
    output_path: str | None = None
    pages_dir: str | None = None  # каталог с изображениями страниц
    pages_total: int = 0
    pages_ok: int = 0
    pages_failed: int = 0
    cached_pages: int = 0
    duration_ms: int = 0
    model: str | None = None
    pages: list[PageResult] = Field(default_factory=list)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Job(BaseModel):
    """Задача конвертации в HTTP API."""

    id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    total: int = 0
    done: int = 0
    current_page: int = 0
    # Номера страниц, запланированных к обработке (для иерархии в UI).
    page_numbers: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: str | None = None
    pages_dir: str | None = None
    result: ConversionResult | None = None
    # Готовые страницы (потоково пополняются по мере распознавания).
    completed_pages: list[PageResult] = Field(default_factory=list)

    @property
    def progress(self) -> float:
        """Прогресс в процентах (0–100)."""
        if self.total <= 0:
            return 0.0
        return round(100.0 * self.done / self.total, 1)
