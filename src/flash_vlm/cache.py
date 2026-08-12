"""Кеширование промежуточных результатов распознавания страниц.

Позволяет при сбое не пересчитывать весь документ заново: результаты каждой
страницы сохраняются на диск и подхватываются при повторном запуске.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .utils import ensure_dir

__all__ = ["PageCache"]


class PageCache:
    """Дисковый кеш результатов по страницам (JSON-файлы)."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def _path(self, job_id: str, page: int) -> Path:
        return self.cache_dir / job_id / f"page_{page:04d}.json"

    def get(self, job_id: str, page: int) -> dict | None:
        """Возвращает сохранённый результат страницы или ``None``."""
        if not self.enabled:
            return None
        path = self._path(job_id, page)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, job_id: str, page: int, result: dict) -> None:
        """Сохраняет результат страницы."""
        if not self.enabled:
            return
        path = self._path(job_id, page)
        ensure_dir(path.parent)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    def clear(self, job_id: str) -> None:
        """Удаляет кеш конкретной задачи."""
        shutil.rmtree(self.cache_dir / job_id, ignore_errors=True)
