"""Конфигурация приложения (переменные окружения / .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки Flash-VLM DocParser.

    Все поля можно переопределить переменными окружения с префиксом
    ``FLASH_VLM_`` (например ``FLASH_VLM_IMAGE_WIDTH=1024``) или файлом ``.env``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FLASH_VLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LM Studio -----------------------------------------------------------
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    model: str = ""  # пустая строка = автоопределение первой доступной модели
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout: float = 300.0

    # --- Рендеринг страниц ----------------------------------------------------
    image_width: int = 512  # принудительная ширина изображения для VLM
    render_dpi: int = 200  # DPI рендеринга до ресайза
    render_backend: str = "pymupdf"  # "pymupdf" или "pdf2image"
    image_format: str = "png"  # "png" или "jpeg"

    # --- Пайплайн / очередь ----------------------------------------------------
    concurrency: int = 1  # параллельные запросы к VLM (1 для 16 ГБ VRAM)
    max_retries: int = 3
    retry_backoff: float = 2.0
    queue_size: int = 32

    # --- Вывод и кеширование ----------------------------------------------------
    cache_dir: Path = Path(".cache/flash_vlm")
    output_dir: Path = Path("output")
    use_cache: bool = True
    page_separator: str = "\n\n"
    page_markers: bool = False  # добавлять комментарии <!-- Page N -->
    add_frontmatter: bool = True  # добавлять YAML-frontmatter в итоговый .md
    save_page_images: bool = True  # сохранять уменьшенные изображения страниц на диск

    # --- HTTP API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 100

    # --- Демо-режим ---------------------------------------------------------------
    fake_vlm: bool = False  # встроенная заглушка VLM для тестов и демо без LM Studio

    @field_validator("page_separator")
    @classmethod
    def _unescape_separator(cls, value: str) -> str:
        """Позволяет задавать переносы строк в переменных окружения как \\n."""
        return value.replace("\\n", "\n")


@lru_cache
def get_settings() -> Settings:
    """Возвращает кешированный экземпляр настроек."""
    return Settings()
