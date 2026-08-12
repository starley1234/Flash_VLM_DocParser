"""Фабрика компонентов: клиент VLM и конвейер."""

from __future__ import annotations

from .config import Settings, get_settings

__all__ = ["build_client", "build_pipeline"]


def build_client(settings: Settings | None = None):
    """Создаёт клиента VLM в зависимости от настроек."""
    settings = settings or get_settings()
    if settings.fake_vlm:
        from .vlm import FakeVisionClient

        return FakeVisionClient()
    from .lmstudio import LmStudioClient

    return LmStudioClient(
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
        timeout=settings.request_timeout,
        image_format=settings.image_format,
    )


def build_pipeline(settings: Settings | None = None):
    """Создаёт конвейер PDF→Markdown с клиентом VLM."""
    from .pipeline import Pipeline

    settings = settings or get_settings()
    return Pipeline(settings, build_client(settings))
