"""Абстракция визуально-языковой модели (VLM).

Реализации:
- ``LmStudioClient`` — реальная модель через LM Studio;
- ``FakeVisionClient`` — детерминированная заглушка для тестов и демо.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

__all__ = ["VisionClient", "FakeVisionClient"]


class VisionClient(ABC):
    """Интерфейс клиента VLM (OpenAI-совместимый chat completions + vision)."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Возвращает идентификаторы доступных моделей."""

    @abstractmethod
    async def ocr_image(
        self,
        image: Image.Image,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Распознаёт изображение страницы и возвращает Markdown-текст."""

    async def resolve_model(self, preferred: str | None = None) -> str:
        """Выбирает модель: переданную явно или первую доступную."""
        if preferred:
            return preferred
        models = await self.list_models()
        if not models:
            raise RuntimeError(
                "В LM Studio не найдено ни одной модели. Загрузите модель (например, "
                "Llama-3.2-Vision или Qwen2-VL) и повторите попытку."
            )
        return models[0]

    async def check_health(self) -> dict:
        """Проверяет доступность VLM-бэкенда."""
        try:
            models = await self.list_models()
            return {"ok": True, "models": models, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "models": [], "error": str(exc)}

    async def close(self) -> None:
        """Освобождает ресурсы (HTTP-клиенты и т.п.)."""


class FakeVisionClient(VisionClient):
    """Детерминированная заглушка VLM для тестов и демонстрации без LM Studio."""

    name = "fake-vision-model"

    async def list_models(self) -> list[str]:
        return [self.name]

    async def ocr_image(
        self,
        image: Image.Image,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        return (
            "## Страница (демо-режим)\n\n"
            f"- Размер изображения: {image.width}×{image.height} px\n"
            f"- Модель: {model or self.name}\n\n"
            "Это пример вывода в режиме `FAKE_VLM`. Подключите LM Studio, "
            "чтобы получать настоящий распознанный текст."
        )
