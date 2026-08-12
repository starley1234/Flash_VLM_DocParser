"""Интеграция с LM Studio через OpenAI-совместимый API.

LM Studio открывает эндпоинт ``http://localhost:1234/v1``. Для vision-запросов
изображение передаётся в формате base64 data URL внутри ``chat.completions``.
"""

from __future__ import annotations

import inspect

import httpx
from openai import AsyncOpenAI
from PIL import Image

from .utils import image_to_data_url
from .vlm import VisionClient

__all__ = ["LmStudioClient", "LmStudioError"]


class LmStudioError(RuntimeError):
    """Ошибка взаимодействия с LM Studio."""


class LmStudioClient(VisionClient):
    """Асинхронный клиент LM Studio (OpenAI-совместимый API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        timeout: float = 300.0,
        image_format: str = "PNG",
    ) -> None:
        base_url = base_url.rstrip("/")
        self.base_url = base_url
        self.image_format = "JPEG" if image_format.upper() in {"JPEG", "JPG"} else "PNG"
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def list_models(self) -> list[str]:
        """Запрашивает список моделей у LM Studio (``GET /v1/models``)."""
        response = await self._http.get("/models")
        response.raise_for_status()
        payload = response.json()
        models: list[str] = []
        for item in payload.get("data", []):
            model_id = item.get("id") if isinstance(item, dict) else item
            if model_id:
                models.append(model_id)
        return models

    async def ocr_image(
        self,
        image: Image.Image,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Отправляет изображение страницы в VLM и возвращает Markdown-текст."""
        data_url = image_to_data_url(image, self.image_format)
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Вот изображение страницы документа:"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = getattr(response.choices[0].message, "content", None)
        return (content or "").strip()

    async def close(self) -> None:
        """Закрывает HTTP-клиенты (совместимо с openai 1.x и 3.x)."""
        await self._http.aclose()
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
