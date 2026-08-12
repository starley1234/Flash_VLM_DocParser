"""Вспомогательные утилиты: base64, хеши, время, парсинг страниц."""

from __future__ import annotations

import base64
import hashlib
import io
import time
from pathlib import Path

from PIL import Image


def now_ms() -> int:
    """Текущее время в миллисекундах (unix)."""
    return int(time.time() * 1000)


def ensure_dir(path: Path) -> Path:
    """Создаёт каталог (включая родительские) и возвращает его."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    """Кодирует PIL-изображение в data URL для OpenAI-совместимого API."""
    fmt = "JPEG" if image_format.upper() in {"JPEG", "JPG"} else "PNG"
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def decode_base64_image(data: str | bytes) -> Image.Image:
    """Декодирует base64-строку (возможно с data:image/... префиксом) в PIL.Image."""
    if isinstance(data, bytes):
        raw = data
    else:
        text = data.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[1]
        raw = base64.b64decode(text)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def sha1_hex(*parts: object) -> str:
    """SHA-1 хеш от произвольного набора значений (для ключей кеша)."""
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def parse_pages(spec: object, total: int) -> list[int]:
    """Преобразует спецификацию страниц в список 1-based индексов.

    Поддерживает: ``None`` (все страницы), ``"1-5,8,10-12"``, ``[2, 3, 4]``,
    одиночное число ``5``.
    """
    if spec is None or spec == "":
        return list(range(1, total + 1))

    if isinstance(spec, bool):  # защита от bool, который является int
        raise ValueError("Некорректная спецификация страниц")

    if isinstance(spec, int):
        raw: list[int] = [spec]
    elif isinstance(spec, (list, tuple, set)):
        raw = [int(p) for p in spec]
    else:
        raw = []
        for chunk in str(spec).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, _, b = chunk.partition("-")
                a, b = a.strip(), b.strip()
                if not a or not b:
                    raise ValueError(f"Некорректный диапазон страниц: {chunk!r}")
                start, end = int(a), int(b)
                if start > end:
                    raise ValueError(f"Некорректный диапазон страниц: {chunk!r}")
                raw.extend(range(start, end + 1))
            else:
                raw.append(int(chunk))

    pages: list[int] = []
    seen: set[int] = set()
    for page in raw:
        if page < 1 or page > total:
            raise ValueError(f"Страница {page} вне диапазона 1–{total}")
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages
