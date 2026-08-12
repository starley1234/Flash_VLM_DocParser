"""Flash-VLM DocParser.

Лёгкий конвейер для высокоточного преобразования PDF в Markdown
с помощью локальных визуально-языковых моделей (VLM), развёрнутых
через LM Studio (OpenAI-совместимый API).

Основные компоненты:
- ``flash_vlm.pdf_processor``  — рендеринг PDF в изображения + умный ресайз;
- ``flash_vlm.queue_manager``  — очередь задач на базе asyncio.Queue;
- ``flash_vlm.lmstudio``       — интеграция с LM Studio (OpenAI-совместимый API);
- ``flash_vlm.assembler``      — сборка итогового Markdown;
- ``flash_vlm.api``            — HTTP REST API (FastAPI);
- ``flash_vlm.mcp``            — MCP-сервер (stdio + streamable HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
