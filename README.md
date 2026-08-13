# Flash-VLM DocParser

Лёгкое Python-приложение для **высокоточного преобразования PDF в Markdown** с
помощью локальных визуально-языковых моделей (VLM) — *Llama-3.2-Vision*,
*Qwen2-VL*, *Moondream2* и др., — развёрнутых через [LM Studio](https://lmstudio.ai/).

Проект реализует концепцию **«VLM-OCR Pipeline»**: PDF разбивается на страницы,
сжимается до заданной ширины, страницы ставятся в очередь и последовательно
отправляются в VLM, а результат склеивается в единый `.md`-файл.

> Оптимизировано под 16 ГБ VRAM: по умолчанию запросы к VLM выполняются
> **последовательно** (concurrency = 1), чтобы не переполнить видеопамять.

---

## Возможности

- 🖼 **PDF-процессор**: рендеринг через PyMuPDF (по умолчанию) или `pdf2image`
  (Poppler) + «умный» ресайз до заданной ширины (по умолчанию 512px) с сохранением
  пропорций — экономит контекстное окно VLM.
- 🧵 **Очередь задач** на базе `asyncio.Queue` с настраиваемым числом воркеров.
- 🔌 **Интеграция с LM Studio** через OpenAI-совместимый API (`localhost:1234`),
  изображения передаются в base64.
- 📦 **Сборщик Markdown**: frontmatter, маркеры страниц, объединение в один файл.
- 💾 **Кеширование страниц**: промежуточные результаты сохраняются на диск — при
  сбое не нужно распознавать 100-страничный файл заново.
- 🌐 **HTTP REST API** (FastAPI) + мини-веб-интерфейс.
- 🔗 **MCP-сервер** (stdio и streamable HTTP) — инструменты OCR и конвертации для
  Claude Desktop / Cursor и других MCP-клиентов.
- 🖥 **Streamlit UI** (опционально).
- 🧪 **Демо-режим** (`FAKE_VLM`) — встроенная заглушка VLM для тестов и знакомства
  без LM Studio.

---

## Архитектура

```
PDF ──► PdfProcessor ──► asyncio.Queue ──► LmStudioClient ──► MarkdownAssembler ──► .md
         (рендеринг +        (TaskQueue,      (OpenAI-совм.      (frontmatter,         (файл)
          ресайз 512px)       concurrency)     API, base64)       склейка страниц)
                                  │
                                  └──► PageCache (диск, кеш страниц)
```

| Компонент | Модуль |
|---|---|
| PDF-процессор | `flash_vlm.pdf_processor` |
| Очередь задач | `flash_vlm.queue_manager` |
| Клиент LM Studio | `flash_vlm.lmstudio` |
| Кеш страниц | `flash_vlm.cache` |
| Сборщик Markdown | `flash_vlm.assembler` |
| Оркестратор | `flash_vlm.pipeline` |
| HTTP API | `flash_vlm.api` |
| MCP-сервер | `flash_vlm.mcp` |
| Streamlit UI | `flash_vlm.ui` |

---

## Требования

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) с запущенным локальным сервером
  (`http://localhost:1234/v1`) и загруженной VLM (например, `Llama-3.2-Vision`).
- Для опционального бэкенда `pdf2image` — [Poppler](https://poppler.freedesktop.org/)
  (`poppler-utils`). PyMuPDF работает без системных зависимостей.

## Установка

```bash
git clone https://github.com/starley1234/Flash_VLM_DocParser.git
cd Flash_VLM_DocParser
python -m venv .venv && source .venv/bin/activate
pip install -e .

# с Streamlit UI:
pip install -e ".[ui]"
```

---

## Быстрый старт (CLI)

```bash
# Проверка LM Studio
flash-vlm health
flash-vlm models

# Конвертация PDF → Markdown (результат рядом с исходником: file.md)
flash-vlm convert document.pdf

# Явно указать модель, ширину и страницы
flash-vlm convert document.pdf --model llama-3.2-vision --width 512 --pages "1-10"

# Сохранить в конкретный файл, добавить маркеры страниц
flash-vlm convert document.pdf -o result.md --page-markers

# Без LM Studio (демо-режим)
flash-vlm convert document.pdf --fake
```

Прогресс выводится в консоль: `Обработка страницы 5 из 20...`.

---

## HTTP API

```bash
# Запуск API (по умолчанию http://0.0.0.0:8000)
flash-vlm serve
# или
uvicorn flash_vlm.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Интерактивная документация: `http://localhost:8000/docs`, мини-интерфейс —
`http://localhost:8000/`.

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Состояние сервера и LM Studio |
| `GET` | `/models` | Список моделей LM Studio |
| `POST` | `/ocr` | OCR одного изображения (JSON `{image_base64, prompt?, model?}`) |
| `POST` | `/convert` | Загрузка PDF (multipart) → создание задачи |
| `GET` | `/jobs` | Список задач |
| `GET` | `/jobs/{id}` | Статус и прогресс задачи |
| `GET` | `/jobs/{id}/result` | Итоговый Markdown |
| `GET` | `/jobs/{id}/download` | Скачать `.md` |
| `DELETE` | `/jobs/{id}` | Отменить/удалить задачу |
| `*` | `/mcp` | MCP-сервер (streamable HTTP) |

### Примеры

```bash
# Конвертация PDF (асинхронная задача)
curl -F "file=@document.pdf" -F "model=" -F "pages=" http://localhost:8000/convert
# → {"job_id": "...", "status_url": "/jobs/...", ...}

# Прогресс
curl http://localhost:8000/jobs/<job_id>
# → {"status": "running", "current_page": 5, "done": 5, "total": 20, ...}

# Результат
curl http://localhost:8000/jobs/<job_id>/result

# Скачать Markdown
curl -OJ http://localhost:8000/jobs/<job_id>/download

# OCR одного изображения
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/png;base64,..."}'
```

---

## MCP-сервер

Сервер экспонирует инструменты:

- `check_health` — проверка LM Studio и список моделей;
- `list_models` — список моделей;
- `ocr_image(image_base64, prompt?, model?)` — OCR изображения → Markdown;
- `convert_pdf_to_markdown(pdf_path, pages?, model?, output_path?, page_markers?)` —
  конвертация PDF → Markdown (с сохранением `.md` на диск).

### Режим stdio (для Claude Desktop / Cursor)

```bash
flash-vlm-mcp
# или
python -m flash_vlm.mcp.server
```

Пример конфигурации `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "flash-vlm-docparser": {
      "command": "flash-vlm-mcp",
      "env": {
        "FLASH_VLM_LMSTUDIO_BASE_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

### Режим HTTP

При запущенном API сервер доступен по адресу `http://localhost:8000/mcp`
(streamable HTTP transport).

---

## Streamlit UI (опционально)

```bash
pip install -e ".[ui]"
flash-vlm ui            # http://localhost:8501
# или
streamlit run src/flash_vlm/ui/streamlit_app.py
```

---

## Конфигурация

Все настройки задаются переменными окружения с префиксом `FLASH_VLM_` или файлом
`.env` (см. [`.env.example`](.env.example)).

| Переменная | По умолчанию | Описание |
|---|---|---|
| `FLASH_VLM_LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Адрес LM Studio |
| `FLASH_VLM_LMSTUDIO_API_KEY` | `lm-studio` | API-ключ (LM Studio — любое значение) |
| `FLASH_VLM_MODEL` | *(пусто)* | Модель; пусто = автоопределение |
| `FLASH_VLM_IMAGE_WIDTH` | `512` | Ширина изображения для VLM |
| `FLASH_VLM_RENDER_DPI` | `200` | DPI рендеринга до ресайза |
| `FLASH_VLM_RENDER_BACKEND` | `pymupdf` | `pymupdf` или `pdf2image` |
| `FLASH_VLM_CONCURRENCY` | `1` | Параллельные запросы к VLM |
| `FLASH_VLM_MAX_RETRIES` | `3` | Повторы при ошибке |
| `FLASH_VLM_TEMPERATURE` | `0.0` | Температура генерации |
| `FLASH_VLM_MAX_TOKENS` | `4096` | Лимит токенов ответа |
| `FLASH_VLM_USE_CACHE` | `true` | Кеширование страниц |
| `FLASH_VLM_CACHE_DIR` | `.cache/flash_vlm` | Каталог кеша |
| `FLASH_VLM_OUTPUT_DIR` | `output` | Каталог результатов API |
| `FLASH_VLM_API_HOST` / `FLASH_VLM_API_PORT` | `0.0.0.0` / `8000` | API |
| `FLASH_VLM_FAKE_VLM` | `false` | Демо-режим без LM Studio |

---

## Docker

```bash
# REST API (http://localhost:8000)
docker compose up -d --build

# API + Streamlit UI (http://localhost:8501)
docker compose --profile ui up -d --build
```

Внутри контейнера LM Studio доступен по адресу
`http://host.docker.internal:1234/v1` (на Linux включён `host-gateway`). При
необходимости переопределите через `FLASH_VLM_LMSTUDIO_BASE_URL`:

```bash
FLASH_VLM_LMSTUDIO_BASE_URL=http://192.168.1.10:1234/v1 docker compose up -d --build
```

---

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

Тесты используют `FakeVisionClient` и не требуют запущенного LM Studio.

---

## Структура проекта

```
src/flash_vlm/
├── api/            # FastAPI REST API + MCP over HTTP
├── mcp/            # MCP-сервер (stdio / streamable HTTP)
├── ui/             # Streamlit UI
├── assembler.py    # сборка Markdown
├── cache.py        # кеш страниц
├── config.py       # настройки (pydantic-settings)
├── lmstudio.py     # клиент LM Studio
├── pdf_processor.py# рендеринг + ресайз
├── pipeline.py     # оркестратор
├── prompts.py      # системные промпты
├── queue_manager.py# asyncio.Queue
└── vlm.py          # абстракция VLM + заглушка
```

## Лицензия

[MIT](LICENSE)

---

## Диагностика

**Ошибка `401 Unauthorized` при обращении к `/v1/models`** — сервер отклонил
API-ключ. Проверьте:

1. что в `FLASH_VLM_LMSTUDIO_API_KEY` лежит корректный ключ для вашего
   эндпоинта (для LM Studio на localhost подойдёт любое непустое значение,
   например `lm-studio`);
2. что `FLASH_VLM_LMSTUDIO_BASE_URL` заканчивается на `/v1`
   (например `https://llmtest.ru/v1`);
3. если ваш провайдер не отдаёт список моделей через `GET /v1/models` — задайте
   модель явно: `FLASH_VLM_MODEL=<id-модели>`.

Ключ всегда передаётся в заголовке `Authorization: Bearer ...` как для запросов
`/models`, так и для `chat/completions`.

**Проверить подключение:**

```bash
flash-vlm health     # покажет доступность и список моделей
flash-vlm models
```

