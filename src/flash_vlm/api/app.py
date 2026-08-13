"""HTTP REST API Flash-VLM DocParser (FastAPI).

Эндпоинты:
- ``GET /health``                — состояние сервера и LM Studio;
- ``GET /models``                — список моделей LM Studio;
- ``POST /ocr``                  — OCR одного изображения (base64);
- ``POST /convert``              — загрузка PDF и запуск конвертации (возвращает job id);
- ``GET /jobs``                  — список задач;
- ``GET /jobs/{id}``             — статус/прогресс задачи;
- ``GET /jobs/{id}/result``      — итоговый Markdown;
- ``GET /jobs/{id}/download``    — скачать .md;
- ``GET /jobs/{id}/page/{n}``    — уменьшенное изображение страницы;
- ``DELETE /jobs/{id}``          — отменить/удалить задачу;
- ``/mcp``                       — MCP-сервер (streamable HTTP).

Запуск: ``flash-vlm serve`` или ``uvicorn flash_vlm.api.app:create_app --factory``.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .. import __version__, prompts
from ..config import Settings, get_settings
from ..factory import build_pipeline
from ..jobs import JobManager
from ..mcp.server import create_mcp_server
from ..schemas import Job, JobStatus
from ..utils import decode_base64_image, ensure_dir, pages_dir_for

__all__ = ["create_app", "main"]

INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Flash-VLM DocParser</title>
<style>
  :root { color-scheme: light; }
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto; padding: 24px;
         background: #ffffff; color: #1f2328; }
  h1 { margin-bottom: 4px; }
  .sub { color: #57606a; margin-top: 0; }
  .panel { border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; display: flex;
           flex-wrap: wrap; gap: 12px; align-items: flex-end; background: #f6f8fa; }
  .field { display: flex; flex-direction: column; gap: 4px; font-size: 14px; }
  input[type=text] { padding: 6px; border: 1px solid #d0d7de; border-radius: 4px;
                     background: #ffffff; color: #1f2328; }
  textarea { width: 100%; padding: 8px; border: 1px solid #d0d7de; border-radius: 6px;
             font-family: ui-monospace, monospace; font-size: 13px; background: #ffffff; color: #1f2328; }
  button { padding: 8px 16px; border: 0; border-radius: 6px; background: #2f81f7;
           color: white; cursor: pointer; font-size: 15px; }
  button:disabled { background: #555; cursor: wait; }
  #status { margin: 16px 0; font-size: 15px; min-height: 1.4em; }
  pre { background: #1f2328; color: #f0f6fc; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; white-space: pre-wrap; overflow: auto; max-height: 60vh; font-size: 13px; }
  a.download { display: inline-block; margin-top: 12px; color: #0969da; }
  .error { color: #d1242f; }
  #gallery { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
  .page-fig { margin: 0; text-align: center; }
  .page-fig img { max-width: 150px; max-height: 200px; border: 1px solid #d0d7de;
                  border-radius: 4px; background: #fff; }
  .page-fig figcaption { font-size: 12px; color: #57606a; margin-top: 2px; }
</style>
</head>
<body>
  <h1>Flash-VLM DocParser</h1>
  <p class="sub">PDF → Markdown через локальные VLM (LM Studio)</p>

  <div class="panel">
    <div class="field">
      <label for="file">PDF-файл</label>
      <input type="file" id="file" accept=".pdf,application/pdf"/>
    </div>
    <div class="field">
      <label for="pages">Страницы (напр. 1-5,8)</label>
      <input type="text" id="pages" placeholder="все"/>
    </div>
    <label><input type="checkbox" id="markers"/> маркеры страниц</label>
    <button id="go" onclick="convert()">Конвертировать</button>
  </div>

  <div class="field" style="margin-top:12px;">
    <label for="prompt">Системный промпт (необязательно)</label>
    <textarea id="prompt" rows="5" placeholder="Оставьте пустым для промпта по умолчанию"></textarea>
  </div>

  <div id="status"></div>
  <div id="gallery"></div>
  <div id="download"></div>
  <pre id="result" hidden></pre>

<script>
async function convert() {
  const file = document.getElementById('file').files[0];
  if (!file) { alert('Выберите PDF-файл'); return; }
  const status = document.getElementById('status');
  const button = document.getElementById('go');
  const result = document.getElementById('result');
  const gallery = document.getElementById('gallery');
  result.hidden = true;
  result.textContent = '';
  gallery.innerHTML = '';
  document.getElementById('download').innerHTML = '';
  button.disabled = true;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('pages', document.getElementById('pages').value);
  fd.append('page_markers', document.getElementById('markers').checked);
  fd.append('prompt', document.getElementById('prompt').value);

  try {
    const res = await fetch('/convert', { method: 'POST', body: fd });
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || 'Ошибка загрузки');
    await poll(job.job_id, status);
  } catch (e) {
    status.innerHTML = '<span class="error">Ошибка: ' + e.message + '</span>';
  } finally {
    button.disabled = false;
  }
}

async function poll(id, status) {
  const pre = document.getElementById('result');
  const gallery = document.getElementById('gallery');
  const rendered = new Set();
  while (true) {
    const r = await fetch('/jobs/' + id);
    const j = await r.json();
    status.textContent = 'Статус: ' + j.status + ' — страница ' + j.current_page +
      ' из ' + j.total + ' (' + j.done + '/' + j.total + ')';

    // Потоково добавляем готовые страницы (текст + уменьшенное изображение).
    for (const p of (j.completed_pages || [])) {
      if (rendered.has(p.page)) continue;
      rendered.add(p.page);

      let block = '\n\n--- Страница ' + p.page + ' ---\n\n';
      if (p.ok) block += p.markdown || '';
      else block += '⚠️ Страница ' + p.page + ' не распознана: ' + (p.error || 'неизвестная ошибка');
      pre.textContent += block;
      pre.hidden = false;
      pre.scrollTop = pre.scrollHeight;

      const fig = document.createElement('figure');
      fig.className = 'page-fig';
      const a = document.createElement('a');
      a.href = '/jobs/' + id + '/page/' + p.page;
      a.target = '_blank';
      const img = document.createElement('img');
      img.src = '/jobs/' + id + '/page/' + p.page;
      img.alt = 'Страница ' + p.page;
      a.appendChild(img);
      const cap = document.createElement('figcaption');
      cap.textContent = 'Стр. ' + p.page + ' — ' + (p.image_width || '?') + '×' + (p.image_height || '?') + ' px';
      fig.appendChild(a);
      fig.appendChild(cap);
      gallery.appendChild(fig);
    }

    if (j.status === 'done') {
      document.getElementById('download').innerHTML =
        '<a class="download" href="/jobs/' + id + '/download" download>Скачать .md</a>';
      status.textContent = 'Готово: ' + j.done + ' из ' + j.total + ' страниц распознано.';
      break;
    } else if (j.status === 'error') {
      status.innerHTML = '<span class="error">Ошибка: ' + (j.error || 'неизвестная') + '</span>';
      break;
    }
    await new Promise(r => setTimeout(r, 800));
  }
}
</script>
</body>
</html>
"""


class OcrRequest(BaseModel):
    image_base64: str
    prompt: str | None = None
    model: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Создаёт и настраивает FastAPI-приложение."""
    settings = settings or get_settings()
    pipeline = build_pipeline(settings)
    jobs = JobManager()

    # MCP-сервер (streamable HTTP). Монтируем как ASGI-приложение на /mcp.
    mcp_server = create_mcp_server(settings, pipeline)
    mcp_http_app = mcp_server.streamable_http_app(streamable_http_path="/", host=settings.api_host)
    mcp_session_manager = mcp_server.session_manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Starlette не запускает lifespan вложенных приложений, поэтому входим
        # в session manager MCP явно (инициализирует внутреннюю task group).
        async with mcp_session_manager.run():
            yield
        await pipeline.client.close()

    app = FastAPI(
        title="Flash-VLM DocParser",
        description="Конвертация PDF в Markdown через локальные VLM (LM Studio).",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.jobs = jobs

    # ------------------------------------------------------------------
    async def run_job(
        job_id: str,
        pdf_path: Path,
        model: str | None,
        pages: str | None,
        page_markers: bool,
        use_cache: bool | None,
        prompt: str | None,
    ) -> None:
        job: Job | None = jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        try:
            stem = Path(job.filename).stem or "document"
            output_path = settings.output_dir / f"{job.id}_{stem}.md"
            if settings.save_page_images:
                # Каталог с изображениями страниц создаём заранее, чтобы
                # эндпоинт /page/{n} мог отдавать их по мере готовности.
                job.pages_dir = str(pages_dir_for(output_path))
                ensure_dir(Path(job.pages_dir))

            def on_progress(done: int, total: int, result) -> None:
                job.done = done
                job.total = total
                job.current_page = result.page
                job.completed_pages.append(result)

            result = await pipeline.convert(
                pdf_path,
                pages=pages,
                model=model,
                prompt=prompt or None,
                page_markers=page_markers,
                use_cache=use_cache,
                output_path=output_path,
                source_name=job.filename,
                on_progress=on_progress,
            )
            job.result = result
            job.output_path = result.output_path
            job.pages_dir = result.pages_dir or job.pages_dir
            job.status = JobStatus.DONE
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.ERROR
            job.error = str(exc)
        finally:
            job.finished_at = datetime.now(timezone.utc)
            pdf_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/health")
    async def health() -> dict:
        lm = await pipeline.client.check_health()
        lm["base_url"] = settings.lmstudio_base_url
        return {
            "status": "ok",
            "version": __version__,
            "fake_vlm": settings.fake_vlm,
            "lmstudio": lm,
        }

    @app.get("/models")
    async def models() -> dict:
        try:
            return {"models": await pipeline.client.list_models()}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"LM Studio недоступен: {exc}") from exc

    @app.post("/ocr")
    async def ocr(request: OcrRequest) -> dict:
        try:
            image = decode_base64_image(request.image_base64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Некорректное изображение: {exc}") from exc
        try:
            model = await pipeline.client.resolve_model(request.model)
            text = await pipeline.client.ocr_image(
                image,
                request.prompt or prompts.DEFAULT_SYSTEM_PROMPT,
                model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Ошибка VLM: {exc}") from exc
        return {"markdown": text, "model": model}

    @app.post("/convert", status_code=202)
    async def convert(
        file: UploadFile = File(...),
        model: str = Form(""),
        pages: str | None = Form(None),
        page_markers: bool = Form(False),
        use_cache: bool | None = Form(None),
        prompt: str | None = Form(None),
    ) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Ожидается файл с расширением .pdf")

        content = await file.read()
        limit = settings.max_upload_mb * 1024 * 1024
        if len(content) > limit:
            raise HTTPException(status_code=413, detail=f"Файл больше {settings.max_upload_mb} МБ")

        uploads = ensure_dir(settings.cache_dir / "uploads")
        pdf_path = uploads / f"{uuid.uuid4().hex}.pdf"
        pdf_path.write_bytes(content)

        job = jobs.create(file.filename or "document.pdf")
        task = asyncio.create_task(
            run_job(job.id, pdf_path, model or None, pages, page_markers, use_cache, prompt or None)
        )
        jobs.bind_task(job.id, task)

        return {
            "job_id": job.id,
            "filename": job.filename,
            "status": job.status.value,
            "status_url": f"/jobs/{job.id}",
            "result_url": f"/jobs/{job.id}/result",
        }

    @app.get("/jobs")
    async def list_jobs() -> dict:
        return {"jobs": [j.model_dump(exclude={"result", "completed_pages"}) for j in jobs.list()]}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return job.model_dump(exclude={"result"})

    @app.get("/jobs/{job_id}/result")
    async def get_job_result(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        if job.status == JobStatus.ERROR:
            raise HTTPException(status_code=422, detail=f"Задача завершилась с ошибкой: {job.error}")
        if job.status != JobStatus.DONE or job.result is None:
            raise HTTPException(status_code=409, detail="Задача ещё выполняется")
        return job.result.model_dump()

    @app.get("/jobs/{job_id}/download")
    async def download_job(job_id: str) -> FileResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        if job.status != JobStatus.DONE or not job.output_path or not Path(job.output_path).exists():
            raise HTTPException(status_code=409, detail="Результат ещё не готов")
        filename = Path(job.filename).with_suffix(".md").name
        return FileResponse(job.output_path, media_type="text/markdown", filename=filename)

    @app.get("/jobs/{job_id}/page/{page_number}")
    async def get_job_page_image(job_id: str, page_number: int) -> FileResponse:
        """Возвращает уменьшенное изображение страницы (контроль ресайза)."""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        if not job.pages_dir:
            raise HTTPException(status_code=404, detail="Изображения страниц недоступны (save_page_images=false)")
        path = Path(job.pages_dir) / f"page_{page_number:04d}.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Изображение страницы не найдено")
        return FileResponse(path, media_type="image/png")

    @app.delete("/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict:
        if not jobs.remove(job_id):
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return {"deleted": job_id}

    # ------------------------------------------------------------------
    # MCP-сервер поверх HTTP (streamable HTTP transport) на /mcp
    app.mount("/mcp", mcp_http_app)

    return app


def main() -> int:
    """Точка входа ``flash-vlm-api``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("flash_vlm.api.app:create_app", factory=True, host=settings.api_host, port=settings.api_port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
