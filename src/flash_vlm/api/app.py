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
from ..utils import decode_base64_image, ensure_dir, pages_dir_for, parse_pages

__all__ = ["create_app", "main"]

INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Flash-VLM DocParser</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0 auto; max-width: 1200px; padding: 20px; background: #f6f8fa; color: #1f2328; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px;
           margin-bottom: 12px; flex-wrap: wrap; }
  .brand h1 { font-size: 20px; margin: 0; }
  .sub { color: #57606a; margin: 2px 0 0; font-size: 13px; }
  .btn { display: inline-block; padding: 8px 16px; border-radius: 6px; background: #2f81f7;
         color: #fff; cursor: pointer; font-size: 14px; border: 0; text-decoration: none; }
  .btn:hover { background: #1a68d8; }
  .btn.disabled { background: #9bb7e0; pointer-events: none; }

  details.settings { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; margin-bottom: 16px; }
  details.settings summary { padding: 10px 14px; cursor: pointer; font-weight: 600; font-size: 14px; user-select: none; }
  details.settings summary:hover { background: #f6f8fa; }
  .settings-grid { display: flex; flex-wrap: wrap; gap: 12px; padding: 4px 14px 14px; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 4px; font-size: 13px; }
  .field.grow { flex: 1 1 320px; }
  .field > span { color: #57606a; }
  .field input[type=text], .field input[type=number], .field textarea {
    padding: 7px 9px; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; color: #1f2328;
    font-family: inherit; font-size: 13px; }
  .field textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; resize: vertical; width: 100%; }
  .field input[type=number] { width: 110px; }
  .field.check { flex-direction: row; align-items: center; gap: 6px; padding-bottom: 7px; }

  .layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 16px; align-items: start; }
  @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }

  aside.docs { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; }
  aside.docs h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #57606a;
                  margin: 0; padding: 12px 14px 8px; }
  .doc-list { max-height: 72vh; overflow-y: auto; }
  .empty { color: #8b949e; font-size: 13px; padding: 12px 14px; margin: 0; }

  .doc { border-top: 1px solid #eaeef2; }
  .doc-head { display: flex; align-items: center; justify-content: space-between; gap: 8px;
              padding: 9px 12px; cursor: pointer; }
  .doc-head:hover { background: #f6f8fa; }
  .doc-head.selected { background: #eef4fe; }
  .doc-name { font-size: 14px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-meta { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .progress { font-size: 12px; color: #57606a; }

  .badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
  .badge.queued { background: #eaeef2; color: #57606a; }
  .badge.running { background: #fff3d6; color: #9a6700; }
  .badge.done { background: #dafbe1; color: #116329; }
  .badge.error { background: #ffebe9; color: #cf222e; }

  .pages { padding: 0 0 8px; }
  .pages-empty { font-size: 12px; color: #8b949e; padding: 4px 14px; }
  .page { display: flex; align-items: center; gap: 8px; padding: 4px 14px 4px 24px; cursor: pointer; font-size: 13px; }
  .page:hover { background: #f6f8fa; }
  .page.selected { background: #eef4fe; }
  .page-num { color: #1f2328; }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot.done { background: #2da44e; }
  .dot.error { background: #cf222e; }
  .dot.pending { background: #d0d7de; }
  .dot.processing { background: #d4a72c; animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .35; } }

  main.content { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; min-height: 60vh; }
  .empty-state { color: #8b949e; text-align: center; padding: 60px 20px; font-size: 14px; }
  .content-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .content-head h2 { font-size: 16px; margin: 0; word-break: break-word; }
  #download-link { font-size: 13px; color: #0969da; text-decoration: none; }
  #download-link:hover { text-decoration: underline; }

  pre { background: #1f2328; color: #f0f6fc; border-radius: 8px; padding: 14px; white-space: pre-wrap;
        word-break: break-word; overflow: auto; max-height: 65vh; font-size: 13px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0; }
  .page-preview { margin: 0 0 12px; }
  .page-preview img { max-width: 320px; width: 100%; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }
  .page-preview figcaption { font-size: 12px; color: #57606a; margin-top: 4px; }
  #upload-status { font-size: 13px; color: #57606a; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Flash-VLM DocParser</h1>
    <p class="sub">PDF → Markdown · локальные VLM</p>
  </div>
  <div class="header-actions">
    <span id="upload-status"></span>
    <label class="btn" id="add-btn" for="file-input">+ Добавить документ</label>
    <input type="file" id="file-input" accept=".pdf,application/pdf" multiple hidden/>
  </div>
</header>

<details class="settings" id="settings">
  <summary>Настройки</summary>
  <div class="settings-grid">
    <label class="field grow">
      <span>Системный промпт</span>
      <textarea id="prompt" rows="4"></textarea>
    </label>
    <label class="field">
      <span>Ширина, px</span>
      <input type="number" id="width" min="128" max="4096" step="64"/>
    </label>
    <label class="field">
      <span>Модель</span>
      <input type="text" id="model" placeholder="авто (из .env)"/>
    </label>
    <label class="field check">
      <input type="checkbox" id="markers"/>
      <span>маркеры страниц</span>
    </label>
  </div>
</details>

<div class="layout">
  <aside class="docs">
    <h2>Документы</h2>
    <div id="doc-list" class="doc-list"><p class="empty">Нет документов</p></div>
  </aside>
  <main class="content">
    <div id="content-empty" class="empty-state">Загрузите PDF, чтобы начать распознавание</div>
    <div id="content" hidden>
      <div class="content-head">
        <h2 id="content-title"></h2>
        <span id="content-status" class="badge queued"></span>
        <a id="download-link" hidden>Скачать .md</a>
      </div>
      <div id="content-body"></div>
    </div>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
const NL = '\n';
const LABELS = { queued: 'в очереди', running: 'обработка', done: 'готово', error: 'ошибка' };

const state = {
  jobs: new Map(),
  selectedDoc: null,
  selectedPage: null,
  expanded: new Set(),
  busy: false,
};

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

function badge(status) {
  const b = document.createElement('span');
  b.className = 'badge ' + status;
  b.textContent = LABELS[status] || status;
  return b;
}

async function init() {
  try {
    const s = await api('/settings');
    $('prompt').value = s.prompt || '';
    $('width').value = s.image_width || 512;
    $('model').value = s.model || '';
  } catch (e) { /* настройки не загрузились — оставляем поля пустыми */ }
  $('file-input').addEventListener('change', onFiles);
  tick();
}

async function onFiles(ev) {
  const files = Array.from(ev.target.files || []);
  ev.target.value = '';
  if (!files.length) return;
  const status = $('upload-status');
  const btn = $('add-btn');
  btn.classList.add('disabled');
  try {
    for (const f of files) {
      status.textContent = 'Загрузка: ' + f.name;
      await upload(f);
    }
  } catch (e) {
    alert('Ошибка загрузки: ' + e.message);
  } finally {
    status.textContent = '';
    btn.classList.remove('disabled');
  }
}

async function upload(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('prompt', $('prompt').value);
  fd.append('model', $('model').value);
  fd.append('image_width', $('width').value || '512');
  fd.append('page_markers', $('markers').checked);
  const job = await api('/convert', { method: 'POST', body: fd });
  state.selectedDoc = job.job_id;
  state.selectedPage = null;
  state.expanded.add(job.job_id);
  await refresh();
}

function selectDoc(id) {
  if (state.selectedDoc === id) {
    if (state.expanded.has(id)) state.expanded.delete(id);
    else state.expanded.add(id);
  } else {
    state.selectedDoc = id;
    state.expanded.add(id);
  }
  state.selectedPage = null;
  renderDocList();
  refreshContent();
}

function selectPage(id, n) {
  state.selectedDoc = id;
  state.selectedPage = n;
  state.expanded.add(id);
  renderDocList();
  refreshContent();
}

function docItem(j) {
  const wrap = document.createElement('div');
  wrap.className = 'doc';

  const head = document.createElement('div');
  head.className = 'doc-head' + (j.id === state.selectedDoc ? ' selected' : '');
  head.onclick = () => selectDoc(j.id);

  const name = document.createElement('div');
  name.className = 'doc-name';
  name.textContent = j.filename;
  name.title = j.filename;

  const meta = document.createElement('div');
  meta.className = 'doc-meta';
  meta.appendChild(badge(j.status));
  const prog = document.createElement('span');
  prog.className = 'progress';
  prog.textContent = j.total ? (j.done + '/' + j.total) : '…';
  meta.appendChild(prog);

  head.appendChild(name);
  head.appendChild(meta);
  wrap.appendChild(head);

  if (state.expanded.has(j.id)) wrap.appendChild(pagesList(j));
  return wrap;
}

function pagesList(j) {
  const wrap = document.createElement('div');
  wrap.className = 'pages';
  const nums = j.page_numbers || [];
  if (!nums.length) {
    const p = document.createElement('div');
    p.className = 'pages-empty';
    p.textContent = 'страницы появятся после запуска…';
    wrap.appendChild(p);
    return wrap;
  }
  const byPage = {};
  for (const p of (j.pages || [])) byPage[p.page] = p;
  for (const n of nums) {
    const row = document.createElement('div');
    row.className = 'page' + (j.id === state.selectedDoc && state.selectedPage === n ? ' selected' : '');
    row.onclick = () => selectPage(j.id, n);

    const info = byPage[n];
    let cls = 'dot pending';
    if (info && info.done) cls = info.ok ? 'dot done' : 'dot error';
    else if (j.status === 'running' && j.current_page === n) cls = 'dot processing';

    const dot = document.createElement('span');
    dot.className = cls;
    const label = document.createElement('span');
    label.className = 'page-num';
    label.textContent = 'Стр. ' + n;
    row.appendChild(dot);
    row.appendChild(label);
    wrap.appendChild(row);
  }
  return wrap;
}

function renderDocList() {
  const list = $('doc-list');
  list.replaceChildren();
  const jobs = Array.from(state.jobs.values()).sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  if (!jobs.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'Нет документов — добавьте PDF.';
    list.appendChild(p);
    return;
  }
  for (const j of jobs) list.appendChild(docItem(j));
}

function setStatus(status) {
  const el = $('content-status');
  el.className = 'badge ' + status;
  el.textContent = LABELS[status] || status;
}

function renderDoc(job) {
  $('content-title').textContent = job.filename;
  setStatus(job.status);
  const dl = $('download-link');
  dl.hidden = job.status !== 'done';
  if (job.status === 'done') dl.href = '/jobs/' + job.id + '/download';

  const pages = Array.from(job.completed_pages || []).sort((a, b) => a.page - b.page);
  const parts = pages.map(p => {
    const head = '--- Страница ' + p.page + ' ---';
    return p.ok ? (head + NL + (p.markdown || '')) : (head + NL + '⚠️ Не распознана: ' + (p.error || 'ошибка'));
  });
  let text = parts.join(NL + NL);
  if (!pages.length) text = (job.status === 'running' || job.status === 'queued') ? 'Подготовка…' : '(пусто)';
  else if (job.status === 'running' || job.status === 'queued') text += NL + NL + '… распознавание продолжается';

  const pre = document.createElement('pre');
  pre.textContent = text;
  $('content-body').replaceChildren(pre);
}

function renderPage(job, n) {
  const page = (job.completed_pages || []).find(p => p.page === n);
  $('content-title').textContent = job.filename + ' — стр. ' + n;
  $('download-link').hidden = true;
  const body = $('content-body');
  body.replaceChildren();

  if (!page) {
    setStatus(job.status);
    const pre = document.createElement('pre');
    pre.textContent = 'Страница ещё не обработана…';
    body.appendChild(pre);
    return;
  }

  setStatus(page.ok ? 'done' : 'error');

  const fig = document.createElement('figure');
  fig.className = 'page-preview';
  const a = document.createElement('a');
  a.href = '/jobs/' + job.id + '/page/' + n;
  a.target = '_blank';
  a.rel = 'noopener';
  const img = document.createElement('img');
  img.src = '/jobs/' + job.id + '/page/' + n;
  img.alt = 'Страница ' + n;
  a.appendChild(img);
  const cap = document.createElement('figcaption');
  cap.textContent = (page.image_width ? (page.image_width + '×' + page.image_height + ' px · ') : '') + 'открыть в новом окне';
  fig.appendChild(a);
  fig.appendChild(cap);
  body.appendChild(fig);

  const pre = document.createElement('pre');
  pre.textContent = page.ok ? (page.markdown || '(пустая страница)') : ('⚠️ Не распознана: ' + (page.error || 'ошибка'));
  body.appendChild(pre);
}

async function refreshContent() {
  const id = state.selectedDoc;
  const empty = $('content-empty');
  const content = $('content');
  if (!id) {
    content.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  content.hidden = false;

  let job;
  try {
    job = await api('/jobs/' + id);
  } catch (e) {
    const summary = state.jobs.get(id);
    $('content-title').textContent = summary ? summary.filename : id;
    setStatus('error');
    $('content-body').replaceChildren();
    return;
  }
  state.jobs.set(id, Object.assign({}, state.jobs.get(id) || {}, job));
  if (state.selectedPage != null) renderPage(job, state.selectedPage);
  else renderDoc(job);
}

async function refresh() {
  const data = await api('/jobs');
  const seen = new Set();
  for (const j of data.jobs) { state.jobs.set(j.id, j); seen.add(j.id); }
  for (const id of Array.from(state.jobs.keys())) if (!seen.has(id)) state.jobs.delete(id);
  renderDocList();
  await refreshContent();
}

async function tick() {
  if (state.busy) return;
  state.busy = true;
  try { await refresh(); } catch (e) { /* сетевая пауза — пропускаем */ }
  finally { state.busy = false; }
  setTimeout(tick, 700);
}

init();
</script>
</body>
</html>
"""


class OcrRequest(BaseModel):
    image_base64: str
    prompt: str | None = None
    model: str | None = None


def _job_summary(job: Job) -> dict:
    """Краткое представление задачи для списка документов.

    ``completed_pages`` с полным текстом не включается (тяжёлый); вместо него —
    компактный список статусов страниц ``pages``.
    """
    data = job.model_dump(exclude={"result", "completed_pages"})
    by_page = {p.page: p for p in job.completed_pages}
    data["pages"] = [
        {"page": n, "done": n in by_page, "ok": by_page[n].ok if n in by_page else None}
        for n in job.page_numbers
    ]
    return data


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
        image_width: int | None,
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

            # Список страниц известен до запуска — для иерархии «документ → страницы».
            total_pages = pipeline.pdf.page_count(pdf_path)
            job.page_numbers = parse_pages(pages, total_pages)
            job.total = len(job.page_numbers)

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
                image_width=image_width,
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
        image_width: int | None = Form(None),
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
            run_job(job.id, pdf_path, model or None, pages, page_markers, use_cache, prompt or None, image_width)
        )
        jobs.bind_task(job.id, task)

        return {
            "job_id": job.id,
            "filename": job.filename,
            "status": job.status.value,
            "status_url": f"/jobs/{job.id}",
            "result_url": f"/jobs/{job.id}/result",
        }

    @app.get("/settings")
    async def get_ui_settings() -> dict:
        """Настройки по умолчанию для предзаполнения UI (промпт, ширина, модель)."""
        return {
            "prompt": prompts.DEFAULT_SYSTEM_PROMPT,
            "image_width": settings.image_width,
            "model": settings.model,
            "lmstudio_base_url": settings.lmstudio_base_url,
            "concurrency": settings.concurrency,
            "fake_vlm": settings.fake_vlm,
        }

    @app.get("/jobs")
    async def list_jobs() -> dict:
        return {"jobs": [_job_summary(j) for j in jobs.list()]}

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
