"""Streamlit-интерфейс Flash-VLM DocParser.

Запуск: ``flash-vlm ui`` или ``streamlit run src/flash_vlm/ui/streamlit_app.py``
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit не установлен. Установите: pip install 'flash-vlm-docparser[ui]'"
    ) from exc

from .. import prompts
from ..config import Settings, get_settings
from ..factory import build_pipeline
from ..schemas import ConversionResult

st.set_page_config(page_title="Flash-VLM DocParser", layout="wide")

_DEFAULT_SETTINGS = get_settings()


def _run_conversion(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    pages: str | None,
    prompt: str,
    progress_bar,
    text_placeholder,
) -> ConversionResult:
    pipeline = build_pipeline(settings)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    collected: list = []

    def on_progress(done: int, total: int, result) -> None:
        collected.append(result)
        progress_bar.progress(min(done / total, 1.0), text=f"Обработка страницы {result.page} из {total}...")
        streamed = "\n\n".join(
            f"<!-- Страница {p.page} -->\n{p.markdown}" if p.ok
            else f"<!-- Страница {p.page} -->\n> ⚠️ Не распознана: {p.error}"
            for p in collected
        )
        text_placeholder.markdown(streamed or "_Ожидание текста…_")

    async def run() -> ConversionResult:
        try:
            # Модель не передаём: она берётся из .env (FLASH_VLM_MODEL)
            # или определяется автоматически.
            return await pipeline.convert(
                tmp_path,
                pages=pages or None,
                prompt=prompt.strip() or None,
                page_markers=settings.page_markers,
                on_progress=on_progress,
            )
        finally:
            await pipeline.client.close()
            tmp_path.unlink(missing_ok=True)

    return asyncio.run(run())


def main() -> None:
    st.title("Flash-VLM DocParser")
    st.caption("PDF → Markdown через локальные VLM (LM Studio)")

    with st.sidebar:
        st.header("Настройки")
        base_url = st.text_input(
            "LM Studio API",
            value=_DEFAULT_SETTINGS.lmstudio_base_url,
        )
        if _DEFAULT_SETTINGS.model:
            st.caption(f"Модель (из FLASH_VLM_MODEL): **{_DEFAULT_SETTINGS.model}**")
        else:
            st.caption("Модель: автоопределение (FLASH_VLM_MODEL не задан)")
        width = st.slider(
            "Ширина изображения, px",
            256,
            1536,
            _DEFAULT_SETTINGS.image_width,
            step=64,
        )
        concurrency = st.slider("Параллельные запросы", 1, 4, _DEFAULT_SETTINGS.concurrency)
        page_markers = st.checkbox("Маркеры страниц (<!-- Page N -->)", value=False)
        use_cache = st.checkbox("Кешировать страницы", value=True)
        fake = st.checkbox("Демо-режим (без LM Studio)", value=_DEFAULT_SETTINGS.fake_vlm)

    st.subheader("Системный промпт")
    prompt = st.text_area(
        "Промпт для распознавания (VLM)",
        value=prompts.DEFAULT_SYSTEM_PROMPT,
        height=180,
        help="Правила, по которым VLM преобразует страницу в Markdown",
    )

    uploaded = st.file_uploader("Загрузите PDF-файл", type=["pdf"])
    pages = st.text_input("Страницы (необязательно), напр. 1-5,8", value="")

    if uploaded is not None and st.button("Конвертировать", type="primary"):
        settings = Settings(
            lmstudio_base_url=base_url,
            image_width=width,
            concurrency=concurrency,
            page_markers=page_markers,
            use_cache=use_cache,
            fake_vlm=fake,
        )

        progress_bar = st.progress(0.0, text="Подготовка…")
        text_placeholder = st.empty()

        with st.spinner("Конвертация…"):
            try:
                result = _run_conversion(
                    uploaded.getvalue(), uploaded.name, settings, pages, prompt,
                    progress_bar, text_placeholder,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка: {exc}")
                return

        progress_bar.progress(1.0, text="Готово")
        st.success(
            f"Готово: {result.output_path} "
            f"(модель {result.model}, страниц {result.pages_total}, "
            f"из кеша {result.cached_pages}, время {result.duration_ms / 1000:.1f} c)"
        )
        st.download_button(
            "Скачать Markdown",
            data=result.markdown,
            file_name=Path(uploaded.name).with_suffix(".md").name,
            mime="text/markdown",
        )

        # Галерея уменьшенных изображений страниц (контроль ресайза).
        with_images = [p for p in result.pages if p.image_path and Path(p.image_path).exists()]
        if with_images:
            st.subheader("Изображения страниц (после ресайза)")
            columns = st.columns(4)
            for index, page in enumerate(with_images):
                with columns[index % 4]:
                    st.image(
                        str(page.image_path),
                        caption=f"Стр. {page.page} — {page.image_width}×{page.image_height} px",
                        use_container_width=True,
                    )


if __name__ == "__main__":  # pragma: no cover
    main()
