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

from ..config import Settings, get_settings
from ..factory import build_pipeline

st.set_page_config(page_title="Flash-VLM DocParser", layout="wide")

_DEFAULT_SETTINGS = get_settings()


def _run_conversion(pdf_bytes: bytes, filename: str, settings: Settings, pages: str | None) -> dict:
    pipeline = build_pipeline(settings)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    progress_bar = st.progress(0.0, text="Подготовка…")

    def on_progress(done: int, total: int, current_page: int) -> None:
        progress_bar.progress(min(done / total, 1.0), text=f"Обработка страницы {current_page} из {total}...")

    async def run() -> dict:
        try:
            result = await pipeline.convert(
                tmp_path,
                pages=pages or None,
                model=settings.model or None,
                page_markers=settings.page_markers,
                on_progress=on_progress,
            )
            return result.model_dump()
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
        model = st.text_input(
            "Модель (пусто = авто)",
            value=_DEFAULT_SETTINGS.model,
            help="По умолчанию берётся из FLASH_VLM_MODEL",
        )
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

    uploaded = st.file_uploader("Загрузите PDF-файл", type=["pdf"])
    pages = st.text_input("Страницы (необязательно), напр. 1-5,8", value="")

    if uploaded is not None and st.button("Конвертировать", type="primary"):
        settings = Settings(
            lmstudio_base_url=base_url,
            model=model,
            image_width=width,
            concurrency=concurrency,
            page_markers=page_markers,
            use_cache=use_cache,
            fake_vlm=fake,
        )
        with st.spinner("Конвертация…"):
            try:
                result = _run_conversion(uploaded.getvalue(), uploaded.name, settings, pages)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка: {exc}")
                return

        st.success(
            f"Готово: {result['output_path']} "
            f"(модель {result['model']}, страниц {result['pages_total']}, "
            f"из кеша {result['cached_pages']}, время {result['duration_ms'] / 1000:.1f} c)"
        )
        st.download_button(
            "Скачать Markdown",
            data=result["markdown"],
            file_name=Path(uploaded.name).with_suffix(".md").name,
            mime="text/markdown",
        )
        with st.expander("Предпросмотр", expanded=True):
            st.markdown(result["markdown"])


if __name__ == "__main__":  # pragma: no cover
    main()
