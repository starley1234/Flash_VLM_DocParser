from __future__ import annotations

import base64
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from flash_vlm.api.app import create_app
from flash_vlm.config import Settings


def make_client(tmp_path) -> TestClient:
    settings = Settings(
        fake_vlm=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )
    return TestClient(create_app(settings))


def test_health(tmp_path):
    client = make_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["fake_vlm"] is True
    assert body["lmstudio"]["ok"] is True


def test_index_page(tmp_path):
    client = make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Flash-VLM DocParser" in response.text


def test_ocr_endpoint(tmp_path):
    client = make_client(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.post("/ocr", json={"image_base64": b64})
    assert response.status_code == 200
    assert "markdown" in response.json()


def test_convert_job_lifecycle(tmp_path, sample_pdf):
    client = make_client(tmp_path)

    with sample_pdf.open("rb") as handle:
        response = client.post(
            "/convert",
            files={"file": ("sample.pdf", handle, "application/pdf")},
            data={"model": "", "pages": "", "page_markers": "true"},
        )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status = {}
    for _ in range(200):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in {"done", "error"}:
            break
        time.sleep(0.02)

    assert status["status"] == "done", status
    assert status["total"] == 3
    assert status["done"] == 3

    result = client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["pages_total"] == 3
    assert result.json()["markdown"].startswith("---\n")

    download = client.get(f"/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/markdown")
    assert download.content.startswith(b"---\n")

    # список задач
    listing = client.get("/jobs").json()
    assert any(j["id"] == job_id for j in listing["jobs"])

    deleted = client.delete(f"/jobs/{job_id}")
    assert deleted.status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_convert_rejects_non_pdf(tmp_path):
    client = make_client(tmp_path)
    response = client.post(
        "/convert",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
