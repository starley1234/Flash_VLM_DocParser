from __future__ import annotations

import asyncio

import httpx
import pytest

from flash_vlm.lmstudio import LmStudioClient, LmStudioError


def _client_with_transport(handler, api_key: str = "secret-123") -> LmStudioClient:
    client = LmStudioClient(base_url="https://example.com/v1", api_key=api_key)
    client._http = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )
    return client


def test_list_models_sends_authorization_header():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    client = _client_with_transport(handler)
    models = asyncio.run(client.list_models())
    assert models == ["model-a"]
    assert captured["authorization"] == "Bearer secret-123"


def test_list_models_401_raises_helpful_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = _client_with_transport(handler)
    with pytest.raises(LmStudioError, match="FLASH_VLM_LMSTUDIO_API_KEY"):
        asyncio.run(client.list_models())


def test_list_models_parses_items():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}, {"id": ""}]})

    client = _client_with_transport(handler)
    assert asyncio.run(client.list_models()) == ["a", "b"]
