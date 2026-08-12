"""HTTP REST API Flash-VLM DocParser (FastAPI)."""

from __future__ import annotations

from .app import create_app, main

__all__ = ["create_app", "main"]
