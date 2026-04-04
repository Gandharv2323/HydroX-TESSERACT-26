"""ASGI entrypoint for deployment.

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from main import app  # backward-compatible reuse of existing FastAPI app

__all__ = ["app"]
