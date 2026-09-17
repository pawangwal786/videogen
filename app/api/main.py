"""Canonical FastAPI application entrypoint for the VideoGen control plane."""

from app.api.app import create_app

# The default application instance for ASGI servers (Uvicorn, Gunicorn, Hypercorn).
# Run with: uvicorn app.api.main:app --host 0.0.0.0 --port 8000
app = create_app()
