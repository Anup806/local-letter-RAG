"""Compatibility ASGI entrypoint for Uvicorn.

This module re-exports the FastAPI application defined in app.main so older
startup commands such as `uvicorn app.server:app` keep working.
"""

from app.main import app
