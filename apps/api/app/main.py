from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.api.routes.ui import router as ui_router


app = FastAPI(title="Sales Analysis Agent API", version="0.1.0")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)
app.include_router(ui_router)
app.include_router(health_router)
app.include_router(analysis_router, prefix="/api/v1")
