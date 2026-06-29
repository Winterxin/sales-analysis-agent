from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse


router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)


@router.get("/app", include_in_schema=False, response_class=HTMLResponse)
def app_page() -> HTMLResponse:
    html_path = Path(__file__).resolve().parents[2] / "static" / "ui" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
