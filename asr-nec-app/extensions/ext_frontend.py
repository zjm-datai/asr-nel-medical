from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from configs.base import Settings


def init_app(app: FastAPI, settings: Settings) -> None:
    if not (settings.frontend_dist_dir / "index.html").is_file():
        return

    app.mount(
        "/",
        StaticFiles(directory=settings.frontend_dist_dir, html=True),
        name="frontend",
    )
