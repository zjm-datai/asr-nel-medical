from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.base import Settings, get_settings
from controllers import corrections, examples, health, metrics
from extensions import (
    ext_database,
    ext_docs,
    ext_frontend,
    ext_logging,
    ext_model,
)

EXTENSIONS = (
    ext_logging,
    ext_database,
    ext_model,
    ext_docs,
    ext_frontend,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=settings.app_name,
        root_path=settings.root_path,
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(corrections.router)
    app.include_router(metrics.router)
    app.include_router(examples.router)

    started = time.perf_counter()
    for extension in EXTENSIONS:
        extension_started = time.perf_counter()
        extension.init_app(app, settings)
        elapsed_ms = (time.perf_counter() - extension_started) * 1000
        logging.getLogger("app.extensions").info(
            "Initialized %s in %.2fms",
            extension.__name__,
            elapsed_ms,
        )
    total_ms = (time.perf_counter() - started) * 1000
    logging.getLogger("app.extensions").info(
        "Initialized %d extensions in %.2fms", len(EXTENSIONS), total_ms
    )
    return app


app = create_app()
