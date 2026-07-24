from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from configs.base import Settings


def init_app(app: FastAPI, settings: Settings) -> None:
    configure_logging(settings)
    install_request_logging(app)


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def install_request_logging(app: FastAPI) -> None:
    logger = logging.getLogger("api.request")

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s %s %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
