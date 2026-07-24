from __future__ import annotations

import logging

from fastapi import FastAPI

from configs.base import Settings
from services.dependencies import get_engine

logger = logging.getLogger("app.model")


def init_app(app: FastAPI, settings: Settings) -> None:
    engine = get_engine()
    app.state.nec_engine = engine
    if settings.nec_skip_model_load:
        logger.info("NEC model load skipped (NEC_SKIP_MODEL_LOAD=true)")
        return
    engine.load()
    logger.info("NEC engine loaded on device=%s", engine.device)
