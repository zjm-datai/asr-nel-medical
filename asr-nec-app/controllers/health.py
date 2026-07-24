from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session

from configs.base import Settings, get_settings
from extensions.ext_database import get_session
from models.schemas import HealthResponse
from services.dependencies import get_engine

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    session.exec(text("select 1"))
    engine = get_engine()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        database="ok",
        model_loaded=engine.loaded,
        device=engine.device,
    )
