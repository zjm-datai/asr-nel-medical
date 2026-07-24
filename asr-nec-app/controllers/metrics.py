from __future__ import annotations

from fastapi import APIRouter, Depends

from configs.base import Settings, get_settings
from models.schemas import MetricsSummaryResponse
from services.metrics_service import load_metrics_summary

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    settings: Settings = Depends(get_settings),
) -> MetricsSummaryResponse:
    summary = load_metrics_summary(str(settings.nec_runs_dir))
    return MetricsSummaryResponse(ss=summary["ss"], gl=summary["gl"])
