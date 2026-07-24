from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    database: str
    model_loaded: bool
    device: str


class CandidateRead(BaseModel):
    rank: int
    surface_id: str
    entity_id: str
    surface_text: str
    canonical_name: str = ""
    entity_type: str = ""
    risk_level: str = ""
    score: float
    gl_prediction: str = ""
    action: str = "reject"
    applied: bool = False


class CorrectionRead(BaseModel):
    id: str
    created_at: datetime
    source: str
    audio_url: str
    duration_seconds: float
    asr_text: str
    corrected_text: str
    top_k: int
    threshold: float
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, Any] = Field(default_factory=dict)


class CorrectionListResponse(BaseModel):
    corrections: list[CorrectionRead]


class RerunRequest(BaseModel):
    asr_text: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SsMetricsSummary(BaseModel):
    run: str
    best_epoch: int | None = None
    test: dict[str, Any] = Field(default_factory=dict)


class GlRunSummary(BaseModel):
    run: str
    best_epoch: int | None = None
    token_test: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)


class MetricsSummaryResponse(BaseModel):
    ss: list[SsMetricsSummary] = Field(default_factory=list)
    gl: list[GlRunSummary] = Field(default_factory=list)


class ExampleRead(BaseModel):
    id: str
    title: str
    utterance_id: str
    domain: str = ""
    audio_url: str
    expected_asr_text: str = ""
    expected_corrected_text: str = ""
    note: str = ""


class ExampleListResponse(BaseModel):
    examples: list[ExampleRead]
