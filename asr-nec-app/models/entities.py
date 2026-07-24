from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Correction(SQLModel, table=True):
    __tablename__ = "corrections"

    id: str = Field(default_factory=new_id, primary_key=True, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)
    source: str = Field(default="upload", index=True, max_length=32)
    audio_path: str = Field(default="", max_length=1000)
    audio_url: str = Field(default="", max_length=1000)
    duration_seconds: float = Field(default=0.0)
    asr_text: str = Field(default="")
    corrected_text: str = Field(default="")
    top_k: int = Field(default=5)
    threshold: float = Field(default=0.3)
    candidates: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    timings: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
