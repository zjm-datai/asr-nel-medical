from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session

from configs.base import Settings, get_settings
from core.nec.engine import NecEngine
from extensions.ext_database import get_session
from models.entities import Correction
from models.schemas import (
    CorrectionListResponse,
    CorrectionRead,
    RerunRequest,
)
from services import correction_service, history_service
from services.dependencies import get_engine

router = APIRouter(prefix="/api/corrections", tags=["corrections"])


def to_read(correction: Correction) -> CorrectionRead:
    return CorrectionRead(
        id=correction.id,
        created_at=correction.created_at,
        source=correction.source,
        audio_url=correction.audio_url,
        duration_seconds=correction.duration_seconds,
        asr_text=correction.asr_text,
        corrected_text=correction.corrected_text,
        top_k=correction.top_k,
        threshold=correction.threshold,
        candidates=correction.candidates or [],
        timings=correction.timings or {},
    )


@router.post("", response_model=CorrectionRead)
def create_correction(
    file: UploadFile | None = File(None),
    example_id: str | None = Form(None),
    source: str | None = Form(None),
    top_k: int = Form(5, ge=1, le=10),
    threshold: float = Form(0.3, ge=0.0, le=1.0),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: NecEngine = Depends(get_engine),
) -> CorrectionRead:
    file_bytes: bytes | None = None
    filename = "audio.wav"
    if file is not None:
        file_bytes = file.file.read()
        filename = file.filename or filename
    correction = correction_service.run_correction(
        session,
        settings,
        engine,
        file_bytes=file_bytes,
        filename=filename,
        example_id=example_id,
        source="mic" if source == "mic" else "upload",
        top_k=top_k,
        threshold=threshold,
    )
    return to_read(correction)


@router.post("/{correction_id}/rerun", response_model=CorrectionRead)
def rerun_correction(
    correction_id: str,
    payload: RerunRequest,
    session: Session = Depends(get_session),
    engine: NecEngine = Depends(get_engine),
) -> CorrectionRead:
    correction = history_service.get_correction(session, correction_id)
    if correction is None:
        raise HTTPException(status_code=404, detail="correction not found")
    updated = correction_service.rerun_correction(
        session,
        engine,
        correction,
        asr_text=payload.asr_text,
        top_k=payload.top_k,
        threshold=payload.threshold,
    )
    return to_read(updated)


@router.get("", response_model=CorrectionListResponse)
def list_corrections(
    session: Session = Depends(get_session),
) -> CorrectionListResponse:
    rows = history_service.list_corrections(session)
    return CorrectionListResponse(corrections=[to_read(row) for row in rows])


@router.get("/{correction_id}", response_model=CorrectionRead)
def get_correction(
    correction_id: str,
    session: Session = Depends(get_session),
) -> CorrectionRead:
    correction = history_service.get_correction(session, correction_id)
    if correction is None:
        raise HTTPException(status_code=404, detail="correction not found")
    return to_read(correction)


@router.get("/{correction_id}/audio")
def get_correction_audio(
    correction_id: str,
    session: Session = Depends(get_session),
) -> FileResponse:
    correction = history_service.get_correction(session, correction_id)
    if correction is None:
        raise HTTPException(status_code=404, detail="correction not found")
    audio_path = Path(correction.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="audio file not found")
    return FileResponse(audio_path)
