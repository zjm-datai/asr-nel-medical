from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session

from configs.base import Settings
from core.nec.engine import NecEngine
from models.entities import Correction
from services.examples_service import get_example

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".opus"}


def run_correction(
    session: Session,
    settings: Settings,
    engine: NecEngine,
    *,
    file_bytes: bytes | None,
    filename: str,
    example_id: str | None,
    source: str,
    top_k: int,
    threshold: float,
) -> Correction:
    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"

    example_audio: Path | None = None
    if example_id:
        example = get_example(settings, example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="example not found")
        example_audio = _example_audio_path(settings, example)
        source = "example"

    stored_path = settings.upload_dir / f"tmp_{uuid.uuid4().hex}{suffix}"
    if example_audio is not None:
        audio_path = example_audio
    else:
        if not file_bytes:
            raise HTTPException(status_code=400, detail="audio file is required")
        stored_path.write_bytes(file_bytes)
        audio_path = stored_path

    try:
        result = engine.correct_audio(audio_path, top_k=top_k, threshold=threshold)
    except RuntimeError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    correction = Correction(
        source=source,
        asr_text=result["asr_text"],
        corrected_text=result["corrected_text"],
        candidates=result["candidates"],
        timings=result["timings"],
        duration_seconds=result["duration_seconds"],
        top_k=top_k,
        threshold=threshold,
    )
    if example_audio is not None:
        correction.audio_path = str(example_audio)
    else:
        final_path = settings.upload_dir / f"{correction.id}{suffix}"
        shutil.move(str(stored_path), str(final_path))
        correction.audio_path = str(final_path)
    correction.audio_url = f"/api/corrections/{correction.id}/audio"

    session.add(correction)
    session.commit()
    session.refresh(correction)
    return correction


def rerun_correction(
    session: Session,
    engine: NecEngine,
    correction: Correction,
    *,
    asr_text: str,
    top_k: int | None,
    threshold: float | None,
) -> Correction:
    audio_path = Path(correction.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=410, detail="audio file no longer available")
    try:
        result = engine.correct_audio(
            audio_path,
            top_k=top_k or correction.top_k,
            threshold=correction.threshold if threshold is None else threshold,
            asr_text=asr_text,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    correction.asr_text = result["asr_text"]
    correction.corrected_text = result["corrected_text"]
    correction.candidates = result["candidates"]
    correction.timings = result["timings"]
    session.add(correction)
    session.commit()
    session.refresh(correction)
    return correction


def _example_audio_path(settings: Settings, example: dict) -> Path:
    audio_path = (settings.examples_audio_dir / example["audio_path"]).resolve()
    root = settings.examples_audio_dir.resolve()
    if root not in audio_path.parents or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="example audio not found")
    return audio_path
