from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session

from configs.base import Settings
from core.nec.engine import NecEngine
from models.entities import Correction
from services.audio_api_transcriber import AudioApiTranscriber
from services.examples_service import get_example

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".opus"}
logger = logging.getLogger(__name__)


def run_correction(
    session: Session,
    settings: Settings,
    engine: NecEngine,
    transcriber: AudioApiTranscriber,
    *,
    file_bytes: bytes | None,
    filename: str,
    example_id: str | None,
    source: str,
    requested_asr_provider: str,
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

    correction_id = uuid.uuid4().hex
    asr_text: str | None = None
    asr_provider = "local_whisper"
    external_asr_ms: float | None = None
    if requested_asr_provider == "audio_api":
        try:
            transcription = transcriber.transcribe(audio_path, correction_id)
            if transcription is not None:
                asr_text = transcription.text
                external_asr_ms = transcription.elapsed_ms
                asr_provider = "audio_api"
        except Exception as exc:
            logger.exception("audio_api transcription failed; falling back to Whisper")
            if not settings.audio_api_fallback_to_whisper:
                stored_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=503, detail=f"audio_api transcription failed: {exc}"
                ) from exc

    try:
        result = engine.correct_audio(
            audio_path,
            top_k=top_k,
            threshold=threshold,
            asr_text=asr_text,
        )
    except RuntimeError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if external_asr_ms is not None:
        result["timings"]["transcribe_ms"] = external_asr_ms
        result["timings"]["total_ms"] = round(
            result["timings"].get("total_ms", 0.0) + external_asr_ms, 1
        )

    correction = Correction(
        id=correction_id,
        source=source,
        asr_provider=asr_provider,
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
    correction.asr_provider = "manual"
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
