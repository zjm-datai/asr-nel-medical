from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from configs.base import Settings


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    elapsed_ms: float


class AudioApiTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def transcribe(
        self, audio_path: Path, conversation_id: str
    ) -> TranscriptionResult | None:
        if not self.settings.audio_api_asr_enabled:
            return None

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="asr-nec-audio-api-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            self._normalize_to_wav(audio_path, wav_path)
            with wav_path.open("rb") as audio_file:
                response = httpx.post(
                    self.settings.audio_api_url,
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    data={
                        "organize_code": self.settings.audio_api_organize_code,
                        "conversation_id": conversation_id,
                        "recognition_mode": "text",
                    },
                    headers={"Accept": "application/json"},
                    timeout=self.settings.audio_api_timeout_seconds,
                )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("transcription", "")).strip()
        if not text:
            raise RuntimeError("audio_api returned an empty transcription")
        return TranscriptionResult(
            text=text,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    @staticmethod
    def _normalize_to_wav(source: Path, destination: Path) -> None:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            detail = getattr(exc, "stderr", b"")
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RuntimeError(f"failed to normalize audio for audio_api: {detail}") from exc
