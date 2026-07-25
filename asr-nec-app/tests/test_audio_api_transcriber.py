from __future__ import annotations

from pathlib import Path

import httpx

from configs.base import get_settings
from services.audio_api_transcriber import AudioApiTranscriber
from tests.conftest import TEST_ROOT


def test_disabled_transcriber_returns_none() -> None:
    transcriber = AudioApiTranscriber(get_settings())
    assert transcriber.transcribe(Path("unused.webm"), "conversation") is None


def test_transcriber_normalizes_and_posts_expected_form(monkeypatch) -> None:
    source = TEST_ROOT / "source.webm"
    source.write_bytes(b"source")
    settings = get_settings().model_copy(
        update={
            "audio_api_asr_enabled": True,
            "audio_api_url": "http://audio_server:8081/audio/transcription",
        }
    )
    normalized: list[tuple[Path, Path]] = []

    def fake_normalize(input_path: Path, output_path: Path) -> None:
        normalized.append((input_path, output_path))
        output_path.write_bytes(b"wav")

    def fake_post(url, *, files, data, headers, timeout):
        assert url == settings.audio_api_url
        assert files["file"][0] == "audio.wav"
        assert files["file"][2] == "audio/wav"
        assert data == {
            "organize_code": "asrnec",
            "conversation_id": "abc123",
            "recognition_mode": "text",
        }
        assert headers == {"Accept": "application/json"}
        assert timeout == 90.0
        return httpx.Response(
            200,
            json={"transcription": "外部转写", "file_id": "file-1"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        AudioApiTranscriber, "_normalize_to_wav", staticmethod(fake_normalize)
    )
    monkeypatch.setattr(httpx, "post", fake_post)

    result = AudioApiTranscriber(settings).transcribe(source, "abc123")

    assert result is not None
    assert result.text == "外部转写"
    assert result.elapsed_ms >= 0
    assert normalized[0][0] == source
