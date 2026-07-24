from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from services.dependencies import get_engine
from tests.conftest import TEST_ROOT


class FakeEngine:
    loaded = True
    device = "cpu"

    def correct_audio(self, audio_path, top_k=5, threshold=0.3, asr_text=None):
        text = asr_text or "我想配点太子参调理"
        return {
            "asr_text": text,
            "corrected_text": text.replace("太子参", "人参"),
            "candidates": [
                {
                    "rank": 1,
                    "surface_id": "herb_001_canonical",
                    "entity_id": "herb_001",
                    "surface_text": "人参",
                    "canonical_name": "人参",
                    "entity_type": "herb",
                    "risk_level": "high",
                    "score": 0.98,
                    "gl_prediction": "太子参",
                    "action": "replace",
                    "applied": True,
                }
            ],
            "timings": {"encode_ms": 1.0, "search_ms": 2.0, "label_ms": 3.0},
            "duration_seconds": 1.5,
        }


@pytest.fixture()
def client() -> TestClient:
    from app import create_app
    from extensions import ext_database

    SQLModel.metadata.create_all(ext_database.engine)
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: FakeEngine()
    return TestClient(app)


def test_full_correction_flow(client: TestClient) -> None:
    response = client.post(
        "/api/corrections",
        files={"file": ("sample.wav", b"fake-wav-bytes", "audio/wav")},
        data={"top_k": "5", "threshold": "0.3"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["asr_text"] == "我想配点太子参调理"
    assert payload["corrected_text"] == "我想配点人参调理"
    assert payload["candidates"][0]["action"] == "replace"
    correction_id = payload["id"]

    rerun = client.post(
        f"/api/corrections/{correction_id}/rerun",
        json={"asr_text": "改过的文本里有太子参"},
    )
    assert rerun.status_code == 200
    assert rerun.json()["asr_text"] == "改过的文本里有太子参"

    listing = client.get("/api/corrections")
    assert listing.status_code == 200
    assert any(row["id"] == correction_id for row in listing.json()["corrections"])

    detail = client.get(f"/api/corrections/{correction_id}")
    assert detail.status_code == 200
    assert detail.json()["audio_url"].endswith("/audio")

    audio = client.get(f"/api/corrections/{correction_id}/audio")
    assert audio.status_code == 200
    assert audio.content == b"fake-wav-bytes"


def test_example_correction_flow(client: TestClient) -> None:
    audio_dir = TEST_ROOT / "examples-audio" / "utterances" / "test"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "ss_00001.wav").write_bytes(b"example-wav")
    (TEST_ROOT / "examples.json").write_text(
        json.dumps(
            [
                {
                    "id": "example_01",
                    "title": "demo",
                    "utterance_id": "ss_00001",
                    "audio_path": "utterances/test/ss_00001.wav",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    examples = client.get("/api/examples")
    assert examples.status_code == 200
    assert examples.json()["examples"][0]["id"] == "example_01"

    audio = client.get("/api/examples/example_01/audio")
    assert audio.status_code == 200
    assert audio.content == b"example-wav"

    response = client.post("/api/corrections", data={"example_id": "example_01"})
    assert response.status_code == 200
    assert response.json()["source"] == "example"


def test_missing_audio_is_rejected(client: TestClient) -> None:
    response = client.post("/api/corrections", data={})
    assert response.status_code == 400
