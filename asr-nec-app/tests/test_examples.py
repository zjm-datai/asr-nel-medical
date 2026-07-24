from __future__ import annotations

import json

from configs.base import get_settings
from services.examples_service import get_example, list_examples
from tests.conftest import TEST_ROOT


def test_examples_roundtrip() -> None:
    settings = get_settings()
    (TEST_ROOT / "examples.json").write_text(
        json.dumps(
            [
                {
                    "id": "example_01",
                    "title": "脾虚湿蕴案例",
                    "utterance_id": "ss_00031",
                    "domain": "atopic_dermatitis",
                    "audio_path": "utterances/test/ss_00031.wav",
                    "expected_asr_text": "原文",
                    "expected_corrected_text": "纠错后",
                    "note": "疲犀失运 → 脾虚湿蕴",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    examples = list_examples(settings)
    assert len(examples) == 1
    assert examples[0]["audio_url"] == "/api/examples/example_01/audio"
    assert get_example(settings, "example_01")["title"] == "脾虚湿蕴案例"
    assert get_example(settings, "missing") is None


def test_examples_missing_file() -> None:
    settings = get_settings()
    settings.examples_file.unlink(missing_ok=True)
    assert list_examples(settings) == []
