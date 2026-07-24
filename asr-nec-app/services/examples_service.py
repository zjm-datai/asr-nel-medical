from __future__ import annotations

import json
from typing import Any

from configs.base import Settings


def list_examples(settings: Settings) -> list[dict[str, Any]]:
    path = settings.examples_file
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    examples = []
    for row in rows:
        examples.append(
            {
                "id": row["id"],
                "title": row.get("title", row["id"]),
                "utterance_id": row.get("utterance_id", ""),
                "domain": row.get("domain", ""),
                "audio_path": row["audio_path"],
                "audio_url": f"/api/examples/{row['id']}/audio",
                "expected_asr_text": row.get("expected_asr_text", ""),
                "expected_corrected_text": row.get("expected_corrected_text", ""),
                "note": row.get("note", ""),
            }
        )
    return examples


def get_example(settings: Settings, example_id: str) -> dict[str, Any] | None:
    for example in list_examples(settings):
        if example["id"] == example_id:
            return example
    return None
