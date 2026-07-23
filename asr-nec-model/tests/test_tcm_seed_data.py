from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def read_jsonl(relative_path: str) -> list[dict]:
    path = WORKSPACE_ROOT / relative_path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_core_entity_lexicon_is_complete_and_consistent():
    entities = read_jsonl("data/entities/tcm_spleen_stomach_core.jsonl")
    assert len(entities) == 150
    assert Counter(item["entity_type"] for item in entities) == {
        "herb": 45,
        "formula": 30,
        "symptom": 35,
        "syndrome": 20,
        "tongue_pulse": 20,
    }

    ids = {item["entity_id"] for item in entities}
    names = {item["canonical_name"] for item in entities}
    assert len(ids) == len(entities)
    assert len(names) == len(entities)
    for item in entities:
        assert item["pinyin"]
        assert set(item["confusables"]) <= names
        assert item["review_status"] == "pending_expert_review"


def test_default_scenarios_cover_every_core_entity_with_valid_offsets():
    entities = read_jsonl("data/entities/tcm_spleen_stomach_core.jsonl")
    scenarios = read_jsonl("data/scenarios/tcm_spleen_stomach_seed.jsonl")
    assert len(scenarios) == 300

    covered = set()
    for scenario in scenarios:
        assert scenario["audio"] is None
        assert scenario["audio_source"] == "pending"
        for entity in scenario["entities"]:
            assert scenario["ref_text"][entity["start"] : entity["end"]] == entity["text"]
            covered.add(entity["entity_id"])

    assert covered == {item["entity_id"] for item in entities}
