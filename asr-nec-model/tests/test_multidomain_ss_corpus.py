from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = WORKSPACE_ROOT / "data" / "speech_searcher"


def read_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_corpus_counts_splits_and_coverage():
    entities = read_jsonl("entities.jsonl")
    surfaces = read_jsonl("entity_surfaces.jsonl")
    utterances = read_jsonl("utterances.jsonl")
    tts_requests = read_jsonl("entity_tts_manifest.jsonl")
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))

    assert len(entities) == summary["entity_count"] == 291
    assert Counter(item["correction_enabled"] for item in entities) == {True: 230, False: 61}
    assert len(surfaces) == summary["surface_count"] == 323
    assert sum(item["approved_for_audio"] for item in surfaces) == 230
    assert len(utterances) == summary["utterance_count"] == 3600
    assert len(tts_requests) == summary["entity_tts_request_count"] == 690
    assert Counter(item["split"] for item in utterances) == {
        "train": 2520,
        "dev": 360,
        "test": 720,
    }
    assert Counter(item["domain"] for item in utterances) == {
        "spleen_stomach": 1800,
        "atopic_dermatitis": 1800,
    }
    assert Counter(item["speaker_role"] for item in utterances) == {
        "doctor": 2068,
        "patient": 1532,
    }
    assert all(item["scene"] == "doctor_patient_turn" for item in utterances)

    focal_counts = Counter(
        (item["domain"], item["focal_entity_id"])
        for item in utterances
        if item["focal_entity_id"] is not None
    )
    core_focal_counts = Counter(
        (item["domain"], item["focal_entity_id"])
        for item in utterances
        if item["sample_type"] in {"correction_core", "link_only_core"}
    )
    entity_by_id = {item["entity_id"]: item for item in entities}
    for key, count in core_focal_counts.items():
        expected = 10 if entity_by_id[key[1]]["correction_enabled"] else 5
        assert count == expected
        assert focal_counts[key] >= expected


def test_annotations_pairs_and_tts_surfaces_are_consistent():
    entities = read_jsonl("entities.jsonl")
    surfaces = read_jsonl("entity_surfaces.jsonl")
    utterances = read_jsonl("utterances.jsonl")
    pairs = read_jsonl("ss_pairs.jsonl")
    tts_requests = read_jsonl("entity_tts_manifest.jsonl")
    entity_by_id = {item["entity_id"]: item for item in entities}
    surface_by_id = {item["surface_id"]: item for item in surfaces}
    utterance_by_id = {item["utterance_id"]: item for item in utterances}

    assert len(pairs) == 43430
    assert len({(item["utterance_id"], item["surface_id"]) for item in pairs}) == len(pairs)
    grouped_pairs = defaultdict(list)
    for pair in pairs:
        assert pair["entity_id"] in entity_by_id
        assert pair["surface_id"] in surface_by_id
        assert surface_by_id[pair["surface_id"]]["entity_id"] == pair["entity_id"]
        assert pair["split"] == utterance_by_id[pair["utterance_id"]]["split"]
        if pair["label"] == 1:
            assert entity_by_id[pair["entity_id"]]["correction_enabled"]
            assert surface_by_id[pair["surface_id"]]["approved_for_audio"]
        grouped_pairs[pair["utterance_id"]].append(pair)

    for utterance in utterances:
        correction_surfaces = {
            annotation["surface_id"]
            for annotation in utterance["entities"]
            if annotation["correction_enabled"]
        }
        for annotation in utterance["entities"]:
            assert utterance["ref_text"][annotation["start"] : annotation["end"]] == annotation["text"]
            assert surface_by_id[annotation["surface_id"]]["surface_text"] == annotation["text"]
        positive_surfaces = {
            pair["surface_id"]
            for pair in grouped_pairs[utterance["utterance_id"]]
            if pair["label"] == 1
        }
        assert positive_surfaces == correction_surfaces

    for request in tts_requests:
        surface = surface_by_id[request["surface_id"]]
        assert surface["approved_for_audio"]
        assert request["tts_text"] == surface["tts_text"]


def test_text_is_unique_and_dialogue_style():
    utterances = read_jsonl("utterances.jsonl")
    texts = [item["ref_text"] for item in utterances]
    assert len(set(texts)) == len(texts) == 3600
    assert all("处方栏记录" not in text and "请复核" not in text for text in texts)
    assert all("有没有季节变化" not in text and "有没有动物皮屑" not in text for text in texts)
    assert all("针对四弯风" not in text and "他克莫司软膏还是乌帕替尼" not in text for text in texts)

    texts_by_split = defaultdict(set)
    for utterance in utterances:
        texts_by_split[utterance["split"]].add(utterance["ref_text"])
    assert not (texts_by_split["train"] & texts_by_split["dev"])
    assert not (texts_by_split["train"] & texts_by_split["test"])
    assert not (texts_by_split["dev"] & texts_by_split["test"])


def test_concepts_correction_tier_and_tts_readings():
    entities = read_jsonl("entities.jsonl")
    utterances = read_jsonl("utterances.jsonl")
    by_name = {item["canonical_name"]: item for item in entities}

    assert "异位性皮炎" not in by_name
    assert "特应性湿疹" not in by_name
    assert set(by_name["特应性皮炎"]["aliases"]) >= {"异位性皮炎", "特应性湿疹", "异位性湿疹", "AD"}
    for name in {"嗳气", "嘈杂", "完谷不化", "矢气", "苔藓样变", "腘窝", "清热利湿"}:
        assert by_name[name]["correction_enabled"]

    acronym_utterances = [
        item
        for item in utterances
        if any(token in item["ref_text"] for token in ("IgE", "EASI", "SCORAD", "IGA"))
    ]
    assert acronym_utterances
    assert all(not any(token in item["audio_text"] for token in ("IgE", "EASI", "SCORAD", "IGA")) for item in acronym_utterances)
