from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PILOT_DIR = WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot"
ASR_DIR = WORKSPACE_ROOT / "data" / "speech_searcher" / "asr_pilot"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_audio_pilot_manifest_is_complete_and_split_voices_are_disjoint():
    rows = read_jsonl(PILOT_DIR / "pilot_manifest.jsonl")
    assert Counter(row["kind"] for row in rows) == {"entity": 303, "utterance": 200}
    assert all(row["status"] == "generated" for row in rows)
    assert all((PILOT_DIR / row["pilot_audio_path"]).is_file() for row in rows)

    split_voices = {
        split: {row["voice"] for row in rows if row["kind"] == "utterance" and row["split"] == split}
        for split in ("train", "dev", "test")
    }
    assert split_voices["train"] == {"my_zero_shot_spk", "yinyao", "yexuejie", "liqiong"}
    assert split_voices["dev"] == {"doctor_woman"}
    assert split_voices["test"] == {"haiwenyuan"}
    assert not (split_voices["train"] & split_voices["dev"])
    assert not (split_voices["train"] & split_voices["test"])
    assert not (split_voices["dev"] & split_voices["test"])


def test_asr_hypotheses_and_oracle_gl_seed_are_explicitly_provenanced():
    hypotheses = read_jsonl(ASR_DIR / "asr_hypotheses.jsonl")
    gl_seed = read_jsonl(ASR_DIR / "gl_oracle_seed.jsonl")
    summary = json.loads((ASR_DIR / "summary.json").read_text(encoding="utf-8"))

    assert len(hypotheses) == len(gl_seed) == summary["utterance_count"] == 200
    assert all(row["asr_model"] == "openai:base" for row in hypotheses)
    assert any(row["asr_text"] != row["ref_text"] for row in hypotheses)
    assert all(row["candidate_source"] == "reference_annotation_oracle_not_ss_output" for row in gl_seed)
    assert all(row["training_status"] == "pipeline_debug_only_pending_ss_candidates" for row in gl_seed)
    assert 0 < summary["cer"] < 1
    assert summary["entity_error_count"] > 0


def test_speech_searcher_pilot_pairs_have_complete_evaluation_candidates():
    train_pairs = read_jsonl(PILOT_DIR / "ss_train_pairs.jsonl")
    eval_pairs = read_jsonl(PILOT_DIR / "ss_eval_pairs.jsonl")
    summary = json.loads((PILOT_DIR / "ss_pair_summary.json").read_text(encoding="utf-8"))

    assert len(train_pairs) == summary["train_pair_count"] == 3747
    assert len(eval_pairs) == summary["eval_pair_count"] == 6060
    assert sum(row["label"] for row in train_pairs) == 267
    assert sum(row["label"] for row in eval_pairs) == 47
    assert summary["candidate_surface_count"] == 101
    assert summary["entity_audio_views_per_surface"] == 3
    assert Counter(row["split"] for row in eval_pairs) == {"dev": 2020, "test": 4040}
    candidates_per_utterance = Counter(row["utterance_id"] for row in eval_pairs)
    assert set(candidates_per_utterance.values()) == {101}
