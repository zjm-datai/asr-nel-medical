import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_tts_full import build_jobs  # noqa: E402


def test_full_tts_plan_matches_corpus_and_is_split_safe():
    jobs = build_jobs(WORKSPACE_ROOT / "data" / "speech_searcher")
    assert len(jobs) == 4290
    assert len({row["source_id"] for row in jobs}) == 4290

    kind_counts = Counter(row["kind"] for row in jobs)
    assert kind_counts == {"utterance": 3600, "entity": 690}
    split_counts = Counter(row["split"] for row in jobs if row["kind"] == "utterance")
    assert split_counts == {"train": 2520, "dev": 360, "test": 720}

    split_voices = {
        split: {row["voice"] for row in jobs if row["kind"] == "utterance" and row["split"] == split}
        for split in ("train", "dev", "test")
    }
    assert split_voices["train"] == {"my_zero_shot_spk", "yinyao", "yexuejie", "liqiong"}
    assert split_voices["dev"] == {"doctor_woman"}
    assert split_voices["test"] == {"haiwenyuan"}
    assert split_voices["train"].isdisjoint(split_voices["dev"] | split_voices["test"])
