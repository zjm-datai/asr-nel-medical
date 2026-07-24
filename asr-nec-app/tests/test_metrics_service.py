from __future__ import annotations

import json

from services.metrics_service import load_metrics_summary
from tests.conftest import TEST_ROOT


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_metrics_summary(tmp_path) -> None:
    runs = tmp_path / "runs"
    _write_json(
        runs / "ss_full_seed_20260724" / "metrics.json",
        {"best_epoch": 12, "test_metrics": {"recall_at_5": 0.991}},
    )
    _write_json(
        runs / "gl_augmented_aligned_e5" / "metrics.json",
        {"best_epoch": 5, "test": {"loss": 0.022, "target_token_accuracy": 0.995}},
    )
    _write_json(
        runs / "gl_augmented_aligned_e5" / "generation_metrics.json",
        {"metrics": {"test": {"replace_f1": 0.931}}},
    )
    _write_json(
        runs / "ss_backup_20260724" / "metrics.json",
        {"best_epoch": 1, "test_metrics": {}},
    )
    (TEST_ROOT / "runs").mkdir(exist_ok=True)

    summary = load_metrics_summary(str(runs))
    assert [row["run"] for row in summary["ss"]] == ["ss_full_seed_20260724"]
    assert summary["ss"][0]["test"]["recall_at_5"] == 0.991
    assert [row["run"] for row in summary["gl"]] == ["gl_augmented_aligned_e5"]
    assert summary["gl"][0]["generation"]["test"]["replace_f1"] == 0.931
    assert all("backup" not in row["run"] for row in summary["ss"])
