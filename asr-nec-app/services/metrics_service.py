"""Aggregates training-run metrics from the mounted ``runs/`` directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_metrics_summary(runs_dir: str) -> dict[str, Any]:
    root = Path(runs_dir)
    ss_runs: list[dict[str, Any]] = []
    gl_runs: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or "backup" in child.name:
                continue
            metrics = _read_json(child / "metrics.json")
            if not metrics:
                continue
            if child.name.startswith("ss_"):
                ss_runs.append(
                    {
                        "run": child.name,
                        "best_epoch": metrics.get("best_epoch"),
                        "test": metrics.get("test_metrics", metrics.get("test", {})),
                    }
                )
            elif child.name.startswith("gl_"):
                generation = _read_json(child / "generation_metrics.json")
                gl_runs.append(
                    {
                        "run": child.name,
                        "best_epoch": metrics.get("best_epoch"),
                        "token_test": metrics.get("test", {}),
                        "generation": generation.get("metrics", {}),
                    }
                )
    return {"ss": ss_runs, "gl": gl_runs}
