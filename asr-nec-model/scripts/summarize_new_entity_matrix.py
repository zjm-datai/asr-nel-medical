from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired A/B/C new-entity evaluation metrics.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing A/B/C seed subdirectories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict = {"conditions": {}, "paired_differences": {}}
    for condition in ("A", "B", "C"):
        metrics = []
        for path in sorted((args.root / condition).glob("seed_*/metrics.json")):
            item = load(path)
            metrics.append({"seed": path.parent.name, **item})
        if not metrics:
            raise ValueError(f"no metrics found for condition {condition}")
        result["conditions"][condition] = {"runs": metrics, "mean": {}, "std": {}}
        keys = ("all", "new", "old")
        for group in keys:
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10"):
                values = [run.get("groups", {}).get(group, {}).get(metric, 0.0) for run in metrics]
                result["conditions"][condition]["mean"][f"{group}.{metric}"] = round(statistics.mean(values), 6)
                result["conditions"][condition]["std"][f"{group}.{metric}"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
    for left, right, name in (("A", "B", "bank_addition"), ("B", "C", "training_gain")):
        result["paired_differences"][name] = {}
        for group in ("new", "old", "all"):
            metric = "recall_at_5"
            result["paired_differences"][name][group] = round(result["conditions"][right]["mean"][f"{group}.{metric}"] - result["conditions"][left]["mean"][f"{group}.{metric}"], 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
