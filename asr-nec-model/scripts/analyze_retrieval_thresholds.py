from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan SS thresholds on frozen two-stage predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--thresholds",
        default="0.3,0.5,0.7,0.8,0.85,0.9,0.95,0.97,0.99",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    new_entity_ids = set(holdout["new_entity_ids"])
    utterances = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}
    predictions = read_jsonl(args.predictions)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    results = []
    for threshold in thresholds:
        totals = {"new": 0, "old": 0, "all": 0}
        hits = {"new": 0, "old": 0, "all": 0}
        no_entity = 0
        no_entity_fp = 0
        candidate_count = 0
        for prediction in predictions:
            selected = [
                row["surface_id"]
                for row in prediction["reranked"]
                if row["score"] >= threshold
            ][: args.top_k]
            candidate_count += len(selected)
            annotations = [
                row
                for row in utterances[prediction["utterance_id"]]["entities"]
                if row["correction_enabled"]
            ]
            if not annotations:
                no_entity += 1
                no_entity_fp += bool(selected)
            for annotation in annotations:
                group = "new" if annotation["entity_id"] in new_entity_ids else "old"
                totals[group] += 1
                totals["all"] += 1
                hit = annotation["surface_id"] in selected
                hits[group] += hit
                hits["all"] += hit
        results.append(
            {
                "threshold": threshold,
                "average_candidate_count": round(candidate_count / max(1, len(predictions)), 6),
                "no_entity_false_positive_rate": round(no_entity_fp / max(1, no_entity), 6),
                **{
                    f"{group}_accepted_recall_at_{args.top_k}": round(
                        hits[group] / max(1, totals[group]), 6
                    )
                    for group in ("all", "new", "old")
                },
            }
        )
    report = {
        "prediction_count": len(predictions),
        "top_k": args.top_k,
        "thresholds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
