from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_stratified(entities: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        groups[entity["entity_type"]].append(entity)
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    selected = []
    ordered_groups = sorted(groups)
    while len(selected) < count and ordered_groups:
        remaining = []
        for group in ordered_groups:
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop())
            if groups[group]:
                remaining.append(group)
        ordered_groups = remaining
    return sorted(selected, key=lambda row: row["entity_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic entity-disjoint evaluation manifest.")
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher"
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE_ROOT / "data" / "evaluation" / "new_entities.json"
    )
    args = parser.parse_args()

    entities_path = args.data_dir / "entities.jsonl"
    utterances_path = args.data_dir / "utterances.jsonl"
    entities = read_jsonl(entities_path)
    correction_entities = [row for row in entities if row["correction_enabled"]]
    if not 0 < args.count < len(correction_entities):
        raise ValueError(f"count must be between 1 and {len(correction_entities) - 1}")
    selected = select_stratified(correction_entities, args.count, args.seed)
    new_ids = {row["entity_id"] for row in selected}
    utterances = read_jsonl(utterances_path)
    heldout_utterances = [
        row
        for row in utterances
        if any(annotation["entity_id"] in new_ids for annotation in row["entities"])
    ]
    evaluation_by_split = {
        split: sorted(row["utterance_id"] for row in heldout_utterances if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "selection": "round_robin_by_entity_type",
        "new_entity_ids": sorted(new_ids),
        "new_entities": [
            {
                "entity_id": row["entity_id"],
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "domains": row["domains"],
            }
            for row in selected
        ],
        "old_entity_ids": sorted(
            row["entity_id"] for row in correction_entities if row["entity_id"] not in new_ids
        ),
        "excluded_training_utterance_ids": sorted(row["utterance_id"] for row in heldout_utterances),
        "evaluation_utterance_ids": evaluation_by_split,
        "counts": {
            "new_entities": len(new_ids),
            "old_entities": len(correction_entities) - len(new_ids),
            "excluded_utterances": len(heldout_utterances),
            **{f"evaluation_{split}_utterances": len(rows) for split, rows in evaluation_by_split.items()},
        },
        "source": {
            "entities": str(entities_path),
            "entities_sha256": sha256_file(entities_path),
            "utterances": str(utterances_path),
            "utterances_sha256": sha256_file(utterances_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
