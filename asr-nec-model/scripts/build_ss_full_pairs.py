from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_DIR = WORKSPACE_ROOT / "data" / "speech_searcher"
FULL_DIR = DATA_DIR / "audio_full"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    manifest = read_jsonl(FULL_DIR / "full_manifest.jsonl")
    failures = [row for row in manifest if row["status"] != "generated"]
    if failures:
        raise ValueError(f"full TTS manifest contains {len(failures)} failed rows")

    utterance_audio = {
        row["source_id"]: row["pilot_audio_path"] for row in manifest if row["kind"] == "utterance"
    }
    entity_audio: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        if row["kind"] == "entity":
            surface_id = row["source_id"].rsplit("_v", 1)[0]
            entity_audio[surface_id].append(row["pilot_audio_path"])
    entity_audio = {surface: sorted(paths) for surface, paths in entity_audio.items()}
    invalid_views = {surface: len(paths) for surface, paths in entity_audio.items() if len(paths) != 3}
    if invalid_views:
        raise ValueError(f"every entity surface must have three audio views: {invalid_views}")

    source_pairs = read_jsonl(DATA_DIR / "ss_pairs.jsonl")
    train_rows = []
    for row in source_pairs:
        if row["split"] != "train":
            continue
        utterance_path = utterance_audio.get(row["utterance_id"])
        entity_paths = entity_audio.get(row["surface_id"])
        if utterance_path is None or entity_paths is None:
            raise ValueError(f"missing audio for pair {row['pair_id']}")
        bound = {**row, "utterance_audio_path": utterance_path}
        pair_number = int(row["pair_id"].rsplit("_", 1)[1])
        bound["entity_audio_path"] = entity_paths[pair_number % len(entity_paths)]
        train_rows.append(bound)

    utterances = read_jsonl(DATA_DIR / "utterances.jsonl")
    eval_rows = []
    eval_sequence = 1
    for utterance in sorted(utterances, key=lambda row: row["utterance_id"]):
        if utterance["split"] == "train":
            continue
        target_surfaces = {
            annotation["surface_id"]
            for annotation in utterance["entities"]
            if annotation["correction_enabled"]
        }
        missing_targets = target_surfaces - entity_audio.keys()
        if missing_targets:
            raise ValueError(f"missing target surfaces for {utterance['utterance_id']}: {sorted(missing_targets)}")
        for surface_id, paths in sorted(entity_audio.items()):
            eval_rows.append(
                {
                    "pair_id": f"ss_full_eval_{eval_sequence:07d}",
                    "utterance_id": utterance["utterance_id"],
                    "utterance_audio_path": utterance_audio[utterance["utterance_id"]],
                    "surface_id": surface_id,
                    "entity_id": surface_id.removesuffix("_canonical"),
                    "entity_audio_paths": paths,
                    "label": int(surface_id in target_surfaces),
                    "pair_type": "full_candidate_evaluation",
                    "split": utterance["split"],
                }
            )
            eval_sequence += 1

    write_jsonl(FULL_DIR / "ss_train_pairs.jsonl", train_rows)
    write_jsonl(FULL_DIR / "ss_eval_pairs.jsonl", eval_rows)
    summary = {
        "candidate_surface_count": len(entity_audio),
        "entity_audio_count": sum(len(paths) for paths in entity_audio.values()),
        "train_pair_count": len(train_rows),
        "train_positive_count": sum(row["label"] for row in train_rows),
        "train_negative_count": sum(not row["label"] for row in train_rows),
        "eval_pair_count": len(eval_rows),
        "eval_positive_count": sum(row["label"] for row in eval_rows),
        "eval_negative_count": sum(not row["label"] for row in eval_rows),
        "eval_split_counts": dict(sorted(Counter(row["split"] for row in eval_rows).items())),
        "entity_audio_views_per_surface": 3,
        "train_source_pair_manifest": "ss_pairs.jsonl",
        "evaluation_scope": "all_candidates_per_utterance",
    }
    (FULL_DIR / "ss_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
