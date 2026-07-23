from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_DIR = WORKSPACE_ROOT / "data" / "speech_searcher"
PILOT_DIR = DATA_DIR / "audio_pilot"
SEED = 20260723


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ranked_negatives(candidates: list[dict], targets: list[dict], domain: str, rng: random.Random) -> list[dict]:
    target_names = {item["canonical_name"] for item in targets}
    target_types = {item["entity_type"] for item in targets}
    confusable_names = {name for item in targets for name in item["confusables"]}
    buckets = defaultdict(list)
    for candidate in candidates:
        if candidate["canonical_name"] in target_names:
            continue
        if candidate["canonical_name"] in confusable_names:
            kind = "confusable"
        elif candidate["entity_type"] in target_types:
            kind = "same_type"
        elif domain in candidate["domains"]:
            kind = "same_domain"
        else:
            kind = "cross_domain"
        buckets[kind].append(candidate)
    ordered = []
    for kind in ("confusable", "same_type", "same_domain", "cross_domain"):
        rng.shuffle(buckets[kind])
        ordered.extend(buckets[kind])
    return ordered


def main() -> None:
    rng = random.Random(SEED)
    entities = {row["entity_id"]: row for row in read_jsonl(DATA_DIR / "entities.jsonl")}
    utterances = {row["utterance_id"]: row for row in read_jsonl(DATA_DIR / "utterances.jsonl")}
    pilot = read_jsonl(PILOT_DIR / "pilot_manifest.jsonl")
    pilot_utterances = {row["source_id"]: row for row in pilot if row["kind"] == "utterance"}
    entity_audio = defaultdict(list)
    for row in pilot:
        if row["kind"] == "entity":
            surface_id = row["source_id"].rsplit("_v", 1)[0]
            entity_audio[surface_id].append(row["pilot_audio_path"])
    if any(len(paths) != 3 for paths in entity_audio.values()):
        raise ValueError("every pilot entity surface must have exactly three audio variants")

    surface_entity = {}
    for entity in entities.values():
        surface_id = f"{entity['entity_id']}_canonical"
        if surface_id in entity_audio:
            surface_entity[surface_id] = entity
    candidates = list(surface_entity.values())

    train_pairs = []
    eval_pairs = []
    train_sequence = 1
    eval_sequence = 1
    for utterance_id, pilot_row in sorted(pilot_utterances.items()):
        utterance = utterances[utterance_id]
        target_annotations = [item for item in utterance["entities"] if item["correction_enabled"]]
        target_surface_ids = {item["surface_id"] for item in target_annotations}
        missing = target_surface_ids - entity_audio.keys()
        if missing:
            raise ValueError(f"missing positive entity audio for {utterance_id}: {sorted(missing)}")
        if utterance["split"] == "train":
            for annotation in target_annotations:
                for entity_path in sorted(entity_audio[annotation["surface_id"]]):
                    train_pairs.append(
                        {
                            "pair_id": f"ss_pilot_train_{train_sequence:06d}",
                            "utterance_id": utterance_id,
                            "utterance_audio_path": pilot_row["pilot_audio_path"],
                            "surface_id": annotation["surface_id"],
                            "entity_id": annotation["entity_id"],
                            "entity_audio_path": entity_path,
                            "label": 1,
                            "pair_type": "positive",
                            "split": "train",
                        }
                    )
                    train_sequence += 1
            target_entities = [entities[item["entity_id"]] for item in target_annotations]
            negatives = ranked_negatives(candidates, target_entities, utterance["domain"], rng)
            negative_count = 30 * len(target_annotations) if target_annotations else 10
            for index in range(negative_count):
                candidate = negatives[index % len(negatives)]
                surface_id = f"{candidate['entity_id']}_canonical"
                paths = sorted(entity_audio[surface_id])
                train_pairs.append(
                    {
                        "pair_id": f"ss_pilot_train_{train_sequence:06d}",
                        "utterance_id": utterance_id,
                        "utterance_audio_path": pilot_row["pilot_audio_path"],
                        "surface_id": surface_id,
                        "entity_id": candidate["entity_id"],
                        "entity_audio_path": paths[index % len(paths)],
                        "label": 0,
                        "pair_type": "no_correction_entity" if not target_annotations else "hard_negative",
                        "split": "train",
                    }
                )
                train_sequence += 1
        else:
            for surface_id, candidate in sorted(surface_entity.items()):
                eval_pairs.append(
                    {
                        "pair_id": f"ss_pilot_eval_{eval_sequence:06d}",
                        "utterance_id": utterance_id,
                        "utterance_audio_path": pilot_row["pilot_audio_path"],
                        "surface_id": surface_id,
                        "entity_id": candidate["entity_id"],
                        "entity_audio_paths": sorted(entity_audio[surface_id]),
                        "label": int(surface_id in target_surface_ids),
                        "split": utterance["split"],
                    }
                )
                eval_sequence += 1

    write_jsonl(PILOT_DIR / "ss_train_pairs.jsonl", train_pairs)
    write_jsonl(PILOT_DIR / "ss_eval_pairs.jsonl", eval_pairs)
    summary = {
        "candidate_surface_count": len(entity_audio),
        "entity_audio_count": sum(len(paths) for paths in entity_audio.values()),
        "train_pair_count": len(train_pairs),
        "train_positive_count": sum(row["label"] for row in train_pairs),
        "train_negative_count": sum(not row["label"] for row in train_pairs),
        "eval_pair_count": len(eval_pairs),
        "eval_positive_count": sum(row["label"] for row in eval_pairs),
        "eval_negative_count": sum(not row["label"] for row in eval_pairs),
        "eval_split_counts": dict(Counter(row["split"] for row in eval_pairs)),
        "entity_audio_views_per_surface": 3,
    }
    (PILOT_DIR / "ss_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
