from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from opencc import OpenCC

from asr_nec_model.data.alignment import aligned_entity_error_span


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMPTY = "<empty>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build oracle-candidate GL evaluation for held-out entities.")
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher"
    )
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--asr-hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    converter = OpenCC("t2s")
    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    new_entity_ids = set(holdout["new_entity_ids"])
    utterances = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}
    hypotheses = read_jsonl(args.asr_hypotheses)
    surfaces = {row["surface_id"]: row for row in read_jsonl(args.data_dir / "entity_surfaces.jsonl")}
    new_surfaces = [
        row for row in surfaces.values() if row["entity_id"] in new_entity_ids and row["approved_for_audio"]
    ]
    rows = []
    sequence = 1
    for hypothesis in hypotheses:
        utterance = utterances[hypothesis["utterance_id"]]
        reference = converter.convert(utterance["ref_text"])
        asr_text = converter.convert(hypothesis["asr_text"])
        targets = [
            row
            for row in utterance["entities"]
            if row["correction_enabled"] and row["entity_id"] in new_entity_ids
        ]
        target_surface_ids = {row["surface_id"] for row in utterance["entities"]}
        for target in targets:
            error, confidence = aligned_entity_error_span(
                reference, asr_text, target["start"], target["end"]
            )
            action = "replace" if error != EMPTY else "reject"
            rows.append(
                {
                    "gl_example_id": f"gl_new_{sequence:06d}",
                    "utterance_id": utterance["utterance_id"],
                    "split": utterance["split"],
                    "domain": utterance["domain"],
                    "audio_path": hypothesis["audio_path"],
                    "asr_text": asr_text,
                    "reference_text": reference,
                    "candidate_surface_id": target["surface_id"],
                    "candidate_entity_id": target["entity_id"],
                    "candidate_text": target["text"],
                    "target_error_text": error,
                    "action": action,
                    "alignment_confidence": confidence,
                    "candidate_source": "heldout_oracle",
                }
            )
            sequence += 1

        negatives = [row for row in new_surfaces if row["surface_id"] not in target_surface_ids]
        if negatives:
            negative = random.Random(f"{args.seed}:{utterance['utterance_id']}").choice(negatives)
            rows.append(
                {
                    "gl_example_id": f"gl_new_{sequence:06d}",
                    "utterance_id": utterance["utterance_id"],
                    "split": utterance["split"],
                    "domain": utterance["domain"],
                    "audio_path": hypothesis["audio_path"],
                    "asr_text": asr_text,
                    "reference_text": reference,
                    "candidate_surface_id": negative["surface_id"],
                    "candidate_entity_id": negative["entity_id"],
                    "candidate_text": negative["surface_text"],
                    "target_error_text": EMPTY,
                    "action": "reject",
                    "alignment_confidence": 1.0,
                    "candidate_source": "heldout_deterministic_negative",
                }
            )
            sequence += 1
    write_jsonl(args.output, rows)
    summary = {
        "example_count": len(rows),
        "utterance_count": len({row["utterance_id"] for row in rows}),
        "replace_count": sum(row["action"] == "replace" for row in rows),
        "reject_count": sum(row["action"] == "reject" for row in rows),
        "new_entity_count": len({row["candidate_entity_id"] for row in rows}),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
