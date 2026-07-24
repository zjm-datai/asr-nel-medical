from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from opencc import OpenCC
from asr_nec_model.data.alignment import aligned_entity_error_span


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMPTY = "<empty>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def aligned_error_span(reference: str, hypothesis: str, start: int, end: int) -> tuple[str, float]:
    return aligned_entity_error_span(reference, hypothesis, start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate-level GL pilot data from real SS retrieval.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--retrieval", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot" / "ss_retrieval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    converter = OpenCC("t2s")
    utterances = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}
    hypotheses = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "asr_pilot" / "asr_hypotheses.jsonl")}
    retrieval = read_jsonl(args.retrieval)
    rows = []
    sequence = 1
    retrieved_targets = Counter()
    total_targets = Counter()

    for result in retrieval:
        utterance = utterances[result["utterance_id"]]
        hypothesis = hypotheses[result["utterance_id"]]
        reference = converter.convert(utterance["ref_text"])
        asr_text = converter.convert(hypothesis["asr_text"])
        target_by_surface = {
            item["surface_id"]: item for item in utterance["entities"] if item["correction_enabled"]
        }
        for surface_id in target_by_surface:
            total_targets[utterance["split"]] += 1
        retrieved_surface_ids = {item["surface_id"] for item in result["top_k"][: args.top_k]}
        for surface_id in target_by_surface.keys() & retrieved_surface_ids:
            retrieved_targets[utterance["split"]] += 1

        for candidate in result["top_k"][: args.top_k]:
            annotation = target_by_surface.get(candidate["surface_id"])
            target = EMPTY
            confidence = 1.0
            action = "reject"
            if annotation is not None:
                target, confidence = aligned_error_span(
                    reference, asr_text, annotation["start"], annotation["end"]
                )
                if target != EMPTY:
                    action = "replace"
            rows.append(
                {
                    "gl_example_id": f"gl_real_{sequence:06d}",
                    "utterance_id": utterance["utterance_id"],
                    "split": utterance["split"],
                    "domain": utterance["domain"],
                    "speaker_role": utterance["speaker_role"],
                    "audio_path": result["audio_path"],
                    "asr_text": asr_text,
                    "reference_text": reference,
                    "candidate_surface_id": candidate["surface_id"],
                    "candidate_entity_id": candidate["entity_id"],
                    "candidate_text": candidate["surface_text"],
                    "candidate_score": candidate["score"],
                    "candidate_rank": candidate["rank"],
                    "target_error_text": target,
                    "action": action,
                    "alignment_confidence": confidence,
                    "needs_manual_review": action == "replace" and confidence < 0.5,
                    "candidate_source": "speech_searcher_best_pt",
                }
            )
            sequence += 1

    for split in ("train", "dev", "test"):
        write_jsonl(args.output_dir / f"gl_{split}.jsonl", [row for row in rows if row["split"] == split])
    write_jsonl(args.output_dir / "gl_all.jsonl", rows)
    summary = {
        "example_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "manual_review_count": sum(row["needs_manual_review"] for row in rows),
        "target_mentions": dict(total_targets),
        "retrieved_target_mentions_at_k": dict(retrieved_targets),
        "top_k": args.top_k,
        "candidate_source": "speech_searcher_best_pt",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
