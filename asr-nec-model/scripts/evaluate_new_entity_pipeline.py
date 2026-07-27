from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import whisper
from opencc import OpenCC
from whisper.tokenizer import get_tokenizer

from asr_nec_model.inference.corrections import apply_candidate_corrections
from evaluate_gl_dataset import cache_audio, generate
from transcribe_whisper_pilot import edit_distance, normalize_text


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_metrics(counts: Counter) -> dict:
    return {
        "mention_count": counts["mentions"],
        "expected_correction_count": counts["expected_corrections"],
        "correction_success_rate": round(
            counts["correction_success"] / max(1, counts["expected_corrections"]), 6
        ),
        "already_correct_count": counts["already_correct"],
        "already_correct_regression_rate": round(
            counts["already_correct_regression"] / max(1, counts["already_correct"]), 6
        ),
        "retrieval_recall": round(counts["retrieved"] / max(1, counts["mentions"]), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out entities through retrieval, GL and replacement.")
    parser.add_argument("--retrieval-predictions", type=Path, required=True)
    parser.add_argument("--asr-hypotheses", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--gl-checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "weights" / "base.pt")
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher"
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    converter = OpenCC("t2s")
    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    new_entity_ids = set(holdout["new_entity_ids"])
    utterances = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}
    surfaces = {row["surface_id"]: row for row in read_jsonl(args.data_dir / "entity_surfaces.jsonl")}
    retrieval = {row["utterance_id"]: row for row in read_jsonl(args.retrieval_predictions)}
    hypotheses = [row for row in read_jsonl(args.asr_hypotheses) if row["utterance_id"] in retrieval]

    model = whisper.load_model(str(args.model), device=args.device)
    checkpoint = torch.load(args.gl_checkpoint, map_location=args.device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.decoder.eval()
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    features = cache_audio(model, hypotheses, args.audio_dir, args.feature_dir, args.device)

    counts_by_group: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    predictions = []
    for index, hypothesis in enumerate(hypotheses, 1):
        utterance_id = hypothesis["utterance_id"]
        utterance = utterances[utterance_id]
        asr_text = converter.convert(hypothesis["asr_text"]).strip()
        reference = converter.convert(utterance["ref_text"]).strip()
        audio = torch.load(features[utterance_id], map_location="cpu", weights_only=True).float()
        ranked = retrieval[utterance_id]["reranked"]
        selected = [row for row in ranked if row["score"] >= args.threshold][: args.top_k]
        decisions = []
        generated = []
        for candidate in selected:
            surface = surfaces[candidate["surface_id"]]
            prompt = f"{surface['surface_text']} <EC> {asr_text}"
            prediction = generate(
                model, tokenizer, audio, prompt, args.device, args.max_new_tokens
            )
            decisions.append(
                {
                    **candidate,
                    "candidate_text": surface["surface_text"],
                    "gl_prediction": prediction,
                }
            )
            generated.append((surface["surface_text"], prediction))
        corrected, correction_decisions = apply_candidate_corrections(asr_text, generated)
        for decision_row, correction_decision in zip(
            decisions, correction_decisions, strict=True
        ):
            decision_row["applied"] = correction_decision.applied
            decision_row["decision_reason"] = correction_decision.reason

        target_annotations = [row for row in utterance["entities"] if row["correction_enabled"]]
        selected_surface_ids = {row["surface_id"] for row in selected}
        for annotation in target_annotations:
            group = "new" if annotation["entity_id"] in new_entity_ids else "old"
            for key in (group, "all"):
                group_counts = counts_by_group[key]
                group_counts["mentions"] += 1
                group_counts["retrieved"] += annotation["surface_id"] in selected_surface_ids
                baseline_correct = annotation["text"] in asr_text
                final_correct = annotation["text"] in corrected
                if baseline_correct:
                    group_counts["already_correct"] += 1
                    group_counts["already_correct_regression"] += not final_correct
                else:
                    group_counts["expected_corrections"] += 1
                    group_counts["correction_success"] += final_correct

        before_distance = edit_distance(normalize_text(reference), normalize_text(asr_text))
        after_distance = edit_distance(normalize_text(reference), normalize_text(corrected))
        totals["utterances"] += 1
        totals["reference_characters"] += len(normalize_text(reference))
        totals["before_edits"] += before_distance
        totals["after_edits"] += after_distance
        totals["modified"] += corrected != asr_text
        totals["harmful_modification"] += corrected != asr_text and after_distance > before_distance
        totals["sentence_exact_before"] += normalize_text(asr_text) == normalize_text(reference)
        totals["sentence_exact_after"] += normalize_text(corrected) == normalize_text(reference)
        if not target_annotations:
            totals["no_entity_utterances"] += 1
            totals["no_entity_modified"] += corrected != asr_text
            totals["no_entity_harmful"] += corrected != asr_text and after_distance > before_distance
        predictions.append(
            {
                "utterance_id": utterance_id,
                "asr_text": asr_text,
                "reference_text": reference,
                "corrected_text": corrected,
                "before_edit_distance": before_distance,
                "after_edit_distance": after_distance,
                "decisions": decisions,
                "new_entity_ids": [
                    row["entity_id"] for row in target_annotations if row["entity_id"] in new_entity_ids
                ],
            }
        )
        if index % 25 == 0:
            print(f"evaluated pipeline {index}/{len(hypotheses)}", flush=True)

    metrics = {
        "checkpoint": str(args.gl_checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "utterance_count": totals["utterances"],
        "top_k": args.top_k,
        "threshold": args.threshold,
        "entities": {group: group_metrics(counts_by_group[group]) for group in ("all", "new", "old")},
        "text": {
            "cer_before": round(totals["before_edits"] / max(1, totals["reference_characters"]), 6),
            "cer_after": round(totals["after_edits"] / max(1, totals["reference_characters"]), 6),
            "sentence_exact_before": round(totals["sentence_exact_before"] / max(1, totals["utterances"]), 6),
            "sentence_exact_after": round(totals["sentence_exact_after"] / max(1, totals["utterances"]), 6),
            "modification_rate": round(totals["modified"] / max(1, totals["utterances"]), 6),
            "harmful_modification_rate": round(
                totals["harmful_modification"] / max(1, totals["utterances"]), 6
            ),
        },
        "no_entity": {
            "utterance_count": totals["no_entity_utterances"],
            "modification_rate": round(
                totals["no_entity_modified"] / max(1, totals["no_entity_utterances"]), 6
            ),
            "harmful_modification_rate": round(
                totals["no_entity_harmful"] / max(1, totals["no_entity_utterances"]), 6
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
