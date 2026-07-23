from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from opencc import OpenCC


PUNCTUATION_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_ASR_OUTPUT = WORKSPACE_ROOT / "data" / "speech_searcher" / "asr_pilot"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = TRADITIONAL_TO_SIMPLIFIED.convert(unicodedata.normalize("NFKC", text)).lower().replace("_", "")
    return PUNCTUATION_RE.sub("", text)


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def load_existing(path: Path, model_name: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {
        row["utterance_id"]: row
        for row in rows
        if row.get("asr_model") == model_name and row.get("status") == "transcribed"
    }


def build_result(source: dict, hypothesis: str, model_name: str) -> dict:
    return {
        "utterance_id": source["source_id"],
        "split": source["split"],
        "domain": source["domain"],
        "speaker_role": source["speaker_role"],
        "sample_type": source["sample_type"],
        "voice": source["voice"],
        "audio_path": source["pilot_audio_path"],
        "ref_text": source["ref_text"],
        "tts_text": source["tts_text"],
        "asr_text": hypothesis.strip(),
        "asr_model": model_name,
        "status": "transcribed",
    }


def transcribe_batch_transformers(
    batch: list[dict],
    audio_root: Path,
    processor: Any,
    model: Any,
    model_name: str,
) -> list[dict]:
    audio_arrays = []
    for row in batch:
        audio, sample_rate = sf.read(audio_root / row["pilot_audio_path"], dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"unexpected audio format for {row['source_id']}: {sample_rate}, shape={audio.shape}")
        audio_arrays.append(audio)
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    with torch.inference_mode():
        generated = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
            language="zh",
            task="transcribe",
            max_new_tokens=128,
        )
    hypotheses = processor.batch_decode(generated, skip_special_tokens=True)
    return [build_result(source, hypothesis, model_name) for source, hypothesis in zip(batch, hypotheses, strict=True)]


def transcribe_batch_openai(batch: list[dict], audio_root: Path, model: Any, model_name: str) -> list[dict]:
    results = []
    for source in batch:
        audio, sample_rate = sf.read(audio_root / source["pilot_audio_path"], dtype="float32")
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"unexpected audio format for {source['source_id']}: {sample_rate}, shape={audio.shape}")
        decoded = model.transcribe(
            audio,
            language="zh",
            task="transcribe",
            fp16=False,
            temperature=0,
            condition_on_previous_text=False,
            verbose=False,
        )
        results.append(build_result(source, decoded["text"], model_name))
    return results


def evaluate(rows: list[dict], utterance_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    total_edits = 0
    total_characters = 0
    exact_count = 0
    entity_totals = Counter()
    entity_hits = Counter()
    errors = []
    per_domain = defaultdict(lambda: [0, 0])
    per_role = defaultdict(lambda: [0, 0])
    per_voice = defaultdict(lambda: [0, 0])
    per_sample_type = defaultdict(lambda: [0, 0])
    for row in rows:
        reference = normalize_text(row["ref_text"])
        hypothesis = normalize_text(row["asr_text"])
        edits = edit_distance(reference, hypothesis)
        total_edits += edits
        total_characters += len(reference)
        exact_count += reference == hypothesis
        per_domain[row["domain"]][0] += edits
        per_domain[row["domain"]][1] += len(reference)
        per_role[row["speaker_role"]][0] += edits
        per_role[row["speaker_role"]][1] += len(reference)
        per_voice[row["voice"]][0] += edits
        per_voice[row["voice"]][1] += len(reference)
        per_sample_type[row["sample_type"]][0] += edits
        per_sample_type[row["sample_type"]][1] += len(reference)
        source = utterance_by_id[row["utterance_id"]]
        for annotation in source["entities"]:
            tier = "correction" if annotation["correction_enabled"] else "link_only"
            entity_totals[tier] += 1
            entity_totals[f"type:{annotation['type']}"] += 1
            normalized_entity = normalize_text(annotation["text"])
            preserved = normalized_entity in hypothesis
            if preserved:
                entity_hits[tier] += 1
                entity_hits[f"type:{annotation['type']}"] += 1
            else:
                errors.append(
                    {
                        "utterance_id": row["utterance_id"],
                        "split": row["split"],
                        "domain": row["domain"],
                        "speaker_role": row["speaker_role"],
                        "voice": row["voice"],
                        "entity_id": annotation["entity_id"],
                        "surface_id": annotation["surface_id"],
                        "entity_text": annotation["text"],
                        "entity_type": annotation["type"],
                        "correction_enabled": annotation["correction_enabled"],
                        "ref_text": row["ref_text"],
                        "asr_text": row["asr_text"],
                    }
                )
    summary = {
        "utterance_count": len(rows),
        "character_count": total_characters,
        "character_errors": total_edits,
        "cer": round(total_edits / max(1, total_characters), 6),
        "normalized_exact_match_count": exact_count,
        "normalized_exact_match_rate": round(exact_count / max(1, len(rows)), 6),
        "entity_mention_count": sum(entity_totals[key] for key in ("correction", "link_only")),
        "entity_error_count": len(errors),
        "entity_exact_recall": {
            key: round(entity_hits[key] / total, 6) for key, total in sorted(entity_totals.items()) if total
        },
        "entity_counts": dict(sorted(entity_totals.items())),
        "cer_by_domain": {key: round(value[0] / max(1, value[1]), 6) for key, value in sorted(per_domain.items())},
        "cer_by_speaker_role": {key: round(value[0] / max(1, value[1]), 6) for key, value in sorted(per_role.items())},
        "cer_by_voice": {key: round(value[0] / max(1, value[1]), 6) for key, value in sorted(per_voice.items())},
        "cer_by_sample_type": {
            key: round(value[0] / max(1, value[1]), 6) for key, value in sorted(per_sample_type.items())
        },
    }
    return summary, errors


def build_gl_oracle_seed(rows: list[dict], utterance_by_id: dict[str, dict]) -> list[dict]:
    records = []
    for sequence, row in enumerate(rows, start=1):
        source = utterance_by_id[row["utterance_id"]]
        correction_candidates = []
        link_entities = []
        normalized_hypothesis = normalize_text(row["asr_text"])
        for annotation in source["entities"]:
            item = {
                "entity_id": annotation["entity_id"],
                "surface_id": annotation["surface_id"],
                "surface_text": annotation["text"],
                "entity_type": annotation["type"],
                "preserved_in_asr": normalize_text(annotation["text"]) in normalized_hypothesis,
            }
            if annotation["correction_enabled"]:
                correction_candidates.append(item)
            else:
                link_entities.append(item)
        records.append(
            {
                "gl_example_id": f"gl_pilot_{sequence:04d}",
                "utterance_id": row["utterance_id"],
                "split": row["split"],
                "domain": row["domain"],
                "speaker_role": row["speaker_role"],
                "voice": row["voice"],
                "audio_path": row["audio_path"],
                "asr_text": row["asr_text"],
                "target_text": row["ref_text"],
                "oracle_candidates": correction_candidates,
                "link_entities": link_entities,
                "needs_entity_correction": any(not item["preserved_in_asr"] for item in correction_candidates),
                "candidate_source": "reference_annotation_oracle_not_ss_output",
                "training_status": "pipeline_debug_only_pending_ss_candidates",
                "asr_model": row["asr_model"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe and evaluate the TTS pilot with the target Whisper model.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--pilot-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASR_OUTPUT)
    parser.add_argument("--backend", choices=("openai", "transformers"), default="openai")
    parser.add_argument("--model", default="base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.output_dir.resolve() == DEFAULT_ASR_OUTPUT.resolve():
        raise ValueError("--limit is a smoke-test option and requires a separate --output-dir")

    pilot_rows = [row for row in read_jsonl(args.pilot_dir / "pilot_manifest.jsonl") if row["kind"] == "utterance"]
    pilot_rows.sort(key=lambda row: row["source_id"])
    if args.limit is not None:
        pilot_rows = pilot_rows[: args.limit]
    utterance_by_id = {
        row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")
    }
    output_path = args.output_dir / "asr_hypotheses.jsonl"
    model_name = f"{args.backend}:{args.model}"
    existing = load_existing(output_path, model_name)

    completed = dict(existing)
    pending = [row for row in pilot_rows if row["source_id"] not in completed]
    processor = None
    model = None
    if pending:
        print(f"loading {model_name} on CPU", flush=True)
        if args.backend == "openai":
            import whisper

            model = whisper.load_model(args.model, device="cpu")
        else:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = WhisperProcessor.from_pretrained(args.model)
            model = WhisperForConditionalGeneration.from_pretrained(args.model)
            model.eval()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        if args.backend == "openai":
            batch_results = transcribe_batch_openai(batch, args.pilot_dir, model, model_name)
        else:
            batch_results = transcribe_batch_transformers(batch, args.pilot_dir, processor, model, model_name)
        for result in batch_results:
            completed[result["utterance_id"]] = result
        selected_results = [completed[row["source_id"]] for row in pilot_rows if row["source_id"] in completed]
        write_jsonl(output_path, selected_results)
        print(f"transcribed {len(selected_results)}/{len(pilot_rows)}", flush=True)

    rows = [completed[row["source_id"]] for row in pilot_rows]
    summary, errors = evaluate(rows, utterance_by_id)
    gl_seed = build_gl_oracle_seed(rows, utterance_by_id)
    summary["asr_model"] = model_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "gl_oracle_seed.jsonl", gl_seed)
    if errors:
        with (args.output_dir / "entity_error_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
            writer.writeheader()
            writer.writerows(errors)
    else:
        (args.output_dir / "entity_error_cases.csv").write_text("", encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
