from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import torch
import whisper
from whisper.tokenizer import get_tokenizer


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


def cache_audio(model, rows: list[dict], audio_dir: Path, cache_dir: Path, device: str) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = {row["utterance_id"]: row for row in rows}
    dtype = next(model.parameters()).dtype
    for index, (utterance_id, row) in enumerate(sorted(unique.items()), 1):
        output = cache_dir / f"{utterance_id}.pt"
        if output.is_file():
            continue
        audio, sample_rate = sf.read(audio_dir / row["audio_path"], dtype="float32", always_2d=False)
        if sample_rate != 16000 or audio.ndim != 1:
            raise ValueError(f"expected mono 16 kHz audio: {row['audio_path']}")
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(torch.from_numpy(audio)), n_mels=model.dims.n_mels
        )
        with torch.inference_mode():
            encoded = model.encoder(mel.unsqueeze(0).to(device=device, dtype=dtype))[0].detach().cpu().half()
        temporary = output.with_suffix(".tmp")
        torch.save(encoded, temporary)
        temporary.replace(output)
        if index % 25 == 0:
            print(f"cached {index}/{len(unique)} GL audio features", flush=True)
    return {utterance_id: cache_dir / f"{utterance_id}.pt" for utterance_id in unique}


@torch.inference_mode()
def generate(model, tokenizer, audio: torch.Tensor, prompt: str, device: str, max_new_tokens: int) -> str:
    tokens = list(tokenizer.sot_sequence) + tokenizer.encode(prompt)
    prefix_length = len(tokens)
    for _ in range(max_new_tokens):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model.decoder(token_tensor, audio.unsqueeze(0).to(device))
        next_token = int(logits[0, -1].argmax().item())
        if next_token == tokenizer.eot:
            break
        tokens.append(next_token)
    return tokenizer.decode(tokens[prefix_length:]).strip()


def summarize(rows: list[dict]) -> dict:
    counts = Counter()
    by_entity: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        predicted_action = row["predicted_action"]
        expected_action = row["action"]
        correct_text = row["prediction"] == row["target_error_text"]
        counts["examples"] += 1
        counts["action_correct"] += predicted_action == expected_action
        counts["true_positive"] += predicted_action == expected_action == "replace"
        counts["false_positive"] += predicted_action == "replace" and expected_action == "reject"
        counts["false_negative"] += predicted_action == "reject" and expected_action == "replace"
        counts["expected_replace"] += expected_action == "replace"
        counts["predicted_replace"] += predicted_action == "replace"
        counts["exact_replacement"] += expected_action == "replace" and correct_text
        counts["exact"] += correct_text
        entity_counts = by_entity[row["candidate_entity_id"]]
        entity_counts["examples"] += 1
        entity_counts["exact"] += correct_text
    precision = counts["true_positive"] / max(1, counts["true_positive"] + counts["false_positive"])
    recall = counts["true_positive"] / max(1, counts["true_positive"] + counts["false_negative"])
    exact_precision = counts["exact_replacement"] / max(1, counts["predicted_replace"])
    exact_recall = counts["exact_replacement"] / max(1, counts["expected_replace"])
    return {
        "example_count": counts["examples"],
        "entity_count": len(by_entity),
        "exact_match": round(counts["exact"] / max(1, counts["examples"]), 6),
        "action_accuracy": round(counts["action_correct"] / max(1, counts["examples"]), 6),
        "replace_precision": round(precision, 6),
        "replace_recall": round(recall, 6),
        "replace_f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
        "replacement_exact_precision": round(exact_precision, 6),
        "replacement_exact_recall": round(exact_recall, 6),
        "false_positive": counts["false_positive"],
        "false_negative": counts["false_negative"],
        "per_entity_exact": {
            entity_id: round(item["exact"] / item["examples"], 6)
            for entity_id, item in sorted(by_entity.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GL checkpoint on an arbitrary candidate dataset.")
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "weights" / "base.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--scope", choices=("all", "new", "old"), default="all")
    args = parser.parse_args()

    rows = read_jsonl(args.evaluation_file)
    if args.holdout_manifest:
        holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
        new_entity_ids = set(holdout["new_entity_ids"])
        excluded_utterances = set(holdout["excluded_training_utterance_ids"])
        if args.scope == "new":
            rows = [row for row in rows if row["candidate_entity_id"] in new_entity_ids]
        elif args.scope == "old":
            rows = [
                row
                for row in rows
                if row["candidate_entity_id"] not in new_entity_ids
                and row["utterance_id"] not in excluded_utterances
            ]
    if not rows:
        raise ValueError("evaluation selection is empty")
    model = whisper.load_model(str(args.model), device=args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.decoder.eval()
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    feature_paths = cache_audio(model, rows, args.audio_dir, args.feature_dir, args.device)
    predictions = []
    for index, row in enumerate(rows, 1):
        audio = torch.load(feature_paths[row["utterance_id"]], map_location="cpu", weights_only=True).float()
        prompt = f"{row['candidate_text']} <EC> {row['asr_text']}"
        prediction = generate(model, tokenizer, audio, prompt, args.device, args.max_new_tokens)
        predictions.append(
            {
                **row,
                "prediction": prediction,
                "predicted_action": "reject" if prediction == EMPTY else "replace",
                "exact_match": prediction == row["target_error_text"],
            }
        )
        if index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluation_file": str(args.evaluation_file),
        "metrics": summarize(predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
