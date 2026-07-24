from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import whisper
from whisper.tokenizer import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMPTY = "<empty>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_text(row: dict) -> str:
    return f"{row['candidate_text']} <EC> {row['asr_text']}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoregressive GL pilot evaluation.")
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "weights" / "base.pt")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE_ROOT / "runs" / "gl_pilot" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot")
    parser.add_argument("--feature-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot" / "audio_features")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "runs" / "gl_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    model = whisper.load_model(str(args.model), device=args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.decoder.eval()
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    metrics = {}
    all_predictions = []
    for split in ("dev", "test"):
        rows = read_jsonl(args.data_dir / f"gl_{split}.jsonl")
        counts = Counter()
        for index, row in enumerate(rows, 1):
            audio = torch.load(
                args.feature_dir / f"{row['utterance_id']}.pt", map_location="cpu", weights_only=True
            ).float()
            prediction = generate(model, tokenizer, audio, prompt_text(row), args.device, args.max_new_tokens)
            expected = row["target_error_text"]
            predicted_action = "reject" if prediction == EMPTY else "replace"
            expected_action = row["action"]
            counts["examples"] += 1
            counts["exact"] += prediction == expected
            counts["action_correct"] += predicted_action == expected_action
            counts["true_positive"] += predicted_action == expected_action == "replace"
            counts["false_positive"] += predicted_action == "replace" and expected_action == "reject"
            counts["false_negative"] += predicted_action == "reject" and expected_action == "replace"
            counts["expected_replace"] += expected_action == "replace"
            counts["predicted_replace"] += predicted_action == "replace"
            counts["exact_replacement"] += expected_action == "replace" and prediction == expected
            counts["wrong_replacement_text"] += (
                predicted_action == expected_action == "replace" and prediction != expected
            )
            all_predictions.append(
                {
                    **row,
                    "prediction": prediction,
                    "predicted_action": predicted_action,
                    "exact_match": prediction == expected,
                }
            )
            if index % 25 == 0:
                print(f"evaluated {split} {index}/{len(rows)}", flush=True)
        precision = counts["true_positive"] / max(1, counts["true_positive"] + counts["false_positive"])
        recall = counts["true_positive"] / max(1, counts["true_positive"] + counts["false_negative"])
        exact_precision = counts["exact_replacement"] / max(1, counts["predicted_replace"])
        exact_recall = counts["exact_replacement"] / max(1, counts["expected_replace"])
        metrics[split] = {
            "example_count": counts["examples"],
            "exact_match": round(counts["exact"] / max(1, counts["examples"]), 6),
            "action_accuracy": round(counts["action_correct"] / max(1, counts["examples"]), 6),
            "replace_precision": round(precision, 6),
            "replace_recall": round(recall, 6),
            "replace_f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
            "replacement_exact_precision": round(exact_precision, 6),
            "replacement_exact_recall": round(exact_recall, 6),
            "replacement_exact_f1": round(
                2 * exact_precision * exact_recall / max(1e-12, exact_precision + exact_recall), 6
            ),
            "exact_replacement": counts["exact_replacement"],
            "wrong_replacement_text": counts["wrong_replacement_text"],
            "true_positive": counts["true_positive"],
            "false_positive": counts["false_positive"],
            "false_negative": counts["false_negative"],
        }
    write_jsonl(args.output_dir / "predictions.jsonl", all_predictions)
    (args.output_dir / "generation_metrics.json").write_text(
        json.dumps({"checkpoint_epoch": checkpoint["epoch"], "metrics": metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
