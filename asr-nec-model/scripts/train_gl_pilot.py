from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import whisper
from whisper.tokenizer import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMPTY = "<empty>"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prompt_text(row: dict) -> str:
    return f"{row['candidate_text']} <EC> {row['asr_text']}"


def cache_audio_features(model, rows: list[dict], audio_dir: Path, cache_dir: Path, device: str, batch_size: int) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = {row["utterance_id"]: row for row in rows}
    pending = [(uid, row) for uid, row in sorted(unique.items()) if not (cache_dir / f"{uid}.pt").is_file()]
    dtype = next(model.parameters()).dtype
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        mels = []
        for _, row in batch:
            audio, sample_rate = sf.read(audio_dir / row["audio_path"], dtype="float32", always_2d=False)
            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(f"expected mono 16 kHz audio: {row['audio_path']}")
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(audio)), n_mels=model.dims.n_mels)
            mels.append(mel)
        with torch.inference_mode():
            encoded = model.encoder(torch.stack(mels).to(device=device, dtype=dtype)).detach().cpu().half()
        for (uid, _), feature in zip(batch, encoded, strict=True):
            temporary = cache_dir / f"{uid}.tmp"
            # A batch slice retains the full batch storage unless materialized.
            torch.save(feature.clone(), temporary)
            temporary.replace(cache_dir / f"{uid}.pt")
        print(f"cached GL audio features {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    return {uid: cache_dir / f"{uid}.pt" for uid in unique}


class GLDataset(Dataset):
    def __init__(self, rows: list[dict], feature_paths: dict[str, Path], tokenizer, max_tokens: int):
        self.rows = rows
        self.feature_paths = feature_paths
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prefix = list(self.tokenizer.sot_sequence) + self.tokenizer.encode(prompt_text(row))
        target = self.tokenizer.encode(row["target_error_text"]) + [self.tokenizer.eot]
        full = (prefix + target)[: self.max_tokens]
        tokens = torch.tensor(full[:-1], dtype=torch.long)
        labels = torch.tensor(full[1:], dtype=torch.long)
        labels[: max(0, len(prefix) - 1)] = -100
        return {
            "audio": torch.load(self.feature_paths[row["utterance_id"]], map_location="cpu", weights_only=True).float(),
            "tokens": tokens,
            "labels": labels,
            "action": row["action"],
        }


def collate(batch: list[dict], eot: int) -> dict:
    return {
        "audio": torch.stack([row["audio"] for row in batch]),
        "tokens": pad_sequence([row["tokens"] for row in batch], batch_first=True, padding_value=eot),
        "labels": pad_sequence([row["labels"] for row in batch], batch_first=True, padding_value=-100),
    }


def evaluate(model, loader: DataLoader, device: str, amp_enabled: bool) -> dict:
    model.decoder.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model.decoder(tokens, audio)
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="sum")
            mask = labels.ne(-100)
            total_loss += loss.item()
            total_tokens += mask.sum().item()
            correct_tokens += (logits.argmax(dim=-1).eq(labels) & mask).sum().item()
    return {
        "loss": round(total_loss / max(1, total_tokens), 6),
        "target_token_accuracy": round(correct_tokens / max(1, total_tokens), 6),
        "target_token_count": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the OpenAI Whisper decoder as a GL pilot model.")
    parser.add_argument("--model", type=Path, default=WORKSPACE_ROOT / "weights" / "base.pt")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot")
    parser.add_argument("--audio-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "runs" / "gl_pilot")
    parser.add_argument("--feature-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot" / "audio_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.data_dir / "gl_train.jsonl")
    dev_rows = read_jsonl(args.data_dir / "gl_dev.jsonl")
    test_rows = read_jsonl(args.data_dir / "gl_test.jsonl")
    all_rows = train_rows + dev_rows + test_rows
    model = whisper.load_model(str(args.model), device=args.device)
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    feature_paths = cache_audio_features(
        model, all_rows, args.audio_dir, args.feature_dir, args.device, args.feature_batch_size
    )

    train_set = GLDataset(train_rows, feature_paths, tokenizer, args.max_tokens)
    dev_set = GLDataset(dev_rows, feature_paths, tokenizer, args.max_tokens)
    test_set = GLDataset(test_rows, feature_paths, tokenizer, args.max_tokens)
    action_counts = {action: sum(row["action"] == action for row in train_rows) for action in {"replace", "reject"}}
    weights = [1.0 / max(1, action_counts[row["action"]]) for row in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    loader_args = {"num_workers": args.num_workers, "pin_memory": args.device.startswith("cuda")}
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        collate_fn=lambda batch: collate(batch, tokenizer.eot), **loader_args
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.eot), **loader_args
    )
    test_loader = DataLoader(
        test_set, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.eot), **loader_args
    )

    parameters = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    history = []
    best_dev_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.decoder.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in train_loader:
            audio = batch["audio"].to(args.device, non_blocking=True)
            tokens = batch["tokens"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                logits = model.decoder(tokens, audio)
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = labels.ne(-100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
        dev_metrics = evaluate(model, dev_loader, args.device, args.device.startswith("cuda"))
        result = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, total_tokens), 6),
            "dev": dev_metrics,
        }
        history.append(result)
        checkpoint = {
            "epoch": epoch,
            "decoder_state": model.decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_args": vars(args),
            "history": history,
            "best_dev_loss": min(best_dev_loss, dev_metrics["loss"]),
            "train_action_counts": action_counts,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            torch.save({key: value for key, value in checkpoint.items() if key != "optimizer_state"}, args.output_dir / "best.pt")
        (args.output_dir / "metrics.json").write_text(
            json.dumps({"best_dev_loss": best_dev_loss, "history": history}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)

    best = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=False)
    model.decoder.load_state_dict(best["decoder_state"])
    test_metrics = evaluate(model, test_loader, args.device, args.device.startswith("cuda"))
    final = {"best_epoch": best["epoch"], "best_dev_loss": best_dev_loss, "history": history, "test": test_metrics}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best["epoch"], "test": test_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
