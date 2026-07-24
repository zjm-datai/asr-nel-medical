from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from asr_nec_model.models.searcher import EncodedSpeechSearcher

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@lru_cache(maxsize=4096)
def load_feature(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


class PairViewDataset(Dataset):
    def __init__(self, rows: list[dict], audio_to_feature: dict[str, Path], evaluation: bool):
        self.items = []
        for row in rows:
            entity_paths = row["entity_audio_paths"] if evaluation else [row["entity_audio_path"]]
            for entity_path in entity_paths:
                self.items.append(
                    {
                        "pair_id": row["pair_id"],
                        "utterance_id": row["utterance_id"],
                        "surface_id": row["surface_id"],
                        "split": row["split"],
                        "label": float(row["label"]),
                        "utterance_feature": audio_to_feature[row["utterance_audio_path"]],
                        "entity_feature": audio_to_feature[entity_path],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        utterance = load_feature(str(item["utterance_feature"]))
        entity = load_feature(str(item["entity_feature"]))
        return {**item, "utterance": utterance, "entity": entity}


def collate_pairs(batch: list[dict]) -> dict:
    utterances = [item["utterance"] for item in batch]
    entities = [item["entity"] for item in batch]
    return {
        "utterances": pad_sequence(utterances, batch_first=True),
        "entities": pad_sequence(entities, batch_first=True),
        "utterance_lengths": torch.tensor([item.shape[0] for item in utterances], dtype=torch.long),
        "entity_lengths": torch.tensor([item.shape[0] for item in entities], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "metadata": [
            {key: item[key] for key in ("pair_id", "utterance_id", "surface_id", "split", "label")}
            for item in batch
        ],
    }


def move_inputs(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["utterances"].to(device, non_blocking=True),
        batch["entities"].to(device, non_blocking=True),
        batch["utterance_lengths"].to(device, non_blocking=True),
        batch["entity_lengths"].to(device, non_blocking=True),
    )


def retrieval_metrics(pair_rows: list[dict], pair_scores: dict[str, float], threshold: float) -> dict:
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["split"], row["utterance_id"])].append(
            (row["surface_id"], pair_scores[row["pair_id"]], bool(row["label"]))
        )
    metrics = {}
    for split in sorted({row["split"] for row in pair_rows}):
        positive_utterances = 0
        total_mentions = 0
        hit_utterances = {1: 0, 5: 0, 10: 0}
        hit_mentions = {1: 0, 5: 0, 10: 0}
        no_entity_count = 0
        no_entity_false_positive = 0
        for (row_split, _), candidates in grouped.items():
            if row_split != split:
                continue
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            positive_surfaces = {surface for surface, _, label in candidates if label}
            if positive_surfaces:
                positive_utterances += 1
                total_mentions += len(positive_surfaces)
                for k in (1, 5, 10):
                    retrieved = {surface for surface, _, _ in ranked[:k]}
                    overlap = positive_surfaces & retrieved
                    hit_utterances[k] += bool(overlap)
                    hit_mentions[k] += len(overlap)
            else:
                no_entity_count += 1
                no_entity_false_positive += ranked[0][1] >= threshold
        split_metrics = {
            f"recall_at_{k}": round(hit_utterances[k] / max(1, positive_utterances), 6) for k in (1, 5, 10)
        }
        split_metrics.update(
            {f"mention_recall_at_{k}": round(hit_mentions[k] / max(1, total_mentions), 6) for k in (1, 5, 10)}
        )
        split_metrics.update(
            {
                "positive_utterance_count": positive_utterances,
                "positive_mention_count": total_mentions,
                "no_entity_utterance_count": no_entity_count,
                "no_entity_false_positive_rate": round(no_entity_false_positive / max(1, no_entity_count), 6),
            }
        )
        metrics[split] = split_metrics
    return metrics


def evaluate(
    model: EncodedSpeechSearcher,
    loader: DataLoader,
    pair_rows: list[dict],
    device: torch.device,
    amp_enabled: bool,
    threshold: float,
) -> dict:
    model.eval()
    view_scores = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            inputs = move_inputs(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                scores = model.score(*inputs).float().cpu().tolist()
            for metadata, score in zip(batch["metadata"], scores, strict=True):
                view_scores[metadata["pair_id"]].append(score)
    pair_scores = {pair_id: sum(values) / len(values) for pair_id, values in view_scores.items()}
    return retrieval_metrics(pair_rows, pair_scores, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpeechSearcher on cached Whisper states.")
    parser.add_argument(
        "--data-dir",
        "--pilot-dir",
        dest="data_dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot",
    )
    parser.add_argument("--feature-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "ss_features")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "runs" / "ss_pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--san-dim", type=int, default=256)
    parser.add_argument("--ffn-hidden", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-pairs", type=int, help="Smoke-test only; omit for real training")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = read_jsonl(args.feature_dir / "feature_manifest.jsonl")
    audio_to_feature = {
        row["audio_path"]: args.feature_dir / row["feature_path"] for row in feature_rows
    }
    train_rows = read_jsonl(args.data_dir / "ss_train_pairs.jsonl")
    if args.max_train_pairs:
        train_rows = train_rows[: args.max_train_pairs]
    eval_rows = read_jsonl(args.data_dir / "ss_eval_pairs.jsonl")
    dev_rows = [row for row in eval_rows if row["split"] == "dev"]
    test_rows = [row for row in eval_rows if row["split"] == "test"]
    if not dev_rows or not test_rows:
        raise ValueError("evaluation pairs must contain both dev and test splits")
    train_set = PairViewDataset(train_rows, audio_to_feature, evaluation=False)
    dev_set = PairViewDataset(dev_rows, audio_to_feature, evaluation=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
        persistent_workers=args.num_workers > 0,
    )

    encoder_dims = {row["encoder_dim"] for row in feature_rows}
    if len(encoder_dims) != 1:
        raise ValueError(f"inconsistent encoder dimensions: {encoder_dims}")
    model = EncodedSpeechSearcher(
        encoder_dim=encoder_dims.pop(),
        san_dim=args.san_dim,
        ffn_hidden=args.ffn_hidden,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positive_count = sum(row["label"] for row in train_rows)
    negative_count = len(train_rows) - positive_count
    pos_weight = torch.tensor(negative_count / max(1, positive_count), device=device)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_recall = -1.0
    patience_used = 0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_recall = checkpoint.get("best_dev_recall_at_5", -1.0)
        history = checkpoint.get("history", [])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            inputs = move_inputs(batch, device)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(*inputs)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
        metrics = evaluate(model, dev_loader, dev_rows, device, amp_enabled, args.threshold)
        epoch_result = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, total_examples), 6),
            "metrics": metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        dev_recall = metrics["dev"]["recall_at_5"]
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config,
            "training_args": vars(args),
            "best_dev_recall_at_5": max(best_recall, dev_recall),
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if dev_recall > best_recall:
            best_recall = dev_recall
            patience_used = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            patience_used += 1
        (args.output_dir / "metrics.json").write_text(
            json.dumps({"best_dev_recall_at_5": best_recall, "history": history}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if patience_used >= args.patience:
            print(f"early stopping after {patience_used} epochs without dev Recall@5 improvement", flush=True)
            break

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_set = PairViewDataset(test_rows, audio_to_feature, evaluation=True)
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
        persistent_workers=args.num_workers > 0,
    )
    test_metrics = evaluate(model, test_loader, test_rows, device, amp_enabled, args.threshold)["test"]
    final_metrics = {
        "best_dev_recall_at_5": best_recall,
        "best_epoch": best_checkpoint["epoch"],
        "history": history,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_checkpoint["epoch"], "test": test_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
