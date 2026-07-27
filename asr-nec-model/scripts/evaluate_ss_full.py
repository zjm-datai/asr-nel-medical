from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from asr_nec_model.models.searcher import EncodedSpeechSearcher

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def project(model, features: list[torch.Tensor], device: str, amp: bool) -> list[torch.Tensor]:
    lengths = torch.tensor([item.shape[0] for item in features], device=device)
    padded = pad_sequence(features, batch_first=True).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        projected = model.project(padded)
    keeps = model.projected_lengths(lengths).cpu().tolist()
    return [item[:keep].detach().cpu() for item, keep in zip(projected, keeps, strict=True)]


def score(model, utterance: torch.Tensor, entities: list[torch.Tensor], device: str, amp: bool) -> list[float]:
    lengths = torch.tensor([item.shape[0] for item in entities], device=device)
    padded = pad_sequence(entities, batch_first=True).to(device)
    speech = utterance.unsqueeze(0).to(device).expand(len(entities), -1, -1)
    padding = torch.arange(padded.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        attended, _ = model.cross_attn(query=padded, key=speech, value=speech, need_weights=False)
        logits = model.ffn(attended).squeeze(-1).masked_fill(padding, 0.0)
        logits = logits.sum(dim=1) / lengths.to(logits.dtype)
        return torch.sigmoid(logits).float().cpu().tolist()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exhaustive SS cross-attention over every entity view.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=WORKSPACE_ROOT / "data/speech_searcher/ss_features_full")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data/speech_searcher")
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--utterance-filter", type=Path, help="JSONL file containing the fixed test utterance_id set")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank", choices=("old", "full"), default="full")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    holdout = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    new_ids = set(holdout["new_entity_ids"])
    surfaces = {row["surface_id"]: row for row in read_jsonl(args.data_dir / "entity_surfaces.jsonl")}
    entities = {row["entity_id"]: row for row in read_jsonl(args.data_dir / "entities.jsonl")}
    utterances = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}
    if args.utterance_filter:
        wanted = {row["utterance_id"] for row in read_jsonl(args.utterance_filter)}
    else:
        wanted = set(holdout["evaluation_utterance_ids"].get("dev", [])) | set(holdout["evaluation_utterance_ids"].get("test", []))
    feature_rows = read_jsonl(args.feature_dir / "feature_manifest.jsonl")
    features_by_surface: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in feature_rows:
        if row["kind"] != "entity":
            continue
        surface_id = row["source_id"].rsplit("_v", 1)[0]
        surface = surfaces.get(surface_id)
        if surface and (args.bank == "full" or surface["entity_id"] not in new_ids):
            features_by_surface[surface_id].append(
                torch.load(args.feature_dir / row["feature_path"], map_location="cpu", weights_only=True).float()
            )
    if not features_by_surface:
        raise ValueError("no entity views matched the selected bank")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = EncodedSpeechSearcher(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    amp = args.device.startswith("cuda")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    projected_views: list[tuple[str, torch.Tensor]] = []
    raw = [(sid, feature) for sid, views in features_by_surface.items() for feature in views]
    for start in range(0, len(raw), args.batch_size):
        batch = raw[start : start + args.batch_size]
        projected_views.extend((sid, feature) for (sid, _), feature in zip(batch, project(model, [x[1] for x in batch], args.device, amp), strict=True))
    surface_scores: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    failures: list[dict] = []
    counts = Counter()
    type_counts: dict[str, Counter] = defaultdict(Counter)
    for uid in sorted(wanted):
        row = next(item for item in feature_rows if item["kind"] == "utterance" and item["source_id"] == uid)
        utterance = torch.load(args.feature_dir / row["feature_path"], map_location="cpu", weights_only=True).float()
        started = time.perf_counter()
        projected_utterance = project(model, [utterance], args.device, amp)[0]
        surface_scores.clear()
        for start in range(0, len(projected_views), args.batch_size):
            batch = projected_views[start : start + args.batch_size]
            values = score(model, projected_utterance, [x[1] for x in batch], args.device, amp)
            for (sid, _), value in zip(batch, values, strict=True):
                surface_scores[sid].append(value)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = sorted(((sid, sum(vals) / len(vals)) for sid, vals in surface_scores.items()), key=lambda x: x[1], reverse=True)
        top = ranked[: args.top_k]
        annotations = [x for x in utterances[uid]["entities"] if x["correction_enabled"]]
        target_ids = {x["surface_id"] for x in annotations}
        if annotations:
            for annotation in annotations:
                group = "new" if annotation["entity_id"] in new_ids else "old"
                entity_type = entities.get(annotation["entity_id"], {}).get("entity_type", "unknown")
                for key in (group, "all"):
                    counts[(key, "mentions")] += 1
                    counts[(key, "hit1")] += annotation["surface_id"] in {sid for sid, _ in ranked[:1]}
                    counts[(key, "hit5")] += annotation["surface_id"] in {sid for sid, _ in ranked[:5]}
                    counts[(key, "hit10")] += annotation["surface_id"] in {sid for sid, _ in ranked[:10]}
                type_counts[entity_type]["mentions"] += 1
                type_counts[entity_type]["hit5"] += annotation["surface_id"] in {sid for sid, _ in ranked[:5]}
                if annotation["surface_id"] not in {sid for sid, _ in ranked[: args.top_k]}:
                    failures.append({"utterance_id": uid, "category": "ss_not_retrieved", "surface_id": annotation["surface_id"], "entity_id": annotation["entity_id"], "top_candidates": [{"surface_id": sid, "score": round(value, 6)} for sid, value in top]})
        else:
            counts[("no_entity", "utterances")] += 1
            counts[("no_entity", "false_positive")] += bool(ranked and ranked[0][1] >= args.threshold)
    metrics = {"bank": args.bank, "threshold": args.threshold, "threshold_policy": "fixed", "checkpoint": str(args.checkpoint), "view_count": len(projected_views), "surface_count": len(features_by_surface), "latency_ms": {"p50": round(percentile(latencies, .5), 3), "p95": round(percentile(latencies, .95), 3)}, "gpu_memory_peak_mb": round(torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20, 2) if args.device.startswith("cuda") else None, "groups": {}, "entity_type": {}, "no_entity": {"utterance_count": counts[("no_entity", "utterances")], "false_positive_rate": round(counts[("no_entity", "false_positive")] / max(1, counts[("no_entity", "utterances")]), 6)}}
    for group in ("all", "new", "old"):
        total = counts[(group, "mentions")]
        metrics["groups"][group] = {f"recall_at_{k}": round(counts[(group, f"hit{k}")] / max(1, total), 6) for k in (1, 5, 10)} | {"mention_count": total}
    for entity_type, values in sorted(type_counts.items()):
        metrics["entity_type"][entity_type] = {"mention_count": values["mentions"], "recall_at_5": round(values["hit5"] / max(1, values["mentions"]), 6)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "failures.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
