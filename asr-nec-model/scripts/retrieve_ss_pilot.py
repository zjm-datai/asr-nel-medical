from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from asr_nec_model.models.searcher import EncodedSpeechSearcher


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def project_features(
    model: EncodedSpeechSearcher,
    features: list[torch.Tensor],
    device: str,
    amp_enabled: bool,
) -> list[torch.Tensor]:
    lengths = torch.tensor([feature.shape[0] for feature in features], device=device)
    batch = pad_sequence(features, batch_first=True).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        projected = model.project(batch)
    projected_lengths = model.projected_lengths(lengths).cpu().tolist()
    return [
        item[:length].detach().cpu()
        for item, length in zip(projected, projected_lengths, strict=True)
    ]


def score_projected(
    model: EncodedSpeechSearcher,
    utterance: torch.Tensor,
    entities: list[torch.Tensor],
    device: str,
    amp_enabled: bool,
) -> torch.Tensor:
    entity_lengths = torch.tensor([item.shape[0] for item in entities], device=device)
    entity_batch = pad_sequence(entities, batch_first=True).to(device)
    speech = utterance.unsqueeze(0).to(device).expand(len(entities), -1, -1)
    entity_padding = torch.arange(entity_batch.size(1), device=device).unsqueeze(0) >= entity_lengths.unsqueeze(1)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        attended, _ = model.cross_attn(
            query=entity_batch,
            key=speech,
            value=speech,
            need_weights=False,
        )
        token_logits = model.ffn(attended).squeeze(-1)
        token_logits = token_logits.masked_fill(entity_padding, 0.0)
        logits = token_logits.sum(dim=1) / entity_lengths.to(token_logits.dtype)
    return torch.sigmoid(logits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve real SS candidates for the GL pilot corpus.")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE_ROOT / "runs" / "ss_pilot" / "best.pt")
    parser.add_argument("--feature-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "ss_features")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_pilot")
    parser.add_argument(
        "--utterance-filter",
        type=Path,
        help="Optional JSONL manifest whose utterance_id values limit retrieval inputs.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    feature_rows = read_jsonl(args.feature_dir / "feature_manifest.jsonl")
    utterance_rows = [row for row in feature_rows if row["kind"] == "utterance"]
    unfiltered_utterance_count = len(utterance_rows)
    if args.utterance_filter is not None:
        selected_ids = {row["utterance_id"] for row in read_jsonl(args.utterance_filter)}
        utterance_rows = [row for row in utterance_rows if row["source_id"] in selected_ids]
        matched_ids = {row["source_id"] for row in utterance_rows}
        missing_ids = selected_ids - matched_ids
        if missing_ids:
            preview = ", ".join(sorted(missing_ids)[:5])
            raise ValueError(f"{len(missing_ids)} filtered utterances have no cached feature: {preview}")
    entity_rows = [row for row in feature_rows if row["kind"] == "entity"]
    surfaces = {row["surface_id"]: row for row in read_jsonl(args.data_dir / "entity_surfaces.jsonl")}
    utterance_meta = {row["utterance_id"]: row for row in read_jsonl(args.data_dir / "utterances.jsonl")}

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = EncodedSpeechSearcher(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    entity_views: dict[str, list[tuple[dict, torch.Tensor]]] = defaultdict(list)
    for row in entity_rows:
        surface_id = row["source_id"].rsplit("_v", 1)[0]
        if surface_id in surfaces:
            feature = torch.load(args.feature_dir / row["feature_path"], map_location="cpu", weights_only=True).float()
            entity_views[surface_id].append((row, feature))
    if not entity_views:
        raise ValueError("no entity feature views matched entity_surfaces.jsonl")

    output_rows = []
    amp_enabled = str(args.device).startswith("cuda")
    raw_views = [
        (surface_id, view_row, feature)
        for surface_id, items in entity_views.items()
        for view_row, feature in items
    ]
    views = []
    for start in range(0, len(raw_views), args.batch_size):
        batch = raw_views[start : start + args.batch_size]
        projected = project_features(model, [item[2] for item in batch], args.device, amp_enabled)
        views.extend(
            (surface_id, view_row, feature)
            for (surface_id, view_row, _), feature in zip(batch, projected, strict=True)
        )
        print(f"projected entity views {len(views)}/{len(raw_views)}", flush=True)
    del raw_views

    for index, utterance_row in enumerate(sorted(utterance_rows, key=lambda row: row["source_id"]), 1):
        utterance_id = utterance_row["source_id"]
        utterance = torch.load(
            args.feature_dir / utterance_row["feature_path"], map_location="cpu", weights_only=True
        ).float()
        utterance = project_features(model, [utterance], args.device, amp_enabled)[0]
        view_scores: dict[str, list[float]] = defaultdict(list)
        for start in range(0, len(views), args.batch_size):
            batch = views[start : start + args.batch_size]
            scores = score_projected(
                model, utterance, [item[2] for item in batch], args.device, amp_enabled
            )
            for (surface_id, _, _), score in zip(batch, scores.float().cpu().tolist(), strict=True):
                view_scores[surface_id].append(score)
        ranked = sorted(
            (
                {
                    "surface_id": surface_id,
                    "entity_id": surfaces[surface_id]["entity_id"],
                    "surface_text": surfaces[surface_id]["surface_text"],
                    "score": round(sum(scores) / len(scores), 8),
                    "view_scores": [round(score, 8) for score in scores],
                }
                for surface_id, scores in view_scores.items()
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        for rank, candidate in enumerate(ranked, 1):
            candidate["rank"] = rank
        meta = utterance_meta[utterance_id]
        output_rows.append(
            {
                "utterance_id": utterance_id,
                "split": meta["split"],
                "domain": meta["domain"],
                "audio_path": utterance_row["audio_path"],
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint["epoch"],
                "candidate_count": len(ranked),
                "top_k": ranked[: args.top_k],
                "all_candidates": ranked,
            }
        )
        print(f"retrieved {index}/{len(utterance_rows)}", flush=True)

    write_jsonl(args.output_dir / "ss_retrieval.jsonl", output_rows)
    summary = {
        "utterance_count": len(output_rows),
        "unfiltered_utterance_count": unfiltered_utterance_count,
        "candidate_count": len(entity_views),
        "top_k": args.top_k,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "feature_dir": str(args.feature_dir),
        "utterance_filter": str(args.utterance_filter) if args.utterance_filter else None,
    }
    (args.output_dir / "ss_retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
