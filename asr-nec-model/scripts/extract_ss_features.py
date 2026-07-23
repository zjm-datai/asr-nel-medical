from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_FEATURE_OUTPUT = WORKSPACE_ROOT / "data" / "speech_searcher" / "ss_features"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_relative_path(row: dict) -> Path:
    return Path("features") / row["kind"] / f"{row['source_id']}.pt"


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError(f"expected mono 16 kHz WAV, got sample_rate={sample_rate}, shape={audio.shape}: {path}")
    encoder_frames = max(1, min(1500, (len(audio) + 319) // 320))
    return torch.from_numpy(audio), encoder_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen OpenAI Whisper encoder states for SS training.")
    parser.add_argument("--pilot-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--model", default="base", help="OpenAI Whisper model name or local .pt checkpoint")
    parser.add_argument("--download-root", type=Path, default=Path.home() / ".cache" / "whisper")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Smoke-test only; omit for the complete cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.output_dir.resolve() == DEFAULT_FEATURE_OUTPUT.resolve():
        raise ValueError("--limit is a smoke-test option and requires a separate --output-dir")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    manifest = read_jsonl(args.pilot_dir / "pilot_manifest.jsonl")
    manifest.sort(key=lambda row: (row["kind"], row["source_id"]))
    if args.limit is not None:
        manifest = manifest[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading OpenAI Whisper {args.model} on {args.device}", flush=True)
    model = whisper.load_model(args.model, device=args.device, download_root=str(args.download_root))
    model.eval()
    model_dtype = next(model.parameters()).dtype

    results_by_id = {}
    feature_manifest_path = args.output_dir / "feature_manifest.jsonl"
    if feature_manifest_path.exists() and not args.overwrite:
        results_by_id = {row["source_id"]: row for row in read_jsonl(feature_manifest_path)}

    pending = []
    for row in manifest:
        relative = output_relative_path(row)
        cached = args.output_dir / relative
        existing = results_by_id.get(row["source_id"])
        if existing and cached.is_file():
            try:
                tensor = torch.load(cached, map_location="cpu", weights_only=True)
                if tensor.ndim == 2 and tensor.shape[0] == existing["frames"]:
                    continue
            except (RuntimeError, OSError, EOFError):
                pass
        pending.append(row)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        mels = []
        lengths = []
        for row in batch:
            audio, encoder_frames = load_audio(args.pilot_dir / row["pilot_audio_path"])
            padded = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)
            mels.append(mel)
            lengths.append(encoder_frames)
        mel_batch = torch.stack(mels).to(device=args.device, dtype=model_dtype)
        with torch.inference_mode():
            hidden_batch = model.encoder(mel_batch).detach().cpu()
        for row, hidden, frames in zip(batch, hidden_batch, lengths, strict=True):
            hidden = hidden[:frames].to(torch.float16).contiguous()
            relative = output_relative_path(row)
            output_path = args.output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp")
            torch.save(hidden, temporary)
            temporary.replace(output_path)
            results_by_id[row["source_id"]] = {
                "source_id": row["source_id"],
                "kind": row["kind"],
                "audio_path": row["pilot_audio_path"],
                "feature_path": relative.as_posix(),
                "frames": frames,
                "encoder_dim": hidden.shape[1],
                "whisper_model": args.model,
                "dtype": "float16",
            }
        ordered = [results_by_id[row["source_id"]] for row in manifest if row["source_id"] in results_by_id]
        write_jsonl(feature_manifest_path, ordered)
        print(f"cached {len(results_by_id)}/{len(manifest)}", flush=True)

    ordered = [results_by_id[row["source_id"]] for row in manifest]
    summary = {
        "feature_count": len(ordered),
        "utterance_count": sum(row["kind"] == "utterance" for row in ordered),
        "entity_audio_count": sum(row["kind"] == "entity" for row in ordered),
        "encoder_dim": sorted({row["encoder_dim"] for row in ordered}),
        "whisper_model": args.model,
        "dtype": "float16",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
