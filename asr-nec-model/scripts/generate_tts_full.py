from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data" / "speech_searcher"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "audio_full"
DEFAULT_PILOT_DIR = DEFAULT_DATA_DIR / "audio_pilot"
TRAIN_VOICES = ("my_zero_shot_spk", "yinyao", "yexuejie", "liqiong")
SPLIT_VOICES = {"dev": ("doctor_woman",), "test": ("haiwenyuan",)}
ENTITY_VOICES = ("yinyao", "doctor_woman", "haiwenyuan")
SPEED_BY_SLOT = {"voice_1": 1.0, "voice_2": 0.9, "voice_3": 1.1}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(data_dir: Path) -> list[dict]:
    utterances = sorted(read_jsonl(data_dir / "utterances.jsonl"), key=lambda row: row["utterance_id"])
    entities = {row["entity_id"]: row for row in read_jsonl(data_dir / "entities.jsonl")}
    entity_requests = sorted(
        read_jsonl(data_dir / "entity_tts_manifest.jsonl"), key=lambda row: row["entity_audio_id"]
    )

    jobs = []
    split_indexes: dict[str, int] = defaultdict(int)
    for row in utterances:
        split = row["split"]
        voices = TRAIN_VOICES if split == "train" else SPLIT_VOICES[split]
        voice = voices[split_indexes[split] % len(voices)]
        split_indexes[split] += 1
        jobs.append(
            {
                "kind": "utterance",
                "source_id": row["utterance_id"],
                "split": split,
                "domain": row["domain"],
                "speaker_role": row["speaker_role"],
                "sample_type": row["sample_type"],
                "ref_text": row["ref_text"],
                "tts_text": row["audio_text"],
                "voice": voice,
                "speed": 1.0,
                "pilot_audio_path": f"utterances/{split}/{row['utterance_id']}.wav",
            }
        )

    for row in entity_requests:
        entity = entities[row["entity_id"]]
        voice_index = int(row["voice_slot"].rsplit("_", 1)[1]) - 1
        jobs.append(
            {
                "kind": "entity",
                "source_id": row["entity_audio_id"],
                "split": "shared_entity_bank",
                "domain": "|".join(entity["domains"]),
                "speaker_role": "entity_prompt",
                "sample_type": entity["entity_type"],
                "ref_text": row["canonical_name"],
                "tts_text": row["tts_text"],
                "voice": ENTITY_VOICES[voice_index],
                "speed": SPEED_BY_SLOT[row["voice_slot"]],
                "pilot_audio_path": f"entities/{row['surface_id']}/{row['entity_audio_id']}.wav",
            }
        )
    return sorted(jobs, key=lambda row: (row["kind"], row["source_id"]))


def compatible(job: dict, existing: dict) -> bool:
    return all(existing.get(key) == job[key] for key in ("kind", "source_id", "tts_text", "voice", "speed"))


def seed_from_pilot(jobs: list[dict], pilot_dir: Path, output_dir: Path) -> int:
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    if not manifest_path.is_file():
        return 0
    pilot_rows = {row["source_id"]: row for row in read_jsonl(manifest_path)}
    reused = 0
    for job in jobs:
        row = pilot_rows.get(job["source_id"])
        if not row or row.get("status") != "generated" or not compatible(job, row):
            continue
        source = pilot_dir / row["pilot_audio_path"]
        target = output_dir / job["pilot_audio_path"]
        if not source.is_file() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        reused += 1
    return reused


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, sorted(rows, key=lambda row: (row["kind"], row["source_id"])))
    temporary.replace(path)


def summarize(results: list[dict], seeded_from_pilot: int) -> dict:
    split_counts = Counter(row["split"] for row in results if row["kind"] == "utterance")
    return {
        "request_count": len(results),
        "utterance_count": sum(row["kind"] == "utterance" for row in results),
        "entity_audio_count": sum(row["kind"] == "entity" for row in results),
        "generated_count": sum(row["status"] == "generated" for row in results),
        "failed_count": sum(row["status"] == "failed" for row in results),
        "reused_count": sum(bool(row.get("reused")) for row in results),
        "seeded_from_pilot": seeded_from_pilot,
        "total_duration_s": round(sum(row.get("duration_s") or 0 for row in results), 3),
        "utterance_split_counts": dict(sorted(split_counts.items())),
        "voices": sorted({row["voice"] for row in results}),
        "split_voice_policy": {
            "train": list(TRAIN_VOICES),
            **{key: list(value) for key, value in SPLIT_VOICES.items()},
        },
        "target_audio": {"sample_rate": 16000, "channels": 1, "subtype": "PCM_16"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete 3,600-utterance SpeechSearcher TTS corpus.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--url", default="http://211.90.240.240/tts_api/v1/audio/speech")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-pilot-reuse", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("--workers and --checkpoint-every must be positive")

    jobs = build_jobs(args.data_dir)
    if args.plan_only:
        counts = Counter((row["kind"], row["split"]) for row in jobs)
        print(json.dumps({"request_count": len(jobs), "counts": {str(k): v for k, v in counts.items()}}, indent=2))
        return

    from generate_tts_pilot import generate_one

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0 if args.no_pilot_reuse else seed_from_pilot(jobs, args.pilot_dir, args.output_dir)
    print(f"planned {len(jobs)} requests; seeded {seeded} compatible files from pilot", flush=True)

    results = []
    manifest_path = args.output_dir / "full_manifest.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_one, job, args.url, args.output_dir, args.retries) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % args.checkpoint_every == 0 or completed == len(futures):
                atomic_write_jsonl(manifest_path, results)
                generated = sum(row["status"] == "generated" for row in results)
                failed = completed - generated
                print(f"completed {completed}/{len(futures)} generated={generated} failed={failed}", flush=True)

    results.sort(key=lambda row: (row["kind"], row["source_id"]))
    write_csv(args.output_dir / "full_manifest.csv", results)
    summary = summarize(results, seeded)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
