from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import soundfile as sf


TRAIN_VOICES = ("my_zero_shot_spk", "yinyao", "yexuejie", "liqiong")
SPLIT_VOICES = {"dev": ("doctor_woman",), "test": ("haiwenyuan",)}
ENTITY_VOICES = ("yinyao", "doctor_woman", "haiwenyuan")
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def protect_level(audio):
    if not len(audio):
        return audio
    peak = float(abs(audio).max())
    rms = float((audio * audio).mean() ** 0.5)
    if rms and 20 * math.log10(rms) < -35:
        audio *= min(10 ** (-25 / 20) / rms, 0.95 / max(peak, 1e-12))
        peak = float(abs(audio).max())
    if peak > 0.95:
        audio *= 0.95 / peak
    return audio


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


def take_diverse(rows: list[dict], count: int, key, rng: random.Random) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(groups, key=str)
    while len(selected) < count and keys:
        next_keys = []
        for group_key in keys:
            if groups[group_key] and len(selected) < count:
                selected.append(groups[group_key].pop())
            if groups[group_key]:
                next_keys.append(group_key)
        keys = next_keys
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but selected {len(selected)}")
    return selected


def select_utterances(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count != 200:
        raise ValueError("the reviewed pilot design currently requires exactly 200 utterances")
    rng = random.Random(seed)
    quotas = {"train": 70, "dev": 10, "test": 20}
    selected = []
    for domain in ("spleen_stomach", "atopic_dermatitis"):
        for split, quota in quotas.items():
            pool = [row for row in rows if row["domain"] == domain and row["split"] == split]
            selected.extend(
                take_diverse(
                    pool,
                    quota,
                    lambda row: (row["speaker_role"], row["sample_type"]),
                    rng,
                )
            )
    return sorted(selected, key=lambda row: row["utterance_id"])


def select_entity_requests(
    rows: list[dict],
    entities: dict[str, dict],
    count: int,
    seed: int,
    required_surface_ids: set[str],
) -> list[dict]:
    rng = random.Random(seed)
    by_surface: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_surface[row["surface_id"]].append(row)
    surfaces = [values[0] for values in by_surface.values()]
    diverse_surfaces = take_diverse(
        surfaces,
        count,
        lambda row: (entities[row["entity_id"]]["entity_type"], tuple(entities[row["entity_id"]]["domains"])),
        rng,
    )
    selected_surface_ids = required_surface_ids | {row["surface_id"] for row in diverse_surfaces}
    missing = selected_surface_ids - by_surface.keys()
    if missing:
        raise ValueError(f"required surfaces missing from entity TTS manifest: {sorted(missing)}")
    requests = []
    for surface_id in sorted(selected_surface_ids):
        variants = sorted(by_surface[surface_id], key=lambda row: row["voice_slot"])
        requests.extend(variants)
    return requests


def synthesize(url: str, text: str, voice: str, speed: float, retries: int) -> tuple[bytes, float]:
    payload = json.dumps(
        {
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": speed,
            "sample_rate": 16000,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "audio/wav,application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                raise ValueError(f"non-WAV response: {data[:200]!r}")
            return data, time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def save_resampled(wav_bytes: bytes, output_path: Path, target_rate: int = 16000) -> dict:
    audio, source_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim != 1:
        audio = audio.mean(axis=1)
    if source_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate)
    audio = protect_level(audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, target_rate, subtype="PCM_16")
    info = sf.info(output_path)
    if info.samplerate != target_rate or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError(f"invalid normalized WAV: {info}")
    return {
        "source_sample_rate": source_rate,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "duration_s": round(info.duration, 3),
        "bytes": output_path.stat().st_size,
    }


def generate_one(job: dict, url: str, output_dir: Path, retries: int) -> dict:
    result = dict(job)
    relative_path = Path(job["pilot_audio_path"])
    output_path = output_dir / relative_path
    if output_path.exists():
        try:
            info = sf.info(output_path)
            if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16" and info.frames > 0:
                audio, _ = sf.read(output_path, dtype="float32")
                protected = protect_level(audio.copy())
                if not (protected == audio).all():
                    sf.write(output_path, protected, info.samplerate, subtype="PCM_16")
                    info = sf.info(output_path)
                result.update(
                    status="generated",
                    request_seconds=0.0,
                    source_sample_rate=24000,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    subtype=info.subtype,
                    duration_s=round(info.duration, 3),
                    bytes=output_path.stat().st_size,
                    reused=True,
                    error="",
                )
                return result
        except RuntimeError:
            pass
    try:
        wav_bytes, request_seconds = synthesize(url, job["tts_text"], job["voice"], job["speed"], retries)
        metadata = save_resampled(wav_bytes, output_path)
        result.update(metadata)
        result.update(status="generated", request_seconds=round(request_seconds, 3), reused=False, error="")
    except Exception as exc:  # Preserve every failed request in the manifest.
        result.update(
            status="failed",
            request_seconds=None,
            source_sample_rate=None,
            sample_rate=None,
            channels=None,
            subtype=None,
            duration_s=None,
            bytes=0,
            reused=False,
            error=str(exc),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and normalize the reviewed 200-utterance TTS pilot.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_pilot")
    parser.add_argument("--url", default="http://211.90.240.240/tts_api/v1/audio/speech")
    parser.add_argument("--utterances", type=int, default=200)
    parser.add_argument("--entity-surfaces", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    all_utterances = read_jsonl(args.data_dir / "utterances.jsonl")
    utterances = select_utterances(all_utterances, args.utterances, args.seed)
    utterance_by_id = {row["utterance_id"]: row for row in all_utterances}
    required_surface_ids = {
        annotation["surface_id"]
        for row in utterances
        for annotation in utterance_by_id[row["utterance_id"]]["entities"]
        if annotation["correction_enabled"]
    }
    entities = {row["entity_id"]: row for row in read_jsonl(args.data_dir / "entities.jsonl")}
    entity_requests = select_entity_requests(
        read_jsonl(args.data_dir / "entity_tts_manifest.jsonl"),
        entities,
        args.entity_surfaces,
        args.seed,
        required_surface_ids,
    )

    jobs = []
    split_indexes = defaultdict(int)
    for row in utterances:
        voices = TRAIN_VOICES if row["split"] == "train" else SPLIT_VOICES[row["split"]]
        voice = voices[split_indexes[row["split"]] % len(voices)]
        split_indexes[row["split"]] += 1
        jobs.append(
            {
                "kind": "utterance",
                "source_id": row["utterance_id"],
                "split": row["split"],
                "domain": row["domain"],
                "speaker_role": row["speaker_role"],
                "sample_type": row["sample_type"],
                "ref_text": row["ref_text"],
                "tts_text": row["audio_text"],
                "voice": voice,
                "speed": 1.0,
                "pilot_audio_path": f"utterances/{row['split']}/{row['utterance_id']}.wav",
            }
        )
    speed_by_slot = {"voice_1": 1.0, "voice_2": 0.9, "voice_3": 1.1}
    for row in entity_requests:
        variant = int(row["voice_slot"].rsplit("_", 1)[1]) - 1
        jobs.append(
            {
                "kind": "entity",
                "source_id": row["entity_audio_id"],
                "split": "shared_entity_bank",
                "domain": "|".join(entities[row["entity_id"]]["domains"]),
                "speaker_role": "entity_prompt",
                "sample_type": entities[row["entity_id"]]["entity_type"],
                "ref_text": row["canonical_name"],
                "tts_text": row["tts_text"],
                "voice": ENTITY_VOICES[variant],
                "speed": speed_by_slot[row["voice_slot"]],
                "pilot_audio_path": f"entities/{row['surface_id']}/{row['entity_audio_id']}.wav",
            }
        )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_one, job, args.url, args.output_dir, args.retries) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}")
    results.sort(key=lambda row: (row["kind"], row["source_id"]))
    write_jsonl(args.output_dir / "pilot_manifest.jsonl", results)
    write_csv(args.output_dir / "pilot_manifest.csv", results)
    summary = {
        "request_count": len(results),
        "utterance_count": sum(row["kind"] == "utterance" for row in results),
        "entity_audio_count": sum(row["kind"] == "entity" for row in results),
        "generated_count": sum(row["status"] == "generated" for row in results),
        "failed_count": sum(row["status"] == "failed" for row in results),
        "total_duration_s": round(sum(row["duration_s"] or 0 for row in results), 3),
        "voices": sorted({row["voice"] for row in results}),
        "split_voice_policy": {"train": list(TRAIN_VOICES), **{key: list(value) for key, value in SPLIT_VOICES.items()}},
        "target_audio": {"sample_rate": 16000, "channels": 1, "subtype": "PCM_16"},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
