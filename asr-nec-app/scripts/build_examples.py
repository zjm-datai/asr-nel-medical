"""Build asr-nec-app/data/examples.json from GL evaluation predictions.

Run on the training server from the workspace root, e.g.:

    python asr-nec-app/scripts/build_examples.py \
        --predictions runs/gl_augmented_aligned_e5/predictions.jsonl \
        --output asr-nec-app/data/examples.json

The manifest references audio files relative to the mounted
``data/speech_searcher/audio_full`` directory, so no audio is copied.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DOMAIN_NAMES = {
    "atopic_dermatitis": "特应性皮炎",
    "spleen_stomach": "中医脾胃病",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/gl_augmented_aligned_e5/predictions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("asr-nec-app/data/examples.json"),
    )
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.predictions)
        if row.get("split") == "test" and row.get("action") == "replace"
    ]
    by_utterance: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("exact_match"):
            by_utterance[row["utterance_id"]].append(row)

    # Prefer utterances where every replace candidate was predicted exactly,
    # then those with the most corrected entities.
    ranked = sorted(
        by_utterance.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    examples = []
    for index, (utterance_id, items) in enumerate(ranked[: args.count], 1):
        first = items[0]
        corrected = first["asr_text"]
        notes = []
        for item in sorted(items, key=lambda row: row["candidate_rank"]):
            span = item["target_error_text"]
            if span in corrected:
                corrected = corrected.replace(span, item["candidate_text"], 1)
            notes.append(f"{span} → {item['candidate_text']}")
        examples.append(
            {
                "id": f"example_{index:02d}",
                "title": f"{DOMAIN_NAMES.get(first['domain'], first['domain'])} · {utterance_id}",
                "utterance_id": utterance_id,
                "domain": first["domain"],
                "audio_path": first["audio_path"],
                "expected_asr_text": first["asr_text"],
                "expected_corrected_text": corrected,
                "note": "；".join(notes),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
