from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from opencc import OpenCC
from pypinyin import Style, lazy_pinyin
from asr_nec_model.data.alignment import aligned_entity_error_span


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMPTY = "<empty>"
T2S = OpenCC("t2s")
COMMON_CONFUSIONS = {
    "白": "百", "苓": "林", "草": "炒", "地": "弟", "风": "封", "肤": "夫",
    "肝": "干", "膏": "高", "黄": "皇", "胶": "交", "金": "津", "颗": "棵",
    "利": "力", "粒": "立", "林": "淋", "脉": "麦", "囊": "狼", "皮": "疲",
    "脾": "皮", "芪": "其", "清": "青", "热": "惹", "乳": "汝", "散": "伞",
    "湿": "失", "术": "树", "汤": "趟", "丸": "完", "胃": "位", "乌": "污",
    "消": "销", "虚": "需", "血": "雪", "痒": "氧", "阴": "音", "蕴": "运",
    "燥": "造", "诊": "枕", "证": "症", "疹": "诊", "脂": "知", "止": "纸",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def aligned_error_span(reference: str, hypothesis: str, start: int, end: int) -> str:
    return aligned_entity_error_span(reference, hypothesis, start, end)[0]


def stable_rng(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_homophone_index(texts: list[str]) -> dict[str, list[str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        text = T2S.convert(text)
        for character in text:
            if not "\u3400" <= character <= "\u9fff":
                continue
            syllables = lazy_pinyin(character, style=Style.NORMAL, strict=False)
            if syllables:
                counts[syllables[0]][character] += 1
    for source, replacement in COMMON_CONFUSIONS.items():
        syllables = lazy_pinyin(source, style=Style.NORMAL, strict=False)
        if syllables:
            counts[syllables[0]][replacement] += 100
    return {
        syllable: [character for character, _ in counter.most_common()]
        for syllable, counter in counts.items()
    }


def generate_error(
    text: str,
    surface_id: str,
    variant: int,
    observed_errors: dict[str, list[str]],
    homophones: dict[str, list[str]],
    seed: int,
) -> tuple[str, str]:
    observed = observed_errors.get(surface_id, [])
    if variant == 0 and observed:
        rng = stable_rng(seed, surface_id, variant, text)
        return observed[rng.randrange(len(observed))], "observed_target_asr"

    rng = stable_rng(seed, surface_id, variant, text)
    characters = list(text)
    positions = list(range(len(characters)))
    rng.shuffle(positions)
    desired_changes = 1 if variant == 0 else min(2, max(1, len(characters) // 3))
    changed = 0
    for position in positions:
        syllables = lazy_pinyin(characters[position], style=Style.NORMAL, strict=False)
        alternatives = [
            item for item in homophones.get(syllables[0], []) if item != characters[position]
        ] if syllables else []
        if alternatives:
            characters[position] = alternatives[rng.randrange(min(5, len(alternatives)))]
            changed += 1
        elif characters[position] in COMMON_CONFUSIONS:
            characters[position] = COMMON_CONFUSIONS[characters[position]]
            changed += 1
        if changed >= desired_changes:
            break
    generated = "".join(characters)
    if generated != text:
        return generated, "homophone_substitution"
    if len(text) > 1:
        position = rng.randrange(len(text))
        return text[:position] + text[position + 1 :], "character_deletion"
    return text + "儿", "character_insertion"


def observed_entity_errors(
    utterances: dict[str, dict], hypotheses: list[dict]
) -> dict[str, list[str]]:
    errors: dict[str, set[str]] = defaultdict(set)
    for hypothesis in hypotheses:
        utterance = utterances[hypothesis["utterance_id"]]
        reference = T2S.convert(utterance["ref_text"])
        asr_text = T2S.convert(hypothesis["asr_text"])
        for annotation in utterance["entities"]:
            if not annotation["correction_enabled"]:
                continue
            error = aligned_error_span(reference, asr_text, annotation["start"], annotation["end"])
            if error != EMPTY and error != annotation["text"]:
                errors[annotation["surface_id"]].add(error)
    return {surface_id: sorted(items) for surface_id, items in errors.items()}


def normalize_real_row(row: dict) -> dict:
    return {
        **row,
        "is_synthetic": False,
        "asr_source": "target_asr",
        "corruption_operator": None,
        "candidate_origin": "ss_top_k",
        "error_review_status": "observed_target_asr",
    }


def validate_rows(rows: list[dict]) -> dict:
    ids = [row["gl_example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate gl_example_id values")
    for row in rows:
        target = row["target_error_text"]
        if row["action"] == "replace":
            if target == EMPTY or target not in row["asr_text"]:
                raise ValueError(f"invalid replacement target in {row['gl_example_id']}")
            if target == row["candidate_text"]:
                raise ValueError(f"replacement target equals candidate in {row['gl_example_id']}")
        elif target != EMPTY:
            raise ValueError(f"reject row has a non-empty target in {row['gl_example_id']}")
        if row["split"] != "train" and row["is_synthetic"]:
            raise ValueError(f"synthetic example leaked into {row['split']}")
        if row["asr_text"] != T2S.convert(row["asr_text"]):
            raise ValueError(f"traditional characters remain in {row['gl_example_id']}")
    return {
        "status": "passed",
        "checked_example_count": len(rows),
        "unique_id_count": len(set(ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GL training augmentation with traceable entity errors.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--real-gl-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_full")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "data" / "gl_augmented")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--error-variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    utterance_rows = read_jsonl(args.data_dir / "utterances.jsonl")
    utterances = {row["utterance_id"]: row for row in utterance_rows}
    surfaces = {row["surface_id"]: row for row in read_jsonl(args.data_dir / "entity_surfaces.jsonl")}
    hypotheses = read_jsonl(args.data_dir / "asr_pilot" / "asr_hypotheses.jsonl")
    real_ids = {row["utterance_id"] for row in hypotheses}
    retrieval = {row["utterance_id"]: row for row in read_jsonl(args.retrieval)}
    missing_retrieval = set(utterances) - set(retrieval)
    if missing_retrieval:
        raise ValueError(f"missing SS retrieval for {len(missing_retrieval)} utterances")

    observed_errors = observed_entity_errors(utterances, hypotheses)
    homophones = build_homophone_index(
        [row["ref_text"] for row in utterance_rows] + [row["asr_text"] for row in hypotheses]
    )
    synthetic_rows = []
    sequence = 1
    oracle_backfill_count = 0
    for utterance in sorted(utterance_rows, key=lambda row: row["utterance_id"]):
        if utterance["split"] != "train" or utterance["utterance_id"] in real_ids:
            continue
        result = retrieval[utterance["utterance_id"]]
        corrections = [item for item in utterance["entities"] if item["correction_enabled"]]
        variants = [(variant, False) for variant in range(args.error_variants)] if corrections else []
        variants.append((0, True))
        for variant, clean in variants:
            reference = T2S.convert(utterance["ref_text"])
            asr_text = reference
            target_errors = {}
            operators = {}
            if not clean:
                replacements = []
                for annotation in corrections:
                    error, operator = generate_error(
                        annotation["text"], annotation["surface_id"], variant,
                        observed_errors, homophones, args.seed,
                    )
                    replacements.append((annotation["start"], annotation["end"], error))
                    target_errors[annotation["surface_id"]] = error
                    operators[annotation["surface_id"]] = operator
                for start, end, error in sorted(replacements, reverse=True):
                    asr_text = asr_text[:start] + error + asr_text[end:]

            candidates = [{**item, "candidate_origin": "ss_top_k"} for item in result["top_k"][: args.top_k]]
            candidate_ids = {item["surface_id"] for item in candidates}
            for annotation in corrections:
                surface_id = annotation["surface_id"]
                if not clean and surface_id not in candidate_ids:
                    surface = surfaces[surface_id]
                    candidates.append(
                        {
                            "surface_id": surface_id,
                            "entity_id": surface["entity_id"],
                            "surface_text": surface["surface_text"],
                            "score": None,
                            "rank": None,
                            "candidate_origin": "oracle_backfill",
                        }
                    )
                    candidate_ids.add(surface_id)
                    oracle_backfill_count += 1

            for candidate in candidates:
                target = target_errors.get(candidate["surface_id"], EMPTY)
                action = "replace" if target != EMPTY else "reject"
                synthetic_rows.append(
                    {
                        "gl_example_id": f"gl_synthetic_{sequence:07d}",
                        "utterance_id": utterance["utterance_id"],
                        "variant_id": f"error_{variant + 1}" if not clean else "clean",
                        "split": "train",
                        "domain": utterance["domain"],
                        "speaker_role": utterance["speaker_role"],
                        "audio_path": result["audio_path"],
                        "asr_text": asr_text,
                        "reference_text": reference,
                        "candidate_surface_id": candidate["surface_id"],
                        "candidate_entity_id": candidate["entity_id"],
                        "candidate_text": candidate["surface_text"],
                        "candidate_score": candidate.get("score"),
                        "candidate_rank": candidate.get("rank"),
                        "target_error_text": target,
                        "action": action,
                        "alignment_confidence": 1.0,
                        "needs_manual_review": False,
                        "candidate_source": "speech_searcher_full_best_pt",
                        "candidate_origin": candidate["candidate_origin"],
                        "is_synthetic": True,
                        "asr_source": "synthetic_entity_corruption" if not clean else "synthetic_clean",
                        "corruption_operator": operators.get(candidate["surface_id"]),
                        "error_review_status": "rule_generated_unreviewed",
                    }
                )
                sequence += 1

    real_train = [normalize_real_row(row) for row in read_jsonl(args.real_gl_dir / "gl_train.jsonl")]
    real_dev = [normalize_real_row(row) for row in read_jsonl(args.real_gl_dir / "gl_dev.jsonl")]
    real_test = [normalize_real_row(row) for row in read_jsonl(args.real_gl_dir / "gl_test.jsonl")]
    combined_train = real_train + synthetic_rows
    stable_rng(args.seed, "shuffle").shuffle(combined_train)
    validation = validate_rows(combined_train + real_dev + real_test)
    write_jsonl(args.output_dir / "gl_train_real.jsonl", real_train)
    write_jsonl(args.output_dir / "gl_train_synthetic.jsonl", synthetic_rows)
    write_jsonl(args.output_dir / "gl_train.jsonl", combined_train)
    write_jsonl(args.output_dir / "gl_dev.jsonl", real_dev)
    write_jsonl(args.output_dir / "gl_test.jsonl", real_test)
    write_jsonl(args.output_dir / "gl_all.jsonl", combined_train + real_dev + real_test)
    review_samples = []
    reviewed_surfaces = set()
    for row in synthetic_rows:
        if row["action"] == "replace" and row["candidate_surface_id"] not in reviewed_surfaces:
            review_samples.append(row)
            reviewed_surfaces.add(row["candidate_surface_id"])
    write_jsonl(args.output_dir / "synthetic_review_sample.jsonl", review_samples)

    summary = {
        "train_example_count": len(combined_train),
        "real_train_example_count": len(real_train),
        "synthetic_train_example_count": len(synthetic_rows),
        "dev_example_count": len(real_dev),
        "test_example_count": len(real_test),
        "train_action_counts": dict(Counter(row["action"] for row in combined_train)),
        "synthetic_action_counts": dict(Counter(row["action"] for row in synthetic_rows)),
        "synthetic_asr_source_counts": dict(Counter(row["asr_source"] for row in synthetic_rows)),
        "corruption_operator_counts": dict(
            Counter(row["corruption_operator"] for row in synthetic_rows if row["corruption_operator"])
        ),
        "synthetic_utterance_count": len({row["utterance_id"] for row in synthetic_rows}),
        "positive_surface_count": len(
            {row["candidate_surface_id"] for row in synthetic_rows if row["action"] == "replace"}
        ),
        "synthetic_review_sample_count": len(review_samples),
        "observed_error_surface_count": len(observed_errors),
        "oracle_backfill_count": oracle_backfill_count,
        "error_variants_per_entity_utterance": args.error_variants,
        "dev_test_source": "target_asr_only",
        "validation": validation,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
