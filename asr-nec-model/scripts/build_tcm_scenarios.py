from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


TEMPLATES = [
    "患者诉{symptom1}伴{symptom2}，辨证为{syndrome}。",
    "患者近来{symptom1}，兼有{symptom2}，舌脉见{tongue_pulse}。",
    "结合{symptom1}、{symptom2}及{tongue_pulse}，考虑{syndrome}。",
    "患者以{symptom1}为主症，伴{symptom2}，拟用{formula}加减。",
    "本次复诊{symptom1}较前减轻，仍有{symptom2}，方用{formula}。",
    "辨为{syndrome}，治以调理脾胃，选{formula}加减。",
    "患者{symptom1}反复发作，查见{tongue_pulse}，证属{syndrome}。",
    "处方考虑{formula}，核心药物包括{herb1}、{herb2}。",
]


def load_entities(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            grouped[item["entity_type"]].append(item)
    required = {"herb", "formula", "symptom", "syndrome", "tongue_pulse"}
    if missing := required - grouped.keys():
        raise ValueError(f"missing entity types: {sorted(missing)}")
    return grouped


def annotate(text: str, selected: list[dict]) -> list[dict]:
    annotations = []
    for item in selected:
        name = item["canonical_name"]
        start = text.index(name)
        annotations.append(
            {
                "entity_id": item["entity_id"],
                "text": name,
                "type": item["entity_type"],
                "start": start,
                "end": start + len(name),
            }
        )
    return sorted(annotations, key=lambda item: item["start"])


def build_scenarios(grouped: dict[str, list[dict]], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools = {entity_type: list(items) for entity_type, items in grouped.items()}
    for items in pools.values():
        rng.shuffle(items)
    cursors = defaultdict(int)

    def take(entity_type: str) -> dict:
        items = pools[entity_type]
        item = items[cursors[entity_type] % len(items)]
        cursors[entity_type] += 1
        return item

    key_types = {
        "symptom1": "symptom",
        "symptom2": "symptom",
        "syndrome": "syndrome",
        "tongue_pulse": "tongue_pulse",
        "formula": "formula",
        "herb1": "herb",
        "herb2": "herb",
    }
    scenarios = []
    for index in range(1, count + 1):
        template = TEMPLATES[(index - 1) % len(TEMPLATES)]
        used_keys = [key for key in key_types if "{" + key + "}" in template]
        values = {key: take(key_types[key]) for key in used_keys}
        text = template.format(**{key: item["canonical_name"] for key, item in values.items()})
        selected = [values[key] for key in used_keys]
        scenarios.append(
            {
                "utterance_id": f"tcm_scenario_{index:05d}",
                "scene": "tcm_spleen_stomach_outpatient",
                "ref_text": text,
                "entities": annotate(text, selected),
                "audio": None,
                "audio_source": "pending",
                "review_status": "synthetic_text_pending_expert_review",
            }
        )
    return scenarios


def validate_scenarios(scenarios: list[dict], grouped: dict[str, list[dict]]) -> None:
    seen_ids = set()
    for scenario in scenarios:
        text = scenario["ref_text"]
        for entity in scenario["entities"]:
            if text[entity["start"] : entity["end"]] != entity["text"]:
                raise ValueError(f"invalid entity offsets in {scenario['utterance_id']}")
            seen_ids.add(entity["entity_id"])
    core_ids = {item["entity_id"] for items in grouped.values() for item in items if item.get("core")}
    if len(scenarios) >= 300 and (missing := core_ids - seen_ids):
        raise ValueError(f"default-size dataset does not cover all core entities: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TCM outpatient recording/TTS scripts.")
    parser.add_argument("--entities", type=Path, default=WORKSPACE_ROOT / "data" / "entities" / "tcm_spleen_stomach_core.jsonl")
    parser.add_argument("--output", type=Path, default=WORKSPACE_ROOT / "data" / "scenarios" / "tcm_spleen_stomach_seed.jsonl")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    grouped = load_entities(args.entities)
    scenarios = build_scenarios(grouped, args.count, args.seed)
    validate_scenarios(scenarios, grouped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in scenarios:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(scenarios)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
