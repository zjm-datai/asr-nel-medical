from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypinyin import Style, lazy_pinyin

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


ENTITY_NAMES = {
    "herb": """人参 党参 太子参 黄芪 白术 苍术 茯苓 炙甘草 甘草 山药 白扁豆 薏苡仁
陈皮 青皮 半夏 厚朴 砂仁 木香 香附 枳实 枳壳 佛手 香橼 大腹皮 莱菔子 鸡内金 神曲 麦芽 山楂
黄连 黄芩 黄柏 干姜 生姜 吴茱萸 高良姜 肉桂 附子 白芍 赤芍 柴胡 郁金 延胡索
丹参 蒲公英""" .split(),
    "formula": """半夏泻心汤 半夏厚朴汤 香砂六君子汤 六君子汤 保和丸 理中丸 附子理中丸
参苓白术散 四君子汤 补中益气汤 旋覆代赭汤 左金丸 黄连温胆汤 温胆汤 平胃散
藿香正气散 柴胡疏肝散 小建中汤 黄芪建中汤 良附丸 枳术丸 枳实消痞丸
越鞠丸 四逆散 痛泻要方 芍药甘草汤 甘露消毒丹 三仁汤 清中汤 一贯煎""".split(),
    "symptom": """胃脘痛 胃脘痞满 脘腹胀满 腹胀 腹痛 嗳气 反酸 烧心 嘈杂 恶心 呕吐
干呕 呃逆 纳差 食欲不振 早饱 餐后饱胀 口苦 口干 口黏 口臭 大便溏薄 泄泻
便秘 排便不畅 黏液便 完谷不化 肠鸣 矢气 胁肋胀痛 神疲乏力 畏寒肢冷
消瘦 失眠 头晕""".split(),
    "syndrome": """脾胃虚弱 脾胃虚寒 脾气虚 胃气虚 胃阴不足 肝胃不和 肝郁脾虚
寒热错杂 湿热中阻 痰湿中阻 饮食积滞 胃络瘀阻 脾阳虚 胃阳虚 中焦气滞
脾虚湿盛 肝火犯胃 寒邪客胃 气阴两虚 脾胃湿热""".split(),
    "tongue_pulse": """舌淡 舌淡胖 舌红 舌暗 舌紫暗 舌有瘀斑 苔薄白 苔白腻 苔黄腻
苔少 舌光少苔 脉弦 脉细 脉弱 脉缓 脉滑 脉数 脉沉 脉濡 脉涩""".split(),
}

ALIASES = {
    "薏苡仁": ["薏米"],
    "莱菔子": ["萝卜子"],
    "鸡内金": ["鸡肫皮"],
    "神曲": ["六神曲"],
    "延胡索": ["元胡"],
    "炙甘草": ["蜜炙甘草"],
    "半夏泻心汤": ["半夏泻心方"],
    "藿香正气散": ["藿香正气方"],
    "胃脘痛": ["胃痛"],
    "胃脘痞满": ["胃痞"],
    "脘腹胀满": ["脘腹胀"],
    "嗳气": ["打嗝"],
    "纳差": ["食纳不佳"],
    "大便溏薄": ["便溏"],
    "泄泻": ["腹泻"],
    "畏寒肢冷": ["怕冷肢凉"],
    "舌有瘀斑": ["舌有瘀点"],
    "舌光少苔": ["光剥舌"],
}

CONFUSABLE_GROUPS = [
    ["人参", "党参", "太子参"],
    ["白术", "苍术"],
    ["炙甘草", "甘草"],
    ["陈皮", "青皮"],
    ["枳实", "枳壳"],
    ["佛手", "香橼"],
    ["黄连", "黄芩", "黄柏"],
    ["干姜", "生姜", "高良姜"],
    ["白芍", "赤芍"],
    ["柴胡", "银柴胡"],
    ["半夏泻心汤", "半夏厚朴汤"],
    ["香砂六君子汤", "六君子汤"],
    ["理中丸", "附子理中丸"],
    ["四君子汤", "六君子汤"],
    ["温胆汤", "黄连温胆汤"],
    ["小建中汤", "黄芪建中汤"],
    ["枳术丸", "枳实消痞丸"],
    ["胃脘痞满", "脘腹胀满", "腹胀"],
    ["嗳气", "呃逆"],
    ["纳差", "食欲不振", "早饱"],
    ["反酸", "烧心", "嘈杂"],
    ["大便溏薄", "泄泻"],
    ["脾胃虚弱", "脾胃虚寒", "脾气虚"],
    ["胃阴不足", "胃阳虚"],
    ["肝胃不和", "肝郁脾虚", "肝火犯胃"],
    ["湿热中阻", "痰湿中阻", "脾胃湿热"],
    ["舌淡", "舌淡胖"],
    ["舌暗", "舌紫暗"],
    ["苔白腻", "苔黄腻"],
    ["脉细", "脉弱", "脉濡"],
    ["脉滑", "脉涩"],
]


def pinyin_for(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="default"))


def confusables_for(name: str, known_names: set[str]) -> list[str]:
    result: list[str] = []
    for group in CONFUSABLE_GROUPS:
        if name in group:
            result.extend(item for item in group if item != name and item in known_names)
    return sorted(set(result))


def build_entities() -> list[dict]:
    known_names = {name for names in ENTITY_NAMES.values() for name in names}
    entities = []
    for entity_type, names in ENTITY_NAMES.items():
        for index, name in enumerate(names, start=1):
            entities.append(
                {
                    "entity_id": f"{entity_type}_{index:03d}",
                    "canonical_name": name,
                    "entity_type": entity_type,
                    "aliases": ALIASES.get(name, []),
                    "pinyin": pinyin_for(name),
                    "confusables": confusables_for(name, known_names),
                    "risk_level": "high" if entity_type in {"herb", "formula"} else "medium",
                    "core": True,
                    "review_status": "pending_expert_review",
                }
            )
    return entities


def validate(entities: list[dict], expected_count: int = 150) -> None:
    if len(entities) != expected_count:
        raise ValueError(f"expected {expected_count} entities, got {len(entities)}")
    ids = [item["entity_id"] for item in entities]
    names = [item["canonical_name"] for item in entities]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("entity IDs and canonical names must be unique")
    known_names = set(names)
    for item in entities:
        if not item["pinyin"].strip():
            raise ValueError(f"missing pinyin: {item['canonical_name']}")
        unknown = set(item["confusables"]) - known_names
        if unknown:
            raise ValueError(f"unknown confusables for {item['canonical_name']}: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the core TCM spleen-stomach entity lexicon.")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "entities" / "tcm_spleen_stomach_core.jsonl",
    )
    args = parser.parse_args()
    entities = build_entities()
    validate(entities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in entities:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(entities)} entities to {args.output}")


if __name__ == "__main__":
    main()
