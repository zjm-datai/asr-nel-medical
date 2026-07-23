from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_tcm_entity_lexicon import build_entities as build_spleen_stomach_entities
from pypinyin import Style, lazy_pinyin

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

AD_ENTITY_NAMES = {
    "disease": """特应性皮炎 脂溢性皮炎 疥疮 湿疮 四弯风 婴儿湿疹 慢性湿疹 接触性皮炎""".split(),
    "symptom_sign": """瘙痒 红斑 丘疹 丘疱疹 水疱 渗出 糜烂 结痂 鳞屑 苔藓样变 皮肤干燥
皲裂 抓痕 色素沉着 色素减退 皮肤增厚 皮损潮红 夜间瘙痒 阵发性瘙痒 灼热感
疼痛 脱屑 继发感染 脓疱 皮肤粗糙 渗液 血痂 风团 皮肤肿胀 毛周隆起""".split(),
    "body_site": """面颊 额部 眼周 口周 颈部 肘窝 腘窝 手背 手腕 踝部 躯干 四肢屈侧
头皮 耳后 外阴""".split(),
    "syndrome": """湿热蕴肤 脾虚湿蕴 血虚风燥 风湿蕴肤 胎火湿热 心火脾虚
血热风盛 风燥血虚 脾虚湿盛 肝肾阴虚""".split(),
    "treatment": """清热利湿 健脾除湿 养血润燥 祛风止痒 凉血解毒 养阴润肤
疏风清热 清热凉血""".split(),
    "formula": """消风散 龙胆泻肝汤 除湿胃苓汤 当归饮子 防风通圣散 萆薢渗湿汤
四物消风饮 凉血消风散 参苓白术散 玉屏风散 黄连解毒汤 五味消毒饮""".split(),
    "herb": """苦参 白鲜皮 地肤子 防风 荆芥 蝉蜕 黄柏 苍术 生地黄 牡丹皮 赤芍 当归
何首乌 白蒺藜 徐长卿 土茯苓 蛇床子 金银花 连翘 紫草 甘草 白术 茯苓 薏苡仁 乌梢蛇""".split(),
    "western_drug": """他克莫司软膏 吡美莫司乳膏 糠酸莫米松乳膏 地奈德乳膏 卤米松乳膏
氢化可的松乳膏 克立硼罗软膏 度普利尤单抗 曲罗芦单抗 乌帕替尼 阿布昔替尼
环孢素 氯雷他定 西替利嗪 非索非那定""".split(),
    "test_scale": """血清总IgE 外周血嗜酸性粒细胞 嗜酸性粒细胞百分比 过敏原特异性IgE
皮肤点刺试验 斑贴试验 SCORAD评分 EASI评分 IGA评分 瘙痒数字评分
皮肤病生活质量指数 经皮水分丢失""".split(),
    "trigger_comorbidity": """尘螨 花粉 动物皮屑 食物过敏 牛奶过敏 鸡蛋过敏 坚果过敏
汗液刺激 羊毛刺激 季节变化 过敏性鼻炎 支气管哮喘 过敏性结膜炎 睡眠障碍
金黄色葡萄球菌定植""".split(),
}

AD_ALIASES = {
    "特应性皮炎": ["异位性皮炎", "特应性湿疹", "异位性湿疹", "AD"],
    "瘙痒数字评分": ["瘙痒NRS评分"],
    "皮肤病生活质量指数": ["DLQI"],
    "血清总IgE": ["总IgE"],
    "外周血嗜酸性粒细胞": ["外周血嗜酸粒细胞"],
    "过敏原特异性IgE": ["特异性IgE"],
    "糠酸莫米松乳膏": ["莫米松乳膏"],
    "氢化可的松乳膏": ["氢化可的松软膏"],
    "度普利尤单抗": ["度匹鲁单抗"],
    "四肢屈侧": ["四肢屈曲侧"],
    "苔藓样变": ["苔藓化"],
}

PREFERRED_TTS_TEXT = {
    "血清总IgE": "血清总 I G E",
    "过敏原特异性IgE": "过敏原特异性 I G E",
    "SCORAD评分": "S C O R A D 评分",
    "EASI评分": "E A S I 评分",
    "IGA评分": "I G A 评分",
}

ENTITY_SUBTYPES = {
    **{name: "lab_test" for name in ("血清总IgE", "外周血嗜酸性粒细胞", "嗜酸性粒细胞百分比", "过敏原特异性IgE")},
    **{name: "clinical_test" for name in ("皮肤点刺试验", "斑贴试验", "经皮水分丢失")},
    **{name: "clinician_scale" for name in ("SCORAD评分", "EASI评分", "IGA评分")},
    **{name: "patient_scale" for name in ("瘙痒数字评分", "皮肤病生活质量指数")},
    **{name: "allergen" for name in ("尘螨", "花粉", "动物皮屑")},
    **{name: "allergy" for name in ("食物过敏", "牛奶过敏", "鸡蛋过敏", "坚果过敏")},
    **{name: "exposure" for name in ("汗液刺激", "羊毛刺激", "季节变化")},
    **{name: "comorbidity" for name in ("过敏性鼻炎", "支气管哮喘", "过敏性结膜炎", "睡眠障碍")},
    "金黄色葡萄球菌定植": "colonization",
}

AD_CONFUSABLE_GROUPS = [
    ["特应性皮炎", "脂溢性皮炎", "疥疮", "慢性湿疹", "接触性皮炎"],
    ["婴儿湿疹", "慢性湿疹", "接触性皮炎"],
    ["丘疹", "丘疱疹", "水疱", "脓疱"],
    ["渗出", "渗液"],
    ["结痂", "血痂"],
    ["鳞屑", "脱屑"],
    ["瘙痒", "夜间瘙痒", "阵发性瘙痒"],
    ["色素沉着", "色素减退"],
    ["肘窝", "腘窝"],
    ["眼周", "口周"],
    ["湿热蕴肤", "风湿蕴肤", "脾虚湿蕴"],
    ["血虚风燥", "风燥血虚", "血热风盛"],
    ["清热利湿", "健脾除湿"],
    ["养血润燥", "养阴润肤"],
    ["消风散", "四物消风饮", "凉血消风散"],
    ["龙胆泻肝汤", "黄连解毒汤"],
    ["苦参", "白鲜皮", "地肤子"],
    ["他克莫司软膏", "吡美莫司乳膏"],
    ["糠酸莫米松乳膏", "地奈德乳膏", "卤米松乳膏", "氢化可的松乳膏"],
    ["度普利尤单抗", "曲罗芦单抗"],
    ["乌帕替尼", "阿布昔替尼"],
    ["氯雷他定", "西替利嗪", "非索非那定"],
    ["血清总IgE", "过敏原特异性IgE"],
    ["SCORAD评分", "EASI评分", "IGA评分"],
    ["皮肤点刺试验", "斑贴试验"],
    ["过敏性鼻炎", "支气管哮喘", "过敏性结膜炎"],
]

CORRECTION_TYPES = {
    "disease",
    "formula",
    "herb",
    "syndrome",
    "test_scale",
    "tongue_pulse",
    "western_drug",
}

# These terms are acoustically difficult and clinically useful even though their
# broad entity type is otherwise link-only.
CORRECTION_ENTITY_NAMES = set("""
胃脘痛 胃脘痞满 脘腹胀满 嗳气 嘈杂 呃逆 纳差 餐后饱胀 口黏 大便溏薄
完谷不化 肠鸣 矢气 胁肋胀痛 神疲乏力 畏寒肢冷
丘疱疹 渗出 糜烂 鳞屑 苔藓样变 皲裂 皮损潮红 夜间瘙痒 阵发性瘙痒
脱屑 继发感染 血痂 风团 毛周隆起 腘窝 四肢屈侧 外阴
清热利湿 健脾除湿 养血润燥 祛风止痒 凉血解毒 养阴润肤 疏风清热 清热凉血
金黄色葡萄球菌定植
""".split())

DIALOGUE_TEMPLATES = {
    "herb": [
        ("doctor", "你以前用过{entity}这味中药吗？"),
        ("patient", "我记得以前的方子里有{entity}。"),
        ("doctor", "我再确认一下，你对{entity}有没有不适反应？"),
        ("patient", "之前医生给我开过{entity}这味药。"),
        ("doctor", "你刚才说的那味中药是{entity}吗？"),
    ],
    "formula": [
        ("doctor", "你以前服用过{entity}吗？"),
        ("patient", "我上次吃的方子叫{entity}。"),
        ("doctor", "我确认一下，之前开的方剂是{entity}吗？"),
        ("patient", "以前医生让我服用过{entity}。"),
        ("doctor", "你对{entity}这个方名还有印象吗？"),
    ],
    "symptom": [
        ("patient", "我最近经常出现{entity}。"),
        ("doctor", "你刚才说有{entity}，大概持续多久了？"),
        ("patient", "这段时间最困扰我的就是{entity}。"),
        ("doctor", "最近{entity}比以前明显吗？"),
        ("patient", "我这次主要想看看{entity}的问题。"),
    ],
    "symptom_sign": [
        ("patient", "我最近发现皮肤有{entity}。"),
        ("doctor", "你说的{entity}大概持续多久了？"),
        ("patient", "这段时间{entity}越来越明显。"),
        ("doctor", "现在最明显的表现是{entity}吗？"),
        ("patient", "我这次主要想看看{entity}的问题。"),
    ],
    "body_site": [
        ("patient", "我的皮损主要在{entity}。"),
        ("doctor", "你说不舒服的位置是在{entity}吗？"),
        ("patient", "最早出现问题的地方是{entity}。"),
        ("doctor", "现在{entity}这个部位还明显吗？"),
        ("patient", "最近{entity}这里反复不舒服。"),
    ],
    "syndrome": [
        ("doctor", "从中医辨证来看，目前考虑{entity}。"),
        ("patient", "上次医生说我的证型是{entity}。"),
        ("doctor", "我确认一下，之前辨证写的是{entity}吗？"),
        ("doctor", "目前先按{entity}记录，还要结合其他情况。"),
        ("patient", "病历里写过{entity}，我想再问问是什么意思。"),
    ],
    "tongue_pulse": [
        ("doctor", "我看你的舌脉表现是{entity}。"),
        ("doctor", "这次舌脉检查记录为{entity}。"),
        ("doctor", "上次病历里写的是{entity}，今天我再看看。"),
        ("doctor", "目前看到的舌脉特点是{entity}。"),
        ("doctor", "你这次的舌脉表现仍然是{entity}。"),
    ],
    "disease": [
        ("doctor", "结合目前情况，首先考虑{entity}。"),
        ("patient", "以前医生诊断我有{entity}。"),
        ("doctor", "我确认一下，你以前得过{entity}吗？"),
        ("patient", "我这次是来复查{entity}的。"),
        ("doctor", "之前病历上的诊断是{entity}，对吗？"),
    ],
    "treatment": [
        ("doctor", "上次记录的中医治疗原则是{entity}。"),
        ("patient", "之前医生说治疗上要以{entity}为主。"),
        ("doctor", "目前病历里记录的治法是{entity}。"),
        ("patient", "我想再了解一下{entity}是什么意思。"),
        ("doctor", "这次先记录{entity}，后面再根据情况调整。"),
    ],
    "western_drug": [
        ("doctor", "你以前使用过{entity}吗？"),
        ("patient", "我之前用过{entity}。"),
        ("doctor", "我确认一下，你现在还在用{entity}吗？"),
        ("patient", "以前医生给我开过{entity}。"),
        ("doctor", "你说的那个药是{entity}吗？"),
    ],
    "test_scale": [
        ("doctor", "这次我们用{entity}记录一下病情。"),
        ("patient", "上次医生给我做过{entity}。"),
        ("doctor", "我确认一下，之前做的是{entity}吗？"),
        ("doctor", "今天还需要复查一下{entity}。"),
        ("patient", "我想问一下{entity}的结果。"),
    ],
    "trigger_comorbidity": [
        ("patient", "我的病史里还提到过{entity}。"),
        ("doctor", "我确认一下，你有没有{entity}方面的情况？"),
        ("patient", "以前医生也问过我有没有{entity}。"),
        ("doctor", "最近{entity}这方面有没有变化？"),
        ("patient", "我担心这次不舒服和{entity}有关系。"),
    ],
}

SECOND_PASS_TAILS = {
    "doctor": ["我再详细记录一下。", "请你慢慢回忆。", "后面还要继续观察。", "我再和你核对一下。", "还需要结合其他情况。"],
    "patient": ["最近这段时间比较明显。", "我想请您再帮我看看。", "具体时间我记不太清楚。", "和上次相比有些变化。", "我也不确定自己记得准不准。"],
}

NEUTRAL_BASES = {
    "doctor": [
        "这种情况大概持续多久了？", "最近有没有比以前更明显？", "一天里什么时候感觉最明显？",
        "以前在其他地方看过吗？", "最近有没有自己处理过？", "家里其他人有类似情况吗？",
        "这次最希望解决什么问题？", "平时作息有没有变化？", "最近工作压力大不大？",
        "还有什么情况需要补充吗？",
    ],
    "patient": [
        "大概有三个月了。", "最近比以前频繁一些。", "具体时间我记不太清楚。",
        "以前也出现过类似情况。", "我暂时没有自己处理。", "家里其他人没有这种情况。",
        "这次是第一次来就诊。", "最近工作确实比较忙。", "平时休息时间不太固定。",
        "其他方面暂时没有变化。",
    ],
}

NEUTRAL_TAILS = {
    "doctor": [
        "请按时间顺序慢慢说。", "不着急，你可以仔细想一想。", "我先把这些情况记下来。",
        "等一下我再继续问。", "你想到什么都可以补充。", "我需要再了解得具体一些。",
        "和以前相比也可以说一说。", "如果记不清可以告诉我。", "我们先从最近的变化说起。",
        "我再确认几个细节。",
    ],
    "patient": [
        "具体日期我已经记不清了。", "我回去以后可以再仔细记录。", "最近一周没有太大变化。",
        "以前没有特别留意。", "目前对平时生活影响不算大。", "我能想起来的暂时只有这些。",
        "其他情况我还要再想一想。", "家里人也帮我回忆过。", "我这几天一直在留意。",
        "如果需要我可以继续说明。",
    ],
}


def pinyin_for(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="default"))


def ad_confusables(name: str, known_names: set[str]) -> list[str]:
    values = []
    for group in AD_CONFUSABLE_GROUPS:
        if name in group:
            values.extend(item for item in group if item != name and item in known_names)
    return sorted(set(values))


def build_atopic_dermatitis_entities() -> list[dict]:
    if sum(len(names) for names in AD_ENTITY_NAMES.values()) != 150:
        raise ValueError("atopic dermatitis source lexicon must contain exactly 150 entities")
    known_names = {name for names in AD_ENTITY_NAMES.values() for name in names}
    entities = []
    for entity_type, names in AD_ENTITY_NAMES.items():
        for index, name in enumerate(names, start=1):
            entities.append(
                {
                    "entity_id": f"ad_{entity_type}_{index:03d}",
                    "canonical_name": name,
                    "entity_type": entity_type,
                    "aliases": AD_ALIASES.get(name, []),
                    "tts_text": PREFERRED_TTS_TEXT.get(name, name),
                    "semantic_subtype": ENTITY_SUBTYPES.get(name),
                    "pinyin": pinyin_for(name),
                    "confusables": ad_confusables(name, known_names),
                    "risk_level": "high" if entity_type in {"formula", "herb", "western_drug"} else "medium",
                    "core": True,
                    "domains": ["atopic_dermatitis"],
                    "review_status": "pending_expert_review",
                }
            )
    return entities


def build_unified_entities() -> tuple[list[dict], dict[str, list[str]]]:
    gi_entities = build_spleen_stomach_entities()
    for item in gi_entities:
        item["domains"] = ["spleen_stomach"]
        item["spoken_forms"] = [item["canonical_name"], *item["aliases"]]
        item["tts_text"] = item["canonical_name"]
        item["semantic_subtype"] = None
    ad_entities = build_atopic_dermatitis_entities()
    by_name = {item["canonical_name"]: item for item in gi_entities}
    domain_source_ids = {"spleen_stomach": [item["entity_id"] for item in gi_entities], "atopic_dermatitis": []}
    for item in ad_entities:
        name = item["canonical_name"]
        if name in by_name:
            existing = by_name[name]
            existing["domains"] = sorted(set(existing["domains"] + item["domains"]))
            existing["aliases"] = sorted(set(existing["aliases"] + item["aliases"]))
            existing["confusables"] = sorted(set(existing["confusables"] + item["confusables"]))
            domain_source_ids["atopic_dermatitis"].append(existing["entity_id"])
        else:
            gi_entities.append(item)
            by_name[name] = item
            domain_source_ids["atopic_dermatitis"].append(item["entity_id"])
    known_names = set(by_name)
    for item in gi_entities:
        item["confusables"] = [name for name in item["confusables"] if name in known_names]
        item["correction_enabled"] = (
            item["entity_type"] in CORRECTION_TYPES
            or item["canonical_name"] in CORRECTION_ENTITY_NAMES
        )
    return gi_entities, domain_source_ids


def build_entity_surfaces(entities: list[dict]) -> list[dict]:
    surfaces = []
    for item in entities:
        surfaces.append(
            {
                "surface_id": f"{item['entity_id']}_canonical",
                "entity_id": item["entity_id"],
                "surface_text": item["canonical_name"],
                "tts_text": item["tts_text"],
                "surface_kind": "canonical",
                "register": "professional",
                "correction_enabled": item["correction_enabled"],
                "approved_for_audio": item["correction_enabled"],
                "review_status": "pending_expert_review",
            }
        )
        for index, alias in enumerate(item["aliases"], start=1):
            surfaces.append(
                {
                    "surface_id": f"{item['entity_id']}_alias_{index:02d}",
                    "entity_id": item["entity_id"],
                    "surface_text": alias,
                    "tts_text": alias,
                    "surface_kind": "professional_alias",
                    "register": "pending_professional_review",
                    "correction_enabled": item["correction_enabled"],
                    "approved_for_audio": False,
                    "review_status": "pending_expert_review",
                }
            )
    return surfaces


def annotate(text: str, selected: list[dict]) -> list[dict]:
    annotations = []
    cursor = 0
    for item in selected:
        name = item["canonical_name"]
        start = text.find(name, cursor)
        if start < 0:
            start = text.index(name)
        annotations.append(
            {
                "entity_id": item["entity_id"],
                "surface_id": f"{item['entity_id']}_canonical",
                "text": name,
                "type": item["entity_type"],
                "correction_enabled": item["correction_enabled"],
                "start": start,
                "end": start + len(name),
                "is_focal": len(annotations) == 0,
            }
        )
        cursor = start + len(name)
    return sorted(annotations, key=lambda value: value["start"])


def split_for_occurrence(index: int) -> str:
    if index < 7:
        return "train"
    if index == 7:
        return "dev"
    return "test"


SUBTYPE_TEMPLATES = {
    "lab_test": [
        ("doctor", "这次需要查看{entity}的结果。"),
        ("patient", "我带来了{entity}的检查结果。"),
        ("doctor", "上次化验包括{entity}，对吗？"),
        ("doctor", "我确认一下，之前查过{entity}吗？"),
        ("patient", "我想问一下{entity}有没有异常。"),
    ],
    "clinical_test": [
        ("doctor", "这次需要做一下{entity}。"),
        ("patient", "上次医生给我做过{entity}。"),
        ("doctor", "我确认一下，之前做的是{entity}吗？"),
        ("doctor", "今天还需要复查{entity}。"),
        ("patient", "我想问一下{entity}的结果。"),
    ],
    "clinician_scale": [
        ("doctor", "这次我用{entity}评估一下病情。"),
        ("patient", "上次医生给我做过{entity}。"),
        ("doctor", "我确认一下，之前记录的是{entity}吗？"),
        ("doctor", "今天还需要重新评估{entity}。"),
        ("patient", "我想问一下{entity}的评分结果。"),
    ],
    "patient_scale": [
        ("doctor", "这次请你填写一下{entity}。"),
        ("patient", "上次我填写过{entity}。"),
        ("doctor", "我确认一下，之前记录的是{entity}吗？"),
        ("doctor", "今天还需要重新记录{entity}。"),
        ("patient", "我想问一下{entity}的评分结果。"),
    ],
    "allergen": [
        ("patient", "我接触{entity}以后容易不舒服。"),
        ("doctor", "你接触{entity}后症状会加重吗？"),
        ("patient", "以前医生问过我是否接触过{entity}。"),
        ("doctor", "最近接触{entity}的机会多吗？"),
        ("patient", "我担心这次发作和{entity}有关系。"),
    ],
    "allergy": [
        ("patient", "我的病史里记录过{entity}。"),
        ("doctor", "我确认一下，你有{entity}吗？"),
        ("patient", "以前医生也问过我有没有{entity}。"),
        ("doctor", "最近{entity}有加重吗？"),
        ("patient", "我担心这次不舒服和{entity}有关系。"),
    ],
    "exposure": [
        ("patient", "我感觉{entity}以后皮肤容易加重。"),
        ("doctor", "出现{entity}时症状会加重吗？"),
        ("patient", "以前医生也问过{entity}的影响。"),
        ("doctor", "最近有没有明显的{entity}？"),
        ("patient", "我担心这次发作和{entity}有关系。"),
    ],
    "comorbidity": [
        ("patient", "我的病史里还记录过{entity}。"),
        ("doctor", "我确认一下，你有没有合并{entity}？"),
        ("patient", "以前医生也问过我有没有{entity}。"),
        ("doctor", "最近{entity}控制得怎么样？"),
        ("patient", "我担心{entity}最近也有变化。"),
    ],
    "colonization": [
        ("doctor", "上次检查提示有{entity}。"),
        ("patient", "医生以前提到过{entity}。"),
        ("doctor", "我确认一下，之前记录的是{entity}吗？"),
        ("doctor", "这次还要评估有没有{entity}。"),
        ("patient", "我想问一下{entity}是什么意思。"),
    ],
}


def render_entity_utterance(entity: dict, occurrence: int, domain: str) -> tuple[str, str]:
    templates = SUBTYPE_TEMPLATES.get(entity.get("semantic_subtype"), DIALOGUE_TEMPLATES[entity["entity_type"]])
    role, template = templates[occurrence % len(templates)]
    text = template.format(entity=entity["canonical_name"])
    if occurrence >= len(templates):
        text = f"{text}{SECOND_PASS_TAILS[role][occurrence % len(SECOND_PASS_TAILS[role])]}"
    if len(entity["domains"]) > 1:
        domain_tail = {
            ("spleen_stomach", "doctor"): "我还要结合你的消化情况一起判断。",
            ("spleen_stomach", "patient"): "我这次也想看看消化方面的问题。",
            ("atopic_dermatitis", "doctor"): "我还要结合目前的皮肤表现一起判断。",
            ("atopic_dermatitis", "patient"): "我这次也想看看皮肤方面的问题。",
        }
        text = f"{text}{domain_tail[(domain, role)]}"
    return role, text


def draft_utterance(
    domain: str,
    role: str,
    text: str,
    selected: list[dict],
    focal_entity_id: str | None,
    sample_type: str,
    split: str | None = None,
) -> dict:
    audio_text = text
    for item in sorted(selected, key=lambda value: len(value["canonical_name"]), reverse=True):
        audio_text = audio_text.replace(item["canonical_name"], item["tts_text"])
    return {
        "domain": domain,
        "speaker_role": role,
        "scene": "doctor_patient_turn",
        "split": split,
        "ref_text": text,
        "audio_text": audio_text,
        "entities": annotate(text, selected),
        "focal_entity_id": focal_entity_id,
        "sample_type": sample_type,
        "review_status": "synthetic_text_pending_expert_review",
    }


def build_core_utterances(entities: list[dict], domain_source_ids: dict[str, list[str]]) -> list[dict]:
    by_id = {item["entity_id"]: item for item in entities}
    utterances = []
    for domain in ("spleen_stomach", "atopic_dermatitis"):
        for entity_id in domain_source_ids[domain]:
            entity = by_id[entity_id]
            count = 10 if entity["correction_enabled"] else 5
            for occurrence in range(count):
                role, text = render_entity_utterance(entity, occurrence, domain)
                split = split_for_occurrence(occurrence) if count == 10 else ("train" if occurrence < 3 else "dev" if occurrence == 3 else "test")
                utterances.append(
                    draft_utterance(
                        domain,
                        role,
                        text,
                        [entity],
                        entity_id,
                        "correction_core" if entity["correction_enabled"] else "link_only_core",
                        split,
                    )
                )
    return utterances


def choose_distinct(items: list[dict], index: int) -> tuple[dict, dict]:
    first = items[index % len(items)]
    second = items[(index * 7 + 3) % len(items)]
    if first["entity_id"] == second["entity_id"]:
        second = items[(items.index(second) + 1) % len(items)]
    return first, second


def build_complex_utterances(
    entities: list[dict],
    domain_source_ids: dict[str, list[str]],
    counts: dict[str, int],
) -> list[dict]:
    by_id = {item["entity_id"]: item for item in entities}
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for domain, ids in domain_source_ids.items():
        for entity_id in ids:
            item = by_id[entity_id]
            if item["correction_enabled"]:
                pools[domain][item["entity_type"]].append(item)
    utterances = []

    def ordered_pairs(items: list[dict]) -> list[tuple[dict, dict]]:
        return [(a, b) for a in items for b in items if a["entity_id"] != b["entity_id"]]

    ad_drug_groups = [
        [item for item in pools["atopic_dermatitis"]["western_drug"] if item["canonical_name"].endswith(("软膏", "乳膏"))],
        [item for item in pools["atopic_dermatitis"]["western_drug"] if item["canonical_name"] in {"度普利尤单抗", "曲罗芦单抗"}],
        [item for item in pools["atopic_dermatitis"]["western_drug"] if item["canonical_name"] in {"乌帕替尼", "阿布昔替尼", "环孢素"}],
        [item for item in pools["atopic_dermatitis"]["western_drug"] if item["canonical_name"] in {"氯雷他定", "西替利嗪", "非索非那定"}],
    ]
    ad_drug_pairs = [pair for group in ad_drug_groups for pair in ordered_pairs(group)]
    primary_ad = next(item for item in pools["atopic_dermatitis"]["disease"] if item["canonical_name"] == "特应性皮炎")
    pair_sets = {
        "spleen_stomach": {
            0: ordered_pairs(pools["spleen_stomach"]["formula"]),
            1: ordered_pairs(pools["spleen_stomach"]["herb"]),
            2: [(a, b) for a in pools["spleen_stomach"]["syndrome"] for b in pools["spleen_stomach"]["tongue_pulse"]],
            3: [(a, b) for a in pools["spleen_stomach"]["formula"] for b in pools["spleen_stomach"]["herb"]],
        },
        "atopic_dermatitis": {
            0: ad_drug_pairs,
            1: ordered_pairs(pools["atopic_dermatitis"]["formula"]),
            2: [(primary_ad, b) for b in pools["atopic_dermatitis"]["test_scale"]],
            3: [(a, b) for a in pools["atopic_dermatitis"]["syndrome"] for b in pools["atopic_dermatitis"]["herb"]],
        },
    }
    for domain, count in counts.items():
        frame_cursors = Counter()
        for index in range(count):
            frame = index % 4
            pair_pool = pair_sets[domain][frame]
            pair_index = frame_cursors[frame]
            first, second = pair_pool[pair_index % len(pair_pool)]
            repetition = pair_index // len(pair_pool)
            frame_cursors[frame] += 1
            if domain == "spleen_stomach":
                if frame == 0:
                    role = "doctor"
                    text = f"你刚才说的是{first['canonical_name']}还是{second['canonical_name']}？我需要确认具体方名。"
                elif frame == 1:
                    role = "doctor"
                    text = f"我再确认一下，你以前用过的是{first['canonical_name']}还是{second['canonical_name']}？"
                elif frame == 2:
                    role = "patient"
                    text = f"上次病历里写了{first['canonical_name']}和{second['canonical_name']}，我想确认一下。"
                else:
                    role = "patient"
                    text = f"我以前服用过{first['canonical_name']}，另外还用过{second['canonical_name']}这味药。"
            else:
                if frame == 0:
                    role = "doctor"
                    text = f"我确认一下，你以前用的是{first['canonical_name']}还是{second['canonical_name']}？"
                elif frame == 1:
                    role = "doctor"
                    text = f"我确认一下，之前开的方剂是{first['canonical_name']}还是{second['canonical_name']}？"
                elif frame == 2:
                    role = "doctor"
                    subtype = second["semantic_subtype"]
                    if subtype == "lab_test":
                        text = f"针对{first['canonical_name']}，这次还需要查看{second['canonical_name']}的结果。"
                    elif subtype == "patient_scale":
                        text = f"针对{first['canonical_name']}，这次还需要填写{second['canonical_name']}。"
                    else:
                        text = f"针对{first['canonical_name']}，这次还需要完成{second['canonical_name']}。"
                else:
                    role = "patient"
                    text = f"上次医生说我的证型是{first['canonical_name']}，还问我有没有用过{second['canonical_name']}。"
            if repetition:
                repeat_tails = {
                    "doctor": ["我把这一项也记下来。", "这项信息需要再核对。", "你可以再回忆一下。", "我继续补充到记录里。", "这次需要确认清楚。", "我再问得具体一些。"],
                    "patient": ["这两个名称我都记在病历上了。", "我想请您再核对一下。", "具体情况我记得不太清楚。", "这次想请您一起看看。", "我把以前的记录也带来了。", "我想确认自己没有记错。"],
                }
                text += repeat_tails[role][(repetition - 1) % len(repeat_tails[role])]
            utterances.append(draft_utterance(domain, role, text, [first, second], first["entity_id"], "correction_complex"))
    return utterances


def build_extra_link_utterances(
    entities: list[dict], domain_source_ids: dict[str, list[str]], counts: dict[str, int]
) -> list[dict]:
    by_id = {item["entity_id"]: item for item in entities}
    utterances = []
    for domain, count in counts.items():
        pool = [by_id[entity_id] for entity_id in domain_source_ids[domain] if not by_id[entity_id]["correction_enabled"]]
        for index in range(count):
            entity = pool[index % len(pool)]
            role, text = render_entity_utterance(entity, 5 + index // len(pool), domain)
            utterances.append(draft_utterance(domain, role, text, [entity], entity["entity_id"], "link_only_extra"))
    return utterances


def build_negative_utterances(domains: tuple[str, ...], entity_names: set[str]) -> list[dict]:
    utterances = []
    for domain in domains:
        for role in ("doctor", "patient"):
            for base in NEUTRAL_BASES[role]:
                for tail in NEUTRAL_TAILS[role]:
                    text = f"{base}{tail}"
                    text += (
                        "后面我再问一些消化方面的情况。" if domain == "spleen_stomach" and role == "doctor"
                        else "这次我主要想看看消化方面的问题。" if domain == "spleen_stomach"
                        else "后面我再问一些皮肤方面的情况。" if role == "doctor"
                        else "这次我主要想看看皮肤方面的问题。"
                    )
                    matched = [name for name in entity_names if name in text]
                    if matched:
                        raise ValueError(f"negative utterance contains target entities {matched}: {text}")
                    utterances.append(draft_utterance(domain, role, text, [], None, "no_entity"))
    return utterances


def finalize_utterances(core: list[dict], remaining: list[dict], seed: int) -> list[dict]:
    targets = {"train": 2520, "dev": 360, "test": 720}
    current = Counter(item["split"] for item in core)
    quotas = {split: targets[split] - current[split] for split in targets}
    if sum(quotas.values()) != len(remaining) or any(value < 0 for value in quotas.values()):
        raise ValueError(f"cannot satisfy split targets with quotas {quotas}")
    rng = random.Random(seed)
    rng.shuffle(remaining)
    cursor = 0
    for split in ("train", "dev", "test"):
        for item in remaining[cursor : cursor + quotas[split]]:
            item["split"] = split
        cursor += quotas[split]
    utterances = core + remaining
    utterances.sort(key=lambda item: (item["domain"], item["sample_type"], item["ref_text"]))
    for sequence, item in enumerate(utterances, start=1):
        utterance_id = f"ss_{sequence:05d}"
        item["utterance_id"] = utterance_id
        item["audio_path"] = f"audio/utterances/{item['split']}/{utterance_id}.wav"
        item["audio_source"] = "pending_tts"
        item["speaker_id"] = None
    return utterances


def build_entity_tts_manifest(surfaces: list[dict], entities: list[dict], variants: int = 3) -> list[dict]:
    by_id = {item["entity_id"]: item for item in entities}
    manifest = []
    for surface in surfaces:
        if not surface["approved_for_audio"]:
            continue
        item = by_id[surface["entity_id"]]
        for variant in range(1, variants + 1):
            manifest.append(
                {
                    "entity_audio_id": f"{surface['surface_id']}_v{variant:02d}",
                    "entity_id": item["entity_id"],
                    "surface_id": surface["surface_id"],
                    "canonical_name": item["canonical_name"],
                    "tts_text": surface["tts_text"],
                    "voice_slot": f"voice_{variant}",
                    "rate": ["normal", "slow", "fast"][variant - 1],
                    "audio_path": f"audio/entities/{surface['surface_id']}/v{variant:02d}.wav",
                    "audio_status": "pending_generation",
                }
            )
    return manifest


def candidate_kind(candidate: dict, target_items: list[dict], domain: str) -> str:
    if any(candidate["canonical_name"] in item["confusables"] for item in target_items):
        return "confusable"
    if any(candidate["entity_type"] == item["entity_type"] for item in target_items):
        return "same_type"
    if domain in candidate["domains"]:
        return "same_domain"
    return "cross_domain"


def select_negative_candidates(
    candidates: list[dict],
    target_items: list[dict],
    domain: str,
    count: int,
    rng: random.Random,
) -> list[tuple[dict, str]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate_kind(candidate, target_items, domain)].append(candidate)
    for items in buckets.values():
        rng.shuffle(items)
    quotas = {
        "confusable": count // 5,
        "same_type": count * 3 // 10,
        "same_domain": count * 3 // 10,
        "cross_domain": count - (count // 5 + count * 3 // 10 + count * 3 // 10),
    }
    selected: list[tuple[dict, str]] = []
    selected_ids = set()
    for kind, quota in quotas.items():
        for candidate in buckets[kind][:quota]:
            selected.append((candidate, kind))
            selected_ids.add(candidate["entity_id"])
    if len(selected) < count:
        remaining = [candidate for candidate in candidates if candidate["entity_id"] not in selected_ids]
        rng.shuffle(remaining)
        for candidate in remaining[: count - len(selected)]:
            selected.append((candidate, candidate_kind(candidate, target_items, domain)))
    return selected


def build_pair_manifest(utterances: list[dict], entities: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_id = {item["entity_id"]: item for item in entities}
    correction_entities = [item for item in entities if item["correction_enabled"]]
    pairs = []
    pair_sequence = 1
    for utterance in utterances:
        target_ids = {
            annotation["entity_id"]
            for annotation in utterance["entities"]
            if annotation["correction_enabled"]
        }
        target_items = [by_id[entity_id] for entity_id in target_ids]
        for entity_id in sorted(target_ids):
            pairs.append(
                {
                    "pair_id": f"pair_{pair_sequence:07d}",
                    "utterance_id": utterance["utterance_id"],
                    "entity_id": entity_id,
                    "surface_id": f"{entity_id}_canonical",
                    "label": 1,
                    "pair_type": "positive",
                    "split": utterance["split"],
                }
            )
            pair_sequence += 1
        negative_count = 10 * max(1, len(target_ids))
        candidates = [item for item in correction_entities if item["entity_id"] not in target_ids]
        selected_candidates = select_negative_candidates(
            candidates,
            target_items,
            utterance["domain"],
            negative_count,
            rng,
        )
        for candidate, kind in selected_candidates:
            pairs.append(
                {
                    "pair_id": f"pair_{pair_sequence:07d}",
                    "utterance_id": utterance["utterance_id"],
                    "entity_id": candidate["entity_id"],
                    "surface_id": f"{candidate['entity_id']}_canonical",
                    "label": 0,
                    "pair_type": "no_correction_entity" if not target_ids else kind,
                    "split": utterance["split"],
                }
            )
            pair_sequence += 1
    return pairs


def validate_all(entities: list[dict], surfaces: list[dict], utterances: list[dict], pairs: list[dict]) -> None:
    entity_ids = {item["entity_id"] for item in entities}
    if len(entity_ids) != len(entities):
        raise ValueError("duplicate entity IDs")
    if len({item["canonical_name"] for item in entities}) != len(entities):
        raise ValueError("duplicate canonical names")
    surface_ids = {item["surface_id"] for item in surfaces}
    if len(surface_ids) != len(surfaces):
        raise ValueError("duplicate surface IDs")
    utterance_ids = {item["utterance_id"] for item in utterances}
    if len(utterance_ids) != len(utterances):
        raise ValueError("duplicate utterance IDs")
    if len({item["ref_text"] for item in utterances}) != len(utterances):
        raise ValueError("duplicate utterance texts")
    for item in utterances:
        if any(token in item["ref_text"] for token in PREFERRED_TTS_TEXT) and item["audio_text"] == item["ref_text"]:
            raise ValueError(f"missing explicit TTS reading in {item['utterance_id']}")
    seen_by_domain = defaultdict(Counter)
    for utterance in utterances:
        for annotation in utterance["entities"]:
            if utterance["ref_text"][annotation["start"] : annotation["end"]] != annotation["text"]:
                raise ValueError(f"invalid offsets in {utterance['utterance_id']}")
            seen_by_domain[utterance["domain"]][annotation["entity_id"]] += 1
    for domain in ("spleen_stomach", "atopic_dermatitis"):
        domain_ids = {item["entity_id"] for item in entities if domain in item["domains"]}
        if missing := domain_ids - seen_by_domain[domain].keys():
            raise ValueError(f"uncovered {domain} entities: {sorted(missing)}")
    for pair in pairs:
        if pair["utterance_id"] not in utterance_ids or pair["entity_id"] not in entity_ids or pair["surface_id"] not in surface_ids:
            raise ValueError(f"broken pair reference: {pair['pair_id']}")
    pair_keys = {(item["utterance_id"], item["surface_id"]) for item in pairs}
    if len(pair_keys) != len(pairs):
        raise ValueError("duplicate utterance/entity pairs")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all non-audio inputs for multidomain SpeechSearcher training.")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "data" / "speech_searcher")
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    entities, domain_source_ids = build_unified_entities()
    surfaces = build_entity_surfaces(entities)
    core_utterances = build_core_utterances(entities, domain_source_ids)
    complex_utterances = build_complex_utterances(
        entities,
        domain_source_ids,
        {"spleen_stomach": 170, "atopic_dermatitis": 250},
    )
    extra_link_utterances = build_extra_link_utterances(
        entities,
        domain_source_ids,
        {"spleen_stomach": 25, "atopic_dermatitis": 60},
    )
    negative_utterances = build_negative_utterances(
        ("spleen_stomach", "atopic_dermatitis"),
        {item["canonical_name"] for item in entities},
    )
    remaining = complex_utterances + extra_link_utterances + negative_utterances
    utterances = finalize_utterances(core_utterances, remaining, args.seed)
    positive_utterances = [item for item in utterances if item["entities"]]
    tts_manifest = build_entity_tts_manifest(surfaces, entities)
    pairs = build_pair_manifest(utterances, entities, args.seed)
    validate_all(entities, surfaces, utterances, pairs)

    write_jsonl(args.output_dir / "entities.jsonl", entities)
    write_jsonl(args.output_dir / "entity_surfaces.jsonl", surfaces)
    write_jsonl(args.output_dir / "utterances.jsonl", utterances)
    write_jsonl(args.output_dir / "entity_tts_manifest.jsonl", tts_manifest)
    write_jsonl(args.output_dir / "ss_pairs.jsonl", pairs)
    write_csv(
        args.output_dir / "entity_review.csv",
        ["entity_id", "canonical_name", "entity_type", "semantic_subtype", "domains", "correction_enabled", "aliases", "confusables", "risk_level", "review_status"],
        [
            {
                "entity_id": item["entity_id"],
                "canonical_name": item["canonical_name"],
                "entity_type": item["entity_type"],
                "semantic_subtype": item["semantic_subtype"] or "",
                "domains": "|".join(item["domains"]),
                "correction_enabled": item["correction_enabled"],
                "aliases": "|".join(item["aliases"]),
                "confusables": "|".join(item["confusables"]),
                "risk_level": item["risk_level"],
                "review_status": item["review_status"],
            }
            for item in entities
        ],
    )
    write_csv(
        args.output_dir / "surface_review.csv",
        ["surface_id", "entity_id", "surface_text", "tts_text", "surface_kind", "register", "correction_enabled", "approved_for_audio", "review_status"],
        surfaces,
    )
    write_csv(
        args.output_dir / "recording_manifest.csv",
        ["utterance_id", "split", "domain", "speaker_role", "scene", "sample_type", "ref_text", "audio_text", "entity_names", "focal_entity_id", "audio_path", "speaker_id", "review_status"],
        [
            {
                "utterance_id": item["utterance_id"],
                "split": item["split"],
                "domain": item["domain"],
                "speaker_role": item["speaker_role"],
                "scene": item["scene"],
                "sample_type": item["sample_type"],
                "ref_text": item["ref_text"],
                "audio_text": item["audio_text"],
                "entity_names": "|".join(annotation["text"] for annotation in item["entities"]),
                "focal_entity_id": item["focal_entity_id"] or "",
                "audio_path": item["audio_path"],
                "speaker_id": item["speaker_id"] or "",
                "review_status": item["review_status"],
            }
            for item in utterances
        ],
    )
    write_csv(
        args.output_dir / "entity_tts_manifest.csv",
        ["entity_audio_id", "entity_id", "surface_id", "canonical_name", "tts_text", "voice_slot", "rate", "audio_path", "audio_status"],
        tts_manifest,
    )
    summary = {
        "entity_count": len(entities),
        "correction_entity_count": sum(item["correction_enabled"] for item in entities),
        "link_only_entity_count": sum(not item["correction_enabled"] for item in entities),
        "surface_count": len(surfaces),
        "approved_audio_surface_count": sum(item["approved_for_audio"] for item in surfaces),
        "domain_entity_source_count": {domain: len(ids) for domain, ids in domain_source_ids.items()},
        "utterance_count": len(utterances),
        "positive_utterance_count": len(positive_utterances),
        "negative_utterance_count": len(negative_utterances),
        "utterance_split_counts": dict(Counter(item["split"] for item in utterances)),
        "utterance_domain_counts": dict(Counter(item["domain"] for item in utterances)),
        "speaker_role_counts": dict(Counter(item["speaker_role"] for item in utterances)),
        "sample_type_counts": dict(Counter(item["sample_type"] for item in utterances)),
        "entity_tts_request_count": len(tts_manifest),
        "pair_count": len(pairs),
        "positive_pair_count": sum(item["label"] == 1 for item in pairs),
        "negative_pair_count": sum(item["label"] == 0 for item in pairs),
        "pair_split_counts": dict(Counter(item["split"] for item in pairs)),
        "pair_type_counts": dict(Counter(item["pair_type"] for item in pairs)),
        "audio_files_required": len(utterances) + len(tts_manifest),
        "review_status": "pending_expert_review",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
