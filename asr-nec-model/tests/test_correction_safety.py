from __future__ import annotations

from asr_nec_model.inference.corrections import apply_candidate_corrections


def test_candidate_edits_do_not_cascade_through_inserted_entities() -> None:
    corrected, decisions = apply_candidate_corrections(
        "你说的中药是经济吗",
        [("荆芥", "经济"), ("中焦气滞", "荆芥")],
    )

    assert corrected == "你说的中药是荆芥吗"
    assert [item.applied for item in decisions] == [True, False]
    assert decisions[1].reason == "span_missing"


def test_candidate_edits_reject_unsafe_generated_spans() -> None:
    corrected, decisions = apply_candidate_corrections(
        "毛州龙起的问题准备确认",
        [
            ("毛周隆起", "备"),
            ("四肢屈侧", "处<empty>"),
            ("继发感染", "发感染染染染染"),
        ],
    )

    assert corrected == "毛州龙起的问题准备确认"
    assert [item.reason for item in decisions] == [
        "unsafe_span_length",
        "sentinel_prediction",
        "degenerate_repetition",
    ]


def test_candidate_edits_keep_the_higher_ranked_overlapping_span() -> None:
    corrected, decisions = apply_candidate_corrections(
        "服用当归影子以后",
        [("当归饮子", "当归影子"), ("当归", "当归")],
    )

    assert corrected == "服用当归饮子以后"
    assert decisions[0].applied is True
    assert decisions[1].reason == "already_correct"
