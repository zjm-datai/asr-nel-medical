from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gl_pilot.py"
SPEC = spec_from_file_location("build_gl_pilot", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aligned_error_span_finds_entity_error_and_preserves_correct_entity():
    reference = "医生说证型是湿热蕴肤。"
    hypothesis = "医生说症刑是失热运浮"
    start = reference.index("湿热蕴肤")
    end = start + len("湿热蕴肤")
    span, confidence = MODULE.aligned_error_span(reference, hypothesis, start, end)
    assert span == "失热运浮"
    assert confidence > 0

    span, confidence = MODULE.aligned_error_span(reference, "医生说证型是湿热蕴肤", start, end)
    assert span == MODULE.EMPTY
    assert confidence == 1.0


def test_aligned_error_span_does_not_capture_surrounding_context():
    reference = "我再确认一下，你以前用过的是人参还是甘草？"
    hypothesis = "我再确认一下,你以前用过的是人生还是干草?"
    start = reference.index("人参")
    span, confidence = MODULE.aligned_error_span(reference, hypothesis, start, start + len("人参"))
    assert span == "人生"
    assert confidence == 1.0

    reference = "我确认一下，之前开的方剂是当归饮子还是凉血消风散？"
    hypothesis = "我却愿意一下,之前开的风机是当规影子还是梁雪萧峰散"
    start = reference.index("当归饮子")
    span, confidence = MODULE.aligned_error_span(reference, hypothesis, start, start + len("当归饮子"))
    assert span == "当规影子"
    assert confidence == 1.0
