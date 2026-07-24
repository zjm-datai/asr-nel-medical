from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gl_augmented.py"
SPEC = spec_from_file_location("build_gl_augmented", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_observed_error_has_priority_and_is_deterministic():
    observed = {"surface_1": ["失热运浮"]}
    first = MODULE.generate_error("湿热蕴肤", "surface_1", 0, observed, {}, 7)
    second = MODULE.generate_error("湿热蕴肤", "surface_1", 0, observed, {}, 7)
    assert first == ("失热运浮", "observed_target_asr")
    assert second == first


def test_generated_error_differs_from_entity_text():
    error, operator = MODULE.generate_error("湿热蕴肤", "surface_1", 1, {}, {}, 7)
    assert error != "湿热蕴肤"
    assert operator in {"homophone_substitution", "character_deletion", "character_insertion"}


def test_aligned_error_span_extracts_real_asr_entity_error():
    reference = "医生说证型是湿热蕴肤"
    hypothesis = "医生说症型是失热运浮"
    start = reference.index("湿热蕴肤")
    end = start + len("湿热蕴肤")
    assert MODULE.aligned_error_span(reference, hypothesis, start, end) == "失热运浮"
