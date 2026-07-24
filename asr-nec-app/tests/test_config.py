from __future__ import annotations

from configs.base import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.app_name == "ASR NEC Demo API"
    assert settings.api_port == 8016
    assert settings.default_top_k == 5
    assert settings.default_threshold == 0.3
    assert settings.nec_skip_model_load is True
    assert settings.ss_checkpoint_path.name == "best.pt"
    assert settings.gl_checkpoint_path.parent.name == "gl_augmented_aligned_e5"
