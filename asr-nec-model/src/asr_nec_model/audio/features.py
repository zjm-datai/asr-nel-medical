from __future__ import annotations

import torch

from transformers import WhisperProcessor


def audio_to_features(
    processor: WhisperProcessor,
    waveform,
    sampling_rate: int = 16000,
) -> torch.Tensor:
    return processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features[0]

