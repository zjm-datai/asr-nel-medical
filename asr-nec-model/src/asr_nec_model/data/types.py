from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class EntitySpeech:
    entity: str
    input_features: torch.Tensor


@dataclass
class RetrievalExample:
    utterance_features: torch.Tensor
    entity_features: torch.Tensor
    label: int


@dataclass
class LabelExample:
    utterance_features: torch.Tensor
    candidates: List[str]
    asr_text: str
    target_error_text: str

