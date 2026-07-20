from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..models.labeler import GenerativeLabeler
from .types import LabelExample, RetrievalExample
from ..utils.text import EMPTY


class RetrievalDataset(Dataset):
    def __init__(self, examples: Sequence[RetrievalExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return ex.utterance_features, ex.entity_features, torch.tensor(ex.label).float()


class LabelDataset(Dataset):
    def __init__(self, examples: Sequence[LabelExample], labeler: GenerativeLabeler):
        self.examples = examples
        self.labeler = labeler

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = self.labeler.make_prompt(ex.candidates, ex.asr_text)
        decoder_ids, labels = self.labeler.build_decoder_sequence(
            prompt,
            ex.target_error_text or EMPTY,
        )
        return ex.utterance_features, decoder_ids, labels

