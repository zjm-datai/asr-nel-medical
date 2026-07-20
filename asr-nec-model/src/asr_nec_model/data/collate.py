from __future__ import annotations

from typing import Sequence

import torch

from ..utils.tensors import pad_1d


def collate_retrieval(batch):
    utterance, entity, label = zip(*batch)
    return torch.stack(utterance), torch.stack(entity), torch.stack(label)


def collate_label(batch, pad_id: int):
    features, decoder_ids, labels = zip(*batch)
    return (
        torch.stack(features),
        pad_1d(decoder_ids, pad_id),
        pad_1d(labels, -100),
    )

