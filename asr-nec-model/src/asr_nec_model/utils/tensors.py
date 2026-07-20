from __future__ import annotations

from typing import Sequence

import torch


def pad_1d(seqs: Sequence[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(x.numel() for x in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : seq.numel()] = seq
    return out

