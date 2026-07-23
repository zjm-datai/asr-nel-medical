from __future__ import annotations

import torch

from asr_nec_model.models.searcher import EncodedSpeechSearcher, lengths_to_padding_mask


def test_lengths_to_padding_mask():
    mask = lengths_to_padding_mask(torch.tensor([2, 4]), 5)
    assert mask.tolist() == [
        [False, False, True, True, True],
        [False, False, False, False, True],
    ]


def test_encoded_searcher_accepts_variable_length_cached_states():
    torch.manual_seed(7)
    model = EncodedSpeechSearcher(
        encoder_dim=16,
        san_dim=8,
        ffn_hidden=16,
        num_heads=2,
        dropout=0.0,
    ).eval()
    utterances = torch.randn(3, 21, 16)
    entities = torch.randn(3, 9, 16)
    utterance_lengths = torch.tensor([21, 15, 7])
    entity_lengths = torch.tensor([9, 6, 3])
    utterances[1, 15:] = 0
    utterances[2, 7:] = 0
    entities[1, 6:] = 0
    entities[2, 3:] = 0

    logits = model(utterances, entities, utterance_lengths, entity_lengths)
    scores = model.score(utterances, entities, utterance_lengths, entity_lengths)

    assert logits.shape == scores.shape == (3,)
    assert torch.isfinite(logits).all()
    assert ((0 <= scores) & (scores <= 1)).all()
    assert model.projected_lengths(torch.tensor([1, 2, 3, 4])).tolist() == [1, 1, 2, 2]
