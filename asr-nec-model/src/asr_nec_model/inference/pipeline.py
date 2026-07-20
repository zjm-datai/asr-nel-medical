from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..data.types import EntitySpeech
from ..models.labeler import GenerativeLabeler
from ..models.searcher import SpeechSearcher
from ..utils.text import EMPTY


@torch.no_grad()
def retrieve_candidates(
    searcher: SpeechSearcher,
    utterance_features: torch.Tensor,
    datastore: Sequence[EntitySpeech],
    threshold: float = 0.3,
    top_k: int = 5,
    batch_size: int = 64,
    device: str = "cuda",
) -> List[Tuple[str, float]]:
    searcher.eval().to(device)
    utterance_features = utterance_features.to(device)
    scored: List[Tuple[str, float]] = []

    for start in range(0, len(datastore), batch_size):
        batch = datastore[start : start + batch_size]
        entity_features = torch.stack([x.input_features for x in batch]).to(device)
        speech = utterance_features.unsqueeze(0).expand(entity_features.shape[0], -1, -1)
        probs = searcher.score(speech, entity_features).detach().cpu().tolist()
        scored.extend((x.entity, float(p)) for x, p in zip(batch, probs) if p >= threshold)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def apply_correction(
    asr_text: str,
    candidates: Sequence[str],
    generated_label: str,
) -> str:
    label = generated_label.strip()
    if not label or label == EMPTY or label.startswith(EMPTY):
        return asr_text
    if len(candidates) != 1:
        return asr_text
    return asr_text.replace(label, candidates[0], 1)


@torch.no_grad()
def correct_one_utterance(
    searcher: SpeechSearcher,
    labeler: GenerativeLabeler,
    utterance_features: torch.Tensor,
    asr_text: str,
    datastore: Sequence[EntitySpeech],
    threshold: float = 0.3,
    top_k: int = 5,
    device: str = "cuda",
) -> str:
    candidates = retrieve_candidates(
        searcher,
        utterance_features,
        datastore,
        threshold=threshold,
        top_k=top_k,
        device=device,
    )
    candidate_names = [name for name, _ in candidates]
    if not candidate_names:
        return asr_text

    labeler.eval().to(device)
    features = utterance_features.unsqueeze(0).to(device)
    generated_label = labeler.generate_label(features, candidate_names, asr_text)

    if len(candidate_names) == 1:
        return apply_correction(asr_text, candidate_names, generated_label)

    corrected = asr_text
    for entity in candidate_names:
        span = labeler.generate_label(features, [entity], corrected)
        corrected = apply_correction(corrected, [entity], span)
    return corrected

