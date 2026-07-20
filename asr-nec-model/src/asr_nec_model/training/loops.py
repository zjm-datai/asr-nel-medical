from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.collate import collate_label, collate_retrieval
from ..data.datasets import LabelDataset, RetrievalDataset
from ..models.labeler import GenerativeLabeler
from ..models.searcher import SpeechSearcher


def train_retriever(
    model: SpeechSearcher,
    train_set: RetrievalDataset,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 5e-5,
    device: str = "cuda",
):
    model.to(device)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_retrieval,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    model.train()
    for _ in range(epochs):
        for utterance, entity, labels in loader:
            utterance = utterance.to(device)
            entity = entity.to(device)
            labels = labels.to(device)
            logits = model(utterance, entity)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_labeler(
    labeler: GenerativeLabeler,
    train_set: LabelDataset,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
):
    labeler.to(device)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_label(b, labeler.tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [p for p in labeler.parameters() if p.requires_grad],
        lr=lr,
    )
    labeler.train()
    for _ in range(epochs):
        for features, decoder_ids, labels in loader:
            features = features.to(device)
            decoder_ids = decoder_ids.to(device)
            labels = labels.to(device)
            loss = labeler(features, decoder_ids, labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

