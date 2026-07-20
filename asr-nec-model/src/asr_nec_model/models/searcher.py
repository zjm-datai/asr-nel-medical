from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import WhisperForConditionalGeneration
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Install dependencies first: uv sync"
    ) from exc


class ConvAcousticProjector(nn.Module):
    """CNN projection used after the frozen Whisper encoder."""

    def __init__(self, hidden_size: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.transpose(1, 2)
        hidden = self.conv(hidden)
        hidden = F.gelu(hidden)
        hidden = hidden.transpose(1, 2)
        return self.norm(hidden)


class SpeechSearcher(nn.Module):
    def __init__(
        self,
        whisper_name: str = "openai/whisper-base",
        san_dim: int = 512,
        ffn_hidden: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        self.encoder = self.whisper.model.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        enc_dim = self.whisper.config.d_model
        self.projector = ConvAcousticProjector(enc_dim)
        self.to_san = nn.Linear(enc_dim, san_dim) if enc_dim != san_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=san_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(san_dim),
            nn.Linear(san_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, 1),
        )

    @torch.no_grad()
    def encode_frozen(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state

    def acoustic_repr(self, input_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.encode_frozen(input_features)
        return self.to_san(self.projector(hidden))

    def forward(
        self,
        utterance_features: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> torch.Tensor:
        speech = self.acoustic_repr(utterance_features)
        entity = self.acoustic_repr(entity_features)
        attended, _ = self.cross_attn(query=entity, key=speech, value=speech)
        logits = self.ffn(attended).squeeze(-1)
        logits = logits.mean(dim=1)
        return logits

    def score(
        self,
        utterance_features: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(utterance_features, entity_features))

