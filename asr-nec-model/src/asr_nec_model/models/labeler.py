from __future__ import annotations

from typing import Sequence, Tuple

import torch

try:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Install dependencies first: uv sync"
    ) from exc

from ..utils.text import EC, EMPTY, SEP, join_candidates


class GenerativeLabeler(torch.nn.Module):
    def __init__(self, whisper_name: str = "openai/whisper-base"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(
            whisper_name,
            language="zh",
            task="transcribe",
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        for p in self.model.model.encoder.parameters():
            p.requires_grad = False

        self.tokenizer.add_tokens([EMPTY, SEP, EC], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def make_prompt(self, candidates: Sequence[str], asr_text: str) -> str:
        return f"{join_candidates(list(candidates))} {EC} {asr_text}"

    def encode_text(self, text: str, max_length: int = 448) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids[0]

    def build_decoder_sequence(
        self,
        prompt: str,
        target: str,
        max_length: int = 448,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = torch.tensor(self.model.config.decoder_start_token_id).view(1)
        prompt_ids = self.encode_text(prompt, max_length=max_length)
        target_ids = self.encode_text(target, max_length=max_length)
        eos = torch.tensor(self.tokenizer.eos_token_id).view(1)

        full = torch.cat([prompt_ids, target_ids, eos], dim=0)[:max_length]
        decoder_input_ids = torch.cat([start, full[:-1]], dim=0)
        labels = full.clone()
        labels[: len(prompt_ids)] = -100
        return decoder_input_ids, labels

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate_label(
        self,
        input_features: torch.Tensor,
        candidates: Sequence[str],
        asr_text: str,
        max_new_tokens: int = 32,
    ) -> str:
        prompt = self.make_prompt(candidates, asr_text)
        start = torch.tensor(
            [[self.model.config.decoder_start_token_id]],
            device=input_features.device,
        )
        prompt_ids = self.encode_text(prompt).unsqueeze(0).to(input_features.device)
        decoder_input_ids = torch.cat([start, prompt_ids], dim=1)

        ids = self.model.generate(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        generated = ids[0, decoder_input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=False).strip()

