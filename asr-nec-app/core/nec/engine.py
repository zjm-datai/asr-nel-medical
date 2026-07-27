"""NEC inference engine: ASR transcription -> SS retrieval -> GL correction.

The pipeline mirrors the training-side scripts in ``asr-nec-model``:

- entity view scoring follows ``scripts/retrieve_ss_pilot.py``
  (project cached Whisper encoder states, batch cross-attention scoring,
  mean over surface views, sigmoid probabilities);
- error-span generation follows ``scripts/evaluate_gl_pilot.py``
  (``"{candidate} <EC> {asr_text}"`` prompt, greedy argmax decoding,
  ``<empty>`` means reject).

Heavy dependencies (torch, openai-whisper, asr-nec-model) are imported
lazily inside ``load()`` so the backend test-suite can run without them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from configs.base import Settings

logger = logging.getLogger("nec.engine")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class NecEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded = False
        self.device = "cpu"
        self._lock = threading.Lock()
        self._torch: Any = None
        self._whisper: Any = None
        self.model: Any = None
        self.asr_decoder: Any = None
        self.tokenizer: Any = None
        self.text_converter: Any = None
        self.searcher: Any = None
        self.surfaces: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self._view_surface_ids: list[str] = []
        self._view_features: list[Any] = []

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import torch
        import whisper
        from asr_nec_model.models.searcher import EncodedSpeechSearcher
        from opencc import OpenCC
        from whisper.tokenizer import get_tokenizer

        self._torch = torch
        self._whisper = whisper
        self.device = self.settings.nec_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("loading whisper base from %s", self.settings.whisper_model_path)
        self.model = whisper.load_model(
            str(self.settings.whisper_model_path), device=self.device
        )
        self.model.eval()
        # GL fine-tuning replaces the decoder. Keep the original decoder for
        # transcription without loading a second Whisper encoder.
        self.asr_decoder = deepcopy(self.model.decoder).eval()

        gl_checkpoint = torch.load(
            self.settings.gl_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.decoder.load_state_dict(gl_checkpoint["decoder_state"])
        self.model.decoder.eval()
        self.tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
        self.text_converter = OpenCC("t2s")

        ss_checkpoint = torch.load(
            self.settings.ss_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.searcher = EncodedSpeechSearcher(**ss_checkpoint["model_config"]).to(
            self.device
        )
        self.searcher.load_state_dict(ss_checkpoint["model_state"])
        self.searcher.eval()

        self._load_feature_bank()
        self.loaded = True

    def _load_feature_bank(self) -> None:
        torch = self._torch
        from torch.nn.utils.rnn import pad_sequence

        data_dir = self.settings.nec_data_dir
        feature_dir = self.settings.nec_feature_dir
        self.surfaces = {
            row["surface_id"]: row
            for row in read_jsonl(data_dir / "entity_surfaces.jsonl")
        }
        self.entities = {
            row["entity_id"]: row for row in read_jsonl(data_dir / "entities.jsonl")
        }

        raw_views: list[tuple[str, Any]] = []
        for row in read_jsonl(feature_dir / "feature_manifest.jsonl"):
            if row["kind"] != "entity":
                continue
            surface_id = row["source_id"].rsplit("_v", 1)[0]
            if surface_id not in self.surfaces:
                continue
            feature = torch.load(
                feature_dir / row["feature_path"],
                map_location="cpu",
                weights_only=True,
            ).float()
            raw_views.append((surface_id, feature))
        if not raw_views:
            raise RuntimeError(f"no entity feature views found under {feature_dir}")

        amp_enabled = self.device.startswith("cuda")
        batch_size = 96
        for start in range(0, len(raw_views), batch_size):
            batch = raw_views[start : start + batch_size]
            lengths = torch.tensor(
                [feature.shape[0] for _, feature in batch], device=self.device
            )
            padded = pad_sequence(
                [feature for _, feature in batch], batch_first=True
            ).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                projected = self.searcher.project(padded)
            projected_lengths = self.searcher.projected_lengths(lengths).cpu().tolist()
            for (surface_id, _), item, keep in zip(
                batch, projected, projected_lengths, strict=True
            ):
                self._view_surface_ids.append(surface_id)
                self._view_features.append(item[:keep].detach())
        logger.info(
            "loaded %d entity views across %d surfaces",
            len(self._view_surface_ids),
            len(set(self._view_surface_ids)),
        )

    # ------------------------------------------------------------- inference
    def correct_audio(
        self,
        audio_path: Path,
        top_k: int | None = None,
        threshold: float | None = None,
        asr_text: str | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("NEC engine is not loaded")
        top_k = top_k or self.settings.default_top_k
        threshold = (
            self.settings.default_threshold if threshold is None else threshold
        )
        with self._lock:
            return self._run_pipeline(audio_path, asr_text, top_k, threshold)

    def _run_pipeline(
        self, audio_path: Path, asr_text: str | None, top_k: int, threshold: float
    ) -> dict[str, Any]:
        torch = self._torch
        whisper = self._whisper
        timings: dict[str, float] = {}

        started = time.perf_counter()
        audio = whisper.load_audio(str(audio_path))
        duration_seconds = round(len(audio) / 16000, 3)
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(torch.from_numpy(audio)),
            n_mels=self.model.dims.n_mels,
        )
        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            hidden = self.model.encoder(
                mel.unsqueeze(0).to(device=self.device, dtype=model_dtype)
            )[0]
        encoder_frames = max(1, min(1500, (len(audio) + 319) // 320))
        ss_feature = hidden[:encoder_frames].float().cpu()
        gl_feature = hidden.float().cpu()
        timings["encode_ms"] = _elapsed_ms(started)

        if asr_text is None:
            started = time.perf_counter()
            gl_decoder = self.model.decoder
            self.model.decoder = self.asr_decoder
            try:
                transcription = self.model.transcribe(audio, language="zh")
            finally:
                self.model.decoder = gl_decoder
            asr_text = str(transcription["text"]).strip()
            timings["transcribe_ms"] = _elapsed_ms(started)
        # Training data is normalized to simplified Chinese. Whisper may emit
        # traditional characters, which otherwise break GL span matching.
        asr_text = self.text_converter.convert(asr_text).strip()

        started = time.perf_counter()
        candidates = self._search(ss_feature, top_k, threshold)
        timings["search_ms"] = _elapsed_ms(started)

        started = time.perf_counter()
        from asr_nec_model.inference.corrections import apply_candidate_corrections

        generated: list[tuple[str, str]] = []
        for candidate in candidates:
            prompt = f"{candidate['surface_text']} <EC> {asr_text}"
            prediction = self._gl_generate(gl_feature, prompt)
            candidate["gl_prediction"] = prediction
            generated.append((candidate["surface_text"], prediction))
        corrected_text, correction_decisions = apply_candidate_corrections(asr_text, generated)
        for candidate, decision in zip(candidates, correction_decisions, strict=True):
            candidate["action"] = "replace" if decision.applied else "reject"
            candidate["applied"] = decision.applied
            candidate["decision_reason"] = decision.reason
        timings["label_ms"] = _elapsed_ms(started)
        timings["total_ms"] = round(sum(timings.values()), 1)

        return {
            "asr_text": asr_text,
            "corrected_text": corrected_text,
            "candidates": candidates,
            "timings": timings,
            "duration_seconds": duration_seconds,
        }

    def _search(
        self, utterance_feature: Any, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        torch = self._torch

        amp_enabled = self.device.startswith("cuda")
        lengths = torch.tensor([utterance_feature.shape[0]], device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            projected = self.searcher.project(
                utterance_feature.unsqueeze(0).to(self.device)
            )
        keep = self.searcher.projected_lengths(lengths).cpu().tolist()[0]
        utterance = projected[0][:keep].detach()

        selected_views = [
            (surface_id, {"view_id": f"legacy_{index}"}, feature)
            for index, (surface_id, feature) in enumerate(
                zip(self._view_surface_ids, self._view_features, strict=True)
            )
        ]
        return self._rerank(utterance, selected_views, top_k, threshold)

    def _rerank(
        self,
        utterance: Any,
        selected_views: list[tuple[str, dict[str, Any], Any]],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        torch = self._torch
        from torch.nn.utils.rnn import pad_sequence

        surface_scores: dict[str, list[float]] = {}
        batch_size = 96
        amp_enabled = self.device.startswith("cuda")
        for start in range(0, len(selected_views), batch_size):
            batch = selected_views[start : start + batch_size]
            batch_surface_ids = [item[0] for item in batch]
            batch_features = [item[2] for item in batch]
            entity_lengths = torch.tensor(
                [feature.shape[0] for feature in batch_features], device=self.device
            )
            entity_batch = pad_sequence(batch_features, batch_first=True).to(
                self.device
            )
            speech = utterance.unsqueeze(0).expand(len(batch_features), -1, -1)
            entity_padding = torch.arange(
                entity_batch.size(1), device=self.device
            ).unsqueeze(0) >= entity_lengths.unsqueeze(1)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                attended, _ = self.searcher.cross_attn(
                    query=entity_batch,
                    key=speech,
                    value=speech,
                    need_weights=False,
                )
                token_logits = self.searcher.ffn(attended).squeeze(-1)
                token_logits = token_logits.masked_fill(entity_padding, 0.0)
                logits = token_logits.sum(dim=1) / entity_lengths.to(token_logits.dtype)
                scores = torch.sigmoid(logits)
            for surface_id, score in zip(
                batch_surface_ids, scores.float().cpu().tolist(), strict=True
            ):
                surface_scores.setdefault(surface_id, []).append(score)

        ranked = sorted(
            (
                (surface_id, sum(scores) / len(scores))
                for surface_id, scores in surface_scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        candidates: list[dict[str, Any]] = []
        for rank, (surface_id, score) in enumerate(ranked, 1):
            if score < threshold or len(candidates) >= top_k:
                break
            surface = self.surfaces.get(surface_id, {})
            entity = self.entities.get(surface.get("entity_id", ""), {})
            candidates.append(
                {
                    "rank": rank,
                    "surface_id": surface_id,
                    "entity_id": surface.get("entity_id", ""),
                    "surface_text": surface.get("surface_text", ""),
                    "canonical_name": entity.get("canonical_name", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "risk_level": entity.get("risk_level", ""),
                    "score": round(score, 6),
                    "gl_prediction": "",
                    "action": "reject",
                    "applied": False,
                }
            )
        return candidates

    def _gl_generate(
        self, audio_feature: Any, prompt: str, max_new_tokens: int = 16
    ) -> str:
        torch = self._torch
        tokens = list(self.tokenizer.sot_sequence) + self.tokenizer.encode(prompt)
        prefix_length = len(tokens)
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                token_tensor = torch.tensor(
                    [tokens], dtype=torch.long, device=self.device
                )
                logits = self.model.decoder(
                    token_tensor, audio_feature.unsqueeze(0).to(self.device)
                )
                next_token = int(logits[0, -1].argmax().item())
                if next_token == self.tokenizer.eot:
                    break
                tokens.append(next_token)
        return self.tokenizer.decode(tokens[prefix_length:]).strip()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
