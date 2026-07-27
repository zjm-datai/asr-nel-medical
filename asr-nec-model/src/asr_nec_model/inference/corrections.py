from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..utils.text import EMPTY


@dataclass(frozen=True)
class CorrectionDecision:
    applied: bool
    reason: str
    start: int | None = None
    end: int | None = None


def apply_candidate_corrections(
    asr_text: str,
    candidates: Sequence[tuple[str, str]],
) -> tuple[str, list[CorrectionDecision]]:
    """Apply ranked GL spans once against the original ASR text."""
    occupied: list[tuple[int, int]] = []
    edits: list[tuple[int, int, str]] = []
    decisions: list[CorrectionDecision] = []
    for candidate_text, generated_span in candidates:
        span = generated_span.strip()
        reason = _unsafe_span_reason(asr_text, candidate_text, span)
        if reason is not None:
            decisions.append(CorrectionDecision(False, reason))
            continue
        start = asr_text.find(span)
        end = start + len(span)
        if any(start < occupied_end and occupied_start < end for occupied_start, occupied_end in occupied):
            decisions.append(CorrectionDecision(False, "overlapping_higher_rank_edit", start, end))
            continue
        occupied.append((start, end))
        edits.append((start, end, candidate_text))
        decisions.append(CorrectionDecision(True, "applied", start, end))

    corrected = asr_text
    for start, end, replacement in sorted(edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    return corrected, decisions


def _unsafe_span_reason(asr_text: str, candidate_text: str, span: str) -> str | None:
    if not span:
        return "empty_prediction"
    if EMPTY in span:
        return "sentinel_prediction"
    if span == candidate_text:
        return "already_correct"
    if re.search(r"(.)\1{3,}", span):
        return "degenerate_repetition"
    candidate_length = _content_length(candidate_text)
    span_length = _content_length(span)
    minimum_length = 1 if candidate_length <= 1 else max(2, candidate_length // 2)
    maximum_length = max(4, candidate_length * 2 + 2)
    if span_length < minimum_length or span_length > maximum_length:
        return "unsafe_span_length"
    occurrences = asr_text.count(span)
    if occurrences == 0:
        return "span_missing"
    if occurrences > 1:
        return "ambiguous_span"
    return None


def _content_length(text: str) -> int:
    return sum(character.isalnum() for character in text)
