"""Camada de domínio — regras de negócio puras da STT API.

Sem dependências de frameworks, GPU, rede ou banco. Tudo aqui é
testável sem CUDA, sem HuggingFace e sem llama.cpp.
"""

from src.domain.entities.transcript import SegmentData, TranscriptData, build_transcript
from src.domain.services.speaker_rules import (
    GENERIC_SPEAKER,
    MIC_SPEAKER,
    SYSTEM_SPEAKER,
    apply_stop_rules,
    match_raw_label,
    normalize_speaker,
    overlap,
    person_label,
    resolve_stable_labels,
)
from src.domain.value_objects.audio_format import (
    BITS_PER_SAMPLE,
    BYTES_PER_SECOND,
    CHANNELS,
    SAMPLE_RATE,
    WAV_HEADER_SIZE,
    estimate_duration_sec,
)
from src.domain.value_objects.timestamp import round2

__all__ = [
    "SegmentData",
    "TranscriptData",
    "build_transcript",
    "GENERIC_SPEAKER",
    "MIC_SPEAKER",
    "SYSTEM_SPEAKER",
    "apply_stop_rules",
    "match_raw_label",
    "normalize_speaker",
    "overlap",
    "person_label",
    "resolve_stable_labels",
    "BITS_PER_SAMPLE",
    "BYTES_PER_SECOND",
    "CHANNELS",
    "SAMPLE_RATE",
    "WAV_HEADER_SIZE",
    "estimate_duration_sec",
    "round2",
]
