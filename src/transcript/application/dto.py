"""DTOs: what the use cases hand back to the transport layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from transcript.domain.entities import Utterance


@dataclass(frozen=True, slots=True)
class OpenedStream:
    stream_id: str
    max_streams: int
    sample_rate: int


@dataclass(frozen=True, slots=True)
class ChunkOutcome:
    """Result of feeding one audio chunk."""

    utterances: list[Utterance] = field(default_factory=list)
    partial_text: str | None = None
    audio_end: float = 0.0


@dataclass(frozen=True, slots=True)
class ClosedStream:
    stream_id: str
    utterances: list[Utterance]
