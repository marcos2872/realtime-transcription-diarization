"""SpeakerTurn value object: who spoke, during which time span."""

from __future__ import annotations

from dataclasses import dataclass

from transcript.domain.value_objects.speaker import Speaker
from transcript.domain.value_objects.timestamp import Timestamp


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One diarization result: ``speaker`` was active during ``timestamp``."""

    timestamp: Timestamp
    speaker: Speaker
