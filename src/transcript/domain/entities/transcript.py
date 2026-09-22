"""Utterance entity: one finalized sentence with an optional assigned speaker."""

from __future__ import annotations

from dataclasses import dataclass, replace

from transcript.domain.value_objects.speaker import Speaker
from transcript.domain.value_objects.timestamp import Timestamp


@dataclass(frozen=True, slots=True)
class Utterance:
    """A finalized sentence belonging to a stream.

    Speaker assignment happens later (once diarization has covered the utterance),
    hence the mutable-through-replacement ``speaker`` field.
    """

    id: int
    timestamp: Timestamp
    text: str
    speaker: Speaker | None = None

    @property
    def is_assigned(self) -> bool:
        return self.speaker is not None

    def assign(self, speaker: Speaker) -> Utterance:
        """Return a copy of this utterance attributed to ``speaker``."""
        return replace(self, speaker=speaker)
