"""Diarizer port: who-speaks-when abstraction used by the use cases."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from transcript.domain.value_objects.speaker_turn import SpeakerTurn


@runtime_checkable
class Diarizer(Protocol):
    """Identifies speakers inside an audio window."""

    async def diarize(
        self,
        *,
        stream_id: str,
        pcm: bytes,
        sample_rate: int,
        offset: float,
    ) -> list[SpeakerTurn]:
        """Analyze ``pcm`` (PCM16 mono) and return speaker turns.

        ``offset`` is the absolute stream time of the first sample; returned
        turn timestamps must be absolute (``offset`` already applied), so they
        align with word/utterance timestamps from the :class:`Transcriber`.
        """
