"""FakeDiarizer: deterministic test double implementing the Diarizer port.

Also used by the server when ``TRANSCRIPT_DIARIZATION_PROVIDER=fake``.
Alternates SPEAKER_00 / SPEAKER_01 on every run so attribution is observable.
"""

from __future__ import annotations

from transcript.application.errors import DiarizationFailed
from transcript.domain.value_objects import Speaker, SpeakerTurn, Timestamp


class FakeDiarizer:
    def __init__(self) -> None:
        self._runs_by_stream: dict[str, int] = {}
        self.fail_with: str | None = None  # when set, every run raises DiarizationFailed

    @property
    def total_runs(self) -> int:
        return sum(self._runs_by_stream.values())

    def runs_for(self, stream_id: str) -> int:
        return self._runs_by_stream.get(stream_id, 0)

    async def diarize(
        self,
        *,
        stream_id: str,
        pcm: bytes,
        sample_rate: int,
        offset: float,
    ) -> list[SpeakerTurn]:
        if self.fail_with is not None:
            raise DiarizationFailed(self.fail_with)

        run_index = self.runs_for(stream_id)
        self._runs_by_stream[stream_id] = run_index + 1

        duration_s = len(pcm) / (sample_rate * 2)
        if duration_s <= 0:
            return []
        speaker = Speaker.numbered(run_index % 2)
        return [
            SpeakerTurn(timestamp=Timestamp(start=offset, end=offset + duration_s), speaker=speaker)
        ]
