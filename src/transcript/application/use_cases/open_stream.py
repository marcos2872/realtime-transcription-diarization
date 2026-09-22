"""OpenStream use case: admit a new stream or refuse it (duplicate/limit)."""

from __future__ import annotations

from transcript.application.dto import OpenedStream
from transcript.application.errors import InvalidStreamId, StreamAlreadyExists, StreamLimitReached
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases._shared import open_transcriber
from transcript.domain.entities import AudioStream
from transcript.domain.ports import Transcriber


class OpenStream:
    def __init__(
        self,
        registry: StreamRegistry,
        transcriber: Transcriber,
        *,
        max_streams: int,
        sample_rate: int,
        audio_retention_s: float,
    ) -> None:
        self._registry = registry
        self._transcriber = transcriber
        self._max_streams = max_streams
        self._sample_rate = sample_rate
        self._audio_retention_s = audio_retention_s

    async def execute(self, stream_id: str) -> OpenedStream:
        if not stream_id or not stream_id.strip():
            raise InvalidStreamId("stream id must not be empty")
        if self._registry.contains(stream_id):
            raise StreamAlreadyExists(f"stream {stream_id!r} is already open")
        if len(self._registry) >= self._max_streams:
            raise StreamLimitReached(
                f"stream limit reached ({self._max_streams}); close a stream before opening another"
            )

        # Register synchronously (atomic in the event loop) so concurrent opens
        # cannot race between the checks above and the reservation below.
        stream = AudioStream(
            stream_id=stream_id,
            sample_rate=self._sample_rate,
            audio_retention_s=self._audio_retention_s,
        )
        self._registry.add(stream)

        try:
            await open_transcriber(self._transcriber, stream_id)
        except Exception:
            self._registry.remove(stream_id)  # roll back: no ghost slot
            raise

        return OpenedStream(
            stream_id=stream_id,
            max_streams=self._max_streams,
            sample_rate=self._sample_rate,
        )
