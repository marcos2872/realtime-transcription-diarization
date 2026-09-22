"""StreamRegistry: in-memory home of the currently open AudioStream aggregates."""

from __future__ import annotations

from transcript.application.errors import StreamNotFound
from transcript.domain.entities import AudioStream


class StreamRegistry:
    """Holds the live streams of this process (one server = one registry)."""

    def __init__(self) -> None:
        self._streams: dict[str, AudioStream] = {}

    def __len__(self) -> int:
        return len(self._streams)

    def contains(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def add(self, stream: AudioStream) -> None:
        if stream.id in self._streams:
            raise ValueError(f"stream {stream.id!r} already registered")
        self._streams[stream.id] = stream

    def get(self, stream_id: str) -> AudioStream:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise StreamNotFound(f"unknown stream {stream_id!r}") from None

    def remove(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
