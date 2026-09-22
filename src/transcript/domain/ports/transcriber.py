"""Transcriber port: streaming speech-to-text abstraction used by the use cases."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from transcript.domain.value_objects.word import Word


@runtime_checkable
class Transcriber(Protocol):
    """Stateful streaming transcriber (one implicit session per ``stream_id``).

    Implementations must be safe to call from the event loop without blocking it.
    """

    async def open_stream(self, stream_id: str) -> None:
        """Allocate per-stream decoding state. Raises if the id is already open."""

    async def push_audio(self, stream_id: str, pcm: bytes) -> list[Word]:
        """Consume PCM16 mono audio at the negotiated sample rate.

        Returns only the words decoded *since the previous call*, with
        timestamps anchored to the stream timeline (monotonic, approximate).
        """

    async def finish_stream(self, stream_id: str) -> list[Word]:
        """Flush buffered audio (e.g. a trailing silence) and return any remaining words."""

    async def close_stream(self, stream_id: str) -> None:
        """Release per-stream state. Idempotent; must not raise for unknown ids."""
