"""FakeTranscriber: deterministic test double implementing the Transcriber port.

Also used by the server when ``TRANSCRIPT_ASR_PROVIDER=fake`` so the whole
pipeline can be exercised without a GPU.
"""

from __future__ import annotations

from transcript.application.errors import TranscriptionFailed
from transcript.domain.value_objects import Timestamp, Word

DEFAULT_SCRIPT = ("Olá", "mundo", ".", "Tudo", "bem", "?")


class FakeTranscriber:
    """Emits one scripted word per pushed chunk, timestamped at push time."""

    def __init__(
        self,
        script: tuple[str, ...] | list[str] = DEFAULT_SCRIPT,
        *,
        fail_open: bool = False,
        fail_push: bool = False,
        fail_finish: bool = False,
        sample_rate: int = 16_000,
    ) -> None:
        self._script = list(script)
        self._fail_open = fail_open
        self._fail_push = fail_push
        self._fail_finish = fail_finish
        self._sample_rate = sample_rate

        self._next_word_index: dict[str, int] = {}
        self._audio_end_s: dict[str, float] = {}
        self._last_word_end_s: dict[str, float] = {}

    # --- test introspection -------------------------------------------------

    @property
    def open_streams(self) -> frozenset[str]:
        return frozenset(self._next_word_index)

    def has_stream(self, stream_id: str) -> bool:
        return stream_id in self._next_word_index

    # --- Transcriber port ---------------------------------------------------

    async def open_stream(self, stream_id: str) -> None:
        if self._fail_open:
            raise TranscriptionFailed("fake transcriber refuses to open")
        if stream_id in self._next_word_index:
            raise TranscriptionFailed(f"transcriber already has stream {stream_id!r}")
        self._next_word_index[stream_id] = 0
        self._audio_end_s[stream_id] = 0.0
        self._last_word_end_s[stream_id] = 0.0

    async def push_audio(self, stream_id: str, pcm: bytes) -> list[Word]:
        if self._fail_push:
            raise TranscriptionFailed("fake transcriber refuses to transcribe")
        self._require_stream(stream_id)

        self._audio_end_s[stream_id] += len(pcm) / (self._sample_rate * 2)
        word_index = self._next_word_index[stream_id]
        self._next_word_index[stream_id] = word_index + 1

        text = self._script[word_index % len(self._script)]
        start = self._last_word_end_s[stream_id]
        end = self._audio_end_s[stream_id]
        self._last_word_end_s[stream_id] = end
        return [Word(text=text, timestamp=Timestamp(start=start, end=end))]

    async def finish_stream(self, stream_id: str) -> list[Word]:
        if self._fail_finish:
            raise TranscriptionFailed("fake transcriber refuses to finish")
        self._require_stream(stream_id)
        return []  # everything was already emitted per push

    async def close_stream(self, stream_id: str) -> None:
        self._next_word_index.pop(stream_id, None)
        self._audio_end_s.pop(stream_id, None)
        self._last_word_end_s.pop(stream_id, None)

    def _require_stream(self, stream_id: str) -> None:
        if stream_id not in self._next_word_index:
            raise TranscriptionFailed(f"unknown transcriber stream {stream_id!r}")
