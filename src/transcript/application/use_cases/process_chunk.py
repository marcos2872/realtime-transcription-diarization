"""ProcessChunk use case: feed one audio chunk, get text and speakers back."""

from __future__ import annotations

from transcript.application.dto import ChunkOutcome
from transcript.application.errors import InvalidAudioChunk
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases._shared import push_audio, run_diarization
from transcript.domain.ports import Diarizer, Transcriber


class ProcessChunk:
    def __init__(
        self,
        registry: StreamRegistry,
        transcriber: Transcriber,
        diarizer: Diarizer,
        *,
        sample_rate: int,
        window_s: float,
        hop_s: float,
        overlap_s: float,
        min_s: float,
    ) -> None:
        self._registry = registry
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._sample_rate = sample_rate
        self._window_s = window_s
        self._hop_s = hop_s
        self._overlap_s = overlap_s
        self._min_s = min_s

    async def execute(self, stream_id: str, pcm: bytes) -> ChunkOutcome:
        if not pcm or len(pcm) % 2 != 0:
            raise InvalidAudioChunk("expected a non-empty, sample-aligned PCM16 chunk")
        stream = self._registry.get(stream_id)

        stream.append_audio(pcm)
        words = await push_audio(self._transcriber, stream_id, pcm)
        stream.feed_words(words)
        partial_text = stream.partial_text()

        if self._should_diarize(stream):
            # Diarization failures propagate as DiarizationFailed: the transport
            # decides whether that is fatal (WebSocket sends a non-fatal error event).
            await run_diarization(
                stream,
                self._diarizer,
                sample_rate=self._sample_rate,
                window_s=self._window_s,
                overlap_s=self._overlap_s,
            )

        return ChunkOutcome(
            utterances=stream.drain_ready(),
            partial_text=partial_text,
            audio_end=stream.audio_end,
        )

    def _should_diarize(self, stream) -> bool:
        if stream.audio_end < self._min_s:
            return False
        return (stream.audio_end - stream.last_diarization_end) >= self._hop_s
