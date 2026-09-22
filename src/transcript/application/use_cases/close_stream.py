"""CloseStream use case: flush, finalize attribution and free the stream slot."""

from __future__ import annotations

import logging

from transcript.application.dto import ClosedStream
from transcript.application.errors import DiarizationFailed
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases._shared import finish_audio, run_diarization
from transcript.domain.ports import Diarizer, Transcriber

logger = logging.getLogger(__name__)

# Audio newer than this (seconds) always deserves one last diarization pass.
_FINAL_DIARIZATION_EPSILON_S = 0.05


class CloseStream:
    def __init__(
        self,
        registry: StreamRegistry,
        transcriber: Transcriber,
        diarizer: Diarizer,
        *,
        sample_rate: int,
        window_s: float,
        overlap_s: float,
    ) -> None:
        self._registry = registry
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._sample_rate = sample_rate
        self._window_s = window_s
        self._overlap_s = overlap_s

    async def execute(self, stream_id: str) -> ClosedStream:
        stream = self._registry.get(stream_id)
        try:
            final_words = await finish_audio(self._transcriber, stream_id)
            stream.feed_words(final_words)
            stream.finalize_open_phrase()
            await self._final_diarization(stream)
            utterances = stream.drain_ready(force=True)
        finally:
            # Whatever happened, the slot and the transcriber state must be freed.
            stream.close()
            self._registry.remove(stream_id)
            await self._transcriber.close_stream(stream_id)

        return ClosedStream(stream_id=stream_id, utterances=utterances)

    async def _final_diarization(self, stream) -> None:
        if stream.audio_end <= 0:
            return
        has_new_audio = (
            stream.audio_end > stream.last_diarization_end + _FINAL_DIARIZATION_EPSILON_S
        )
        if stream.turns and not has_new_audio:
            return  # existing turns already cover everything we have
        try:
            await run_diarization(
                stream,
                self._diarizer,
                sample_rate=self._sample_rate,
                window_s=self._window_s,
                overlap_s=self._overlap_s,
            )
        except DiarizationFailed as exc:
            # Closing must not lose the transcript: attribution degrades to
            # continuity/UNKNOWN instead of failing the close.
            logger.warning(
                "final diarization failed for stream %s (%s); emitting with degraded attribution",
                stream.id,
                exc,
            )
