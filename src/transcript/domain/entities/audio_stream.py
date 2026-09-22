"""AudioStream aggregate root: audio timeline, sentences and speaker turns of one stream."""

from __future__ import annotations

from enum import Enum

from transcript.domain.entities.transcript import Utterance
from transcript.domain.services.speaker_assignment import assign_speaker
from transcript.domain.value_objects.speaker_turn import SpeakerTurn
from transcript.domain.value_objects.timestamp import Timestamp
from transcript.domain.value_objects.word import Word


class StreamState(Enum):
    OPEN = "open"
    CLOSED = "closed"


class StreamClosedError(RuntimeError):
    """Raised when a stream receives data after it was closed."""


class AudioStream:
    """Aggregate root grouping everything known about one live audio stream.

    Responsibilities (single reason to change: stream-level business rules):
    - keep a bounded rolling copy of the raw audio (for diarization windows);
    - assemble words into sentence-like utterances;
    - store diarization turns and release utterances once they are attributed.
    """

    def __init__(
        self, stream_id: str, sample_rate: int = 16_000, audio_retention_s: float = 60.0
    ) -> None:
        if not stream_id:
            raise ValueError("stream_id must not be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if audio_retention_s <= 0:
            raise ValueError("audio_retention_s must be positive")

        self._stream_id = stream_id
        self._sample_rate = sample_rate
        self._bytes_per_second = sample_rate * 2  # PCM16 mono
        self._audio_retention_s = audio_retention_s

        self._state = StreamState.OPEN
        self._audio = bytearray()
        self._audio_start_s = 0.0  # absolute time of the first retained byte
        self._audio_end_s = 0.0  # absolute time of all audio ever appended

        self._phrase_words: list[Word] = []
        self._phrase_start_s: float | None = None
        self._awaiting_speaker: list[Utterance] = []
        self._next_utterance_id = 0

        self._turns: list[SpeakerTurn] = []
        self._watermark_s = 0.0  # diarization has covered [0, watermark]
        self._last_diarization_end_s = 0.0

    # --- identity / state -------------------------------------------------

    @property
    def id(self) -> str:
        return self._stream_id

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def audio_end(self) -> float:
        """Absolute seconds of audio ever appended."""
        return self._audio_end_s

    @property
    def watermark(self) -> float:
        """Diarization has trustworthy speaker info up to this point in time."""
        return self._watermark_s

    @property
    def last_diarization_end(self) -> float:
        """End of the region covered by the most recent diarization run."""
        return self._last_diarization_end_s

    @property
    def turns(self) -> tuple[SpeakerTurn, ...]:
        return tuple(self._turns)

    def close(self) -> None:
        self._state = StreamState.CLOSED

    def _raise_if_closed(self) -> None:
        if self._state is StreamState.CLOSED:
            raise StreamClosedError(f"stream {self._stream_id!r} is closed")

    # --- audio ------------------------------------------------------------

    def append_audio(self, pcm: bytes) -> float:
        """Append PCM16 audio, keeping only the last ``audio_retention_s`` seconds."""
        self._raise_if_closed()
        self._audio += pcm
        self._audio_end_s += len(pcm) / self._bytes_per_second
        # Trim old audio so long-running streams stay bounded in memory.
        max_bytes = int(self._audio_retention_s * self._bytes_per_second)
        excess = len(self._audio) - max_bytes
        if excess > 0:
            excess -= excess % 2  # keep PCM16 sample alignment
            del self._audio[:excess]
            self._audio_start_s = self._audio_end_s - len(self._audio) / self._bytes_per_second
        return self._audio_end_s

    def audio_window(self, window_s: float) -> tuple[bytes, float]:
        """Return ``(pcm, absolute_start_time)`` for the last ``window_s`` seconds."""
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        start_s = max(self._audio_start_s, self._audio_end_s - window_s)
        offset_bytes = round((start_s - self._audio_start_s) * self._bytes_per_second)
        offset_bytes -= offset_bytes % 2
        return bytes(self._audio[offset_bytes:]), start_s

    # --- sentences ---------------------------------------------------------

    def feed_words(self, words: list[Word]) -> list[Utterance]:
        """Consume new words, returning the utterances finalized by this call."""
        self._raise_if_closed()
        finalized: list[Utterance] = []
        for word in words:
            if self._phrase_start_s is None:
                self._phrase_start_s = word.timestamp.start
            self._phrase_words.append(word)
            if word.ends_sentence:
                finalized.append(self._close_phrase(end_s=word.timestamp.end))
        return finalized

    def partial_text(self) -> str | None:
        """Text of the still-open sentence, or ``None`` when nothing is pending."""
        if not self._phrase_words:
            return None
        return " ".join(word.text for word in self._phrase_words)

    def finalize_open_phrase(self) -> Utterance | None:
        """Close the trailing sentence that never got terminal punctuation (stream end)."""
        self._raise_if_closed()
        if not self._phrase_words:
            return None
        return self._close_phrase(end_s=self._phrase_words[-1].timestamp.end)

    def _close_phrase(self, end_s: float) -> Utterance:
        start_s = self._phrase_start_s if self._phrase_start_s is not None else 0.0
        utterance = Utterance(
            id=self._next_utterance_id,
            timestamp=Timestamp(start=start_s, end=max(end_s, start_s)),
            text=" ".join(word.text for word in self._phrase_words),
        )
        self._next_utterance_id += 1
        self._awaiting_speaker.append(utterance)
        self._phrase_words = []
        self._phrase_start_s = None
        return utterance

    # --- diarization -------------------------------------------------------

    def record_diarization(self, turns: list[SpeakerTurn], region: Timestamp) -> None:
        """Store a fresh diarization run covering ``region`` and advance the watermark.

        Stale turns inside ``region`` are replaced (a newer run knows better);
        the watermark only ever moves forward. Speakers are attributed lazily at
        :meth:`drain_ready` time, so utterances always see the freshest turns.
        """
        self._raise_if_closed()
        region_turns = [
            turn
            for turn in (self._clip_to_region(turn, region) for turn in turns)
            if turn is not None
        ]

        # The fresh run re-decides everything inside `region`; keep only the part
        # of older turns that lies strictly before it (they are still valid history).
        self._turns = self._retain_before(self._turns, region)
        self._turns.extend(region_turns)
        self._turns.sort(key=lambda turn: turn.timestamp.start)

        self._last_diarization_end_s = max(self._last_diarization_end_s, region.end)
        self._watermark_s = max(self._watermark_s, region.end)

    @staticmethod
    def _clip_to_region(turn: SpeakerTurn, region: Timestamp) -> SpeakerTurn | None:
        """Project a fresh turn onto ``region`` (fresh runs only speak about their region)."""
        if not turn.timestamp.overlaps(region):
            return None
        start = max(turn.timestamp.start, region.start)
        end = min(turn.timestamp.end, region.end)
        if end <= start:
            return None
        return SpeakerTurn(timestamp=Timestamp(start=start, end=end), speaker=turn.speaker)

    @staticmethod
    def _retain_before(turns: list[SpeakerTurn], region: Timestamp) -> list[SpeakerTurn]:
        """Keep old turns untouched before ``region``, clipping any head that reaches into it."""
        retained: list[SpeakerTurn] = []
        for turn in turns:
            if turn.timestamp.end <= region.start:
                retained.append(turn)
            elif turn.timestamp.start < region.start:
                clipped = Timestamp(start=turn.timestamp.start, end=region.start)
                retained.append(SpeakerTurn(timestamp=clipped, speaker=turn.speaker))
        return retained

    def drain_ready(self, force: bool = False) -> list[Utterance]:
        """Attribute and pop utterances; ``force`` includes those past the watermark.

        Assignment is deferred until this moment so utterances are attributed
        using the most recent diarization run possible.
        """
        self._raise_if_closed()
        still_waiting: list[Utterance] = []
        drained: list[Utterance] = []
        for utterance in self._awaiting_speaker:
            if force or utterance.timestamp.end <= self._watermark_s:
                drained.append(utterance.assign(assign_speaker(utterance, self._turns)))
            else:
                still_waiting.append(utterance)
        self._awaiting_speaker = still_waiting
        return drained
