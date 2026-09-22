"""Tests for the AudioStream aggregate root."""

import pytest

from transcript.domain.entities import AudioStream, StreamClosedError, StreamState, Utterance
from transcript.domain.value_objects import Speaker, SpeakerTurn, Timestamp, Word

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM16 mono


def _pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SAMPLE_RATE)


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, timestamp=Timestamp(start=start, end=end))


def make_stream(retention_s: float = 60.0) -> AudioStream:
    return AudioStream(stream_id="s1", sample_rate=SAMPLE_RATE, audio_retention_s=retention_s)


def test_append_audio_advances_the_timeline():
    stream = make_stream()

    assert stream.append_audio(_pcm(1.0)) == pytest.approx(1.0)
    assert stream.append_audio(_pcm(0.5)) == pytest.approx(1.5)
    assert stream.audio_end == pytest.approx(1.5)


def test_audio_window_returns_tail_and_absolute_start_time():
    stream = make_stream(retention_s=10.0)
    stream.append_audio(_pcm(8.0))

    window_pcm, start = stream.audio_window(window_s=5.0)

    assert start == pytest.approx(3.0)
    assert len(window_pcm) == int(5.0 * BYTES_PER_SECOND)


def test_audio_window_never_returns_more_than_retained_audio():
    stream = make_stream(retention_s=2.0)
    stream.append_audio(_pcm(10.0))

    window_pcm, start = stream.audio_window(window_s=30.0)

    assert start == pytest.approx(8.0)
    assert len(window_pcm) == int(2.0 * BYTES_PER_SECOND)


def test_feed_words_finalizes_utterance_on_terminal_punctuation():
    stream = make_stream()

    finalized = stream.feed_words(
        [
            _word("Olá", 0.0, 0.5),
            _word("mundo", 0.5, 1.0),
            _word("!", 1.0, 1.2),
            _word("Tudo", 1.3, 1.6),
            _word("bem", 1.6, 2.0),
            _word("?", 2.0, 2.2),
        ]
    )

    assert [u.text for u in finalized] == ["Olá mundo !", "Tudo bem ?"]
    assert finalized[0].timestamp == Timestamp(start=0.0, end=1.2)
    assert finalized[0].speaker is None  # not yet diarized
    assert stream.partial_text() is None  # nothing left open


def test_partial_text_reflects_the_open_sentence():
    stream = make_stream()

    stream.feed_words([_word("Olá", 0.0, 0.5), _word("pessoal", 0.5, 1.0)])

    assert stream.partial_text() == "Olá pessoal"


def test_partial_text_is_none_right_after_a_closed_sentence():
    stream = make_stream()

    stream.feed_words([_word("Oi.", 0.0, 0.5)])

    assert stream.partial_text() is None


def test_finalize_open_phrase_flushes_pending_words():
    stream = make_stream()

    stream.feed_words([_word("sem", 0.0, 0.3), _word("ponto", 0.3, 0.8)])
    flushed = stream.finalize_open_phrase()

    assert flushed is not None
    assert flushed.text == "sem ponto"
    assert flushed.timestamp == Timestamp(start=0.0, end=0.8)
    assert stream.finalize_open_phrase() is None  # second flush is a no-op


def test_record_diarization_assigns_and_emits_ready_utterances():
    stream = make_stream()
    stream.feed_words([_word("frase", 1.0, 2.0), _word("curta.", 2.0, 2.5)])
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=0.0, end=5.0), speaker=Speaker.numbered(0))],
        region=Timestamp(start=0.0, end=5.0),
    )

    ready = stream.drain_ready()

    assert len(ready) == 1
    assert ready[0].speaker == Speaker.numbered(0)
    assert stream.drain_ready() == []  # drained exactly once


def test_utterances_waiting_for_the_diarization_watermark_are_held_back():
    stream = make_stream()
    stream.feed_words([_word("tarde.", 8.0, 9.0)])
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=0.0, end=3.0), speaker=Speaker.numbered(0))],
        region=Timestamp(start=0.0, end=3.0),  # watermark 3s < utterance end 9s
    )

    assert stream.drain_ready() == []


def test_a_later_region_replaces_stale_turns_inside_it():
    stream = make_stream()
    stream.feed_words([_word("falou.", 3.5, 4.5)])
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=0.0, end=5.0), speaker=Speaker.numbered(0))],
        region=Timestamp(start=0.0, end=5.0),
    )
    # A fresher run re-decides [3, 8] for another speaker; the utterance sits inside it.
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=3.0, end=8.0), speaker=Speaker.numbered(1))],
        region=Timestamp(start=3.0, end=8.0),
    )

    ready = stream.drain_ready()

    assert ready[0].speaker == Speaker.numbered(1)


def test_coverage_before_a_new_region_is_preserved():
    stream = make_stream()
    stream.feed_words([_word("primeira.", 1.0, 1.5)])
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=0.0, end=5.0), speaker=Speaker.numbered(0))],
        region=Timestamp(start=0.0, end=5.0),
    )
    # New region starts at 3s; the utterance at 1.5s must keep the old attribution.
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=3.0, end=8.0), speaker=Speaker.numbered(1))],
        region=Timestamp(start=3.0, end=8.0),
    )

    ready = stream.drain_ready()

    assert ready[0].speaker == Speaker.numbered(0)


def test_drain_ready_with_force_assigns_utterances_beyond_the_watermark():
    stream = make_stream()
    stream.feed_words([_word("última.", 9.0, 9.5)])
    stream.record_diarization(
        turns=[SpeakerTurn(timestamp=Timestamp(start=0.0, end=5.0), speaker=Speaker.numbered(0))],
        region=Timestamp(start=0.0, end=5.0),
    )

    assert stream.drain_ready() == []  # still held back

    forced = stream.drain_ready(force=True)

    assert len(forced) == 1
    assert forced[0].speaker == Speaker.numbered(0)


def test_operations_after_close_raise():
    stream = make_stream()
    stream.close()

    assert stream.state is StreamState.CLOSED
    with pytest.raises(StreamClosedError):
        stream.append_audio(_pcm(1.0))
    with pytest.raises(StreamClosedError):
        stream.feed_words([_word("oi.", 0.0, 1.0)])


def test_utterances_get_monotonic_ids_per_stream():
    stream = make_stream()

    first = stream.feed_words([_word("um.", 0.0, 0.5)])
    second = stream.feed_words([_word("dois.", 1.0, 1.5)])

    assert [u.id for u in first] == [0]
    assert [u.id for u in second] == [1]
    assert isinstance(first[0], Utterance)
