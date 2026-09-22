"""Tests for the CloseStream use case."""

import pytest

from transcript.adapters.testing import FakeDiarizer, FakeTranscriber
from transcript.application.errors import StreamNotFound, TranscriptionFailed
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases import CloseStream, OpenStream, ProcessChunk


def build_close_stream(settings):
    registry = StreamRegistry()
    transcriber = FakeTranscriber()
    diarizer = FakeDiarizer()
    opener = OpenStream(
        registry,
        transcriber,
        max_streams=settings.max_streams,
        sample_rate=settings.sample_rate,
        audio_retention_s=settings.diarization_window_s,
    )
    processor = ProcessChunk(
        registry,
        transcriber,
        diarizer,
        sample_rate=settings.sample_rate,
        window_s=settings.diarization_window_s,
        hop_s=settings.diarization_hop_s,
        overlap_s=settings.diarization_overlap_s,
        min_s=settings.diarization_min_s,
    )
    closer = CloseStream(
        registry,
        transcriber,
        diarizer,
        sample_rate=settings.sample_rate,
        window_s=settings.diarization_window_s,
        overlap_s=settings.diarization_overlap_s,
    )
    return closer, opener, processor, registry, transcriber, diarizer


async def test_close_flushes_the_trailing_sentence_and_releases_everything(settings):
    closer, opener, processor, registry, transcriber, _ = build_close_stream(settings)
    await opener.execute("s1")
    # Script word #1 ("Olá") has no terminal punctuation yet.
    await processor.execute("s1", b"\x00\x00" * 16_000)

    closed = await closer.execute("s1")

    assert closed.stream_id == "s1"
    assert [u.text for u in closed.utterances] == ["Olá"]
    assert closed.utterances[0].speaker is not None  # final diarization ran
    assert len(registry) == 0
    assert not transcriber.has_stream("s1")


async def test_closing_an_unknown_stream_fails(settings):
    closer, *_ = build_close_stream(settings)

    with pytest.raises(StreamNotFound) as exc:
        await closer.execute("ghost")

    assert exc.value.code == "stream_not_found"


async def test_diarization_failure_at_close_does_not_lose_the_transcript(settings):
    closer, opener, processor, registry, _, diarizer = build_close_stream(settings)
    await opener.execute("s1")
    await processor.execute("s1", b"\x00\x00" * 8_000)  # 0.5s: below diarization_min_s
    diarizer.fail_with = "gpu exploded"

    closed = await closer.execute("s1")  # must not raise: closing is best-effort

    assert [u.text for u in closed.utterances] == ["Olá"]
    assert closed.utterances[0].speaker is not None
    assert closed.utterances[0].speaker.label == "UNKNOWN"  # no turns available
    assert len(registry) == 0  # slot freed even so


async def test_transcriber_failure_at_close_still_releases_the_stream(settings):
    closer, opener, processor, registry, transcriber, _ = build_close_stream(settings)
    await opener.execute("s1")
    await processor.execute("s1", b"\x00\x00" * 16_000)
    transcriber._fail_finish = True  # noqa: SLF001 - test double controlled by the test

    with pytest.raises(TranscriptionFailed):
        await closer.execute("s1")

    assert len(registry) == 0  # finally-block cleanup happened
    assert not transcriber.has_stream("s1")


async def test_closing_twice_the_second_time_reports_not_found(settings):
    closer, opener, *_ = build_close_stream(settings)
    await opener.execute("s1")
    await closer.execute("s1")

    with pytest.raises(StreamNotFound):
        await closer.execute("s1")
