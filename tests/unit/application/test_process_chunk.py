"""Tests for the ProcessChunk use case."""

import pytest

from transcript.adapters.testing import FakeDiarizer, FakeTranscriber
from transcript.application.errors import (
    DiarizationFailed,
    InvalidAudioChunk,
    StreamNotFound,
    TranscriptionFailed,
)
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases import OpenStream, ProcessChunk


def build_process_chunk(settings):
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
    use_case = ProcessChunk(
        registry,
        transcriber,
        diarizer,
        sample_rate=settings.sample_rate,
        window_s=settings.diarization_window_s,
        hop_s=settings.diarization_hop_s,
        overlap_s=settings.diarization_overlap_s,
        min_s=settings.diarization_min_s,
    )
    return use_case, registry, transcriber, diarizer, opener


def pcm(seconds: float, sample_rate: int = 16_000) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


async def test_chunk_of_unknown_stream_is_rejected(settings):
    use_case, *_ = build_process_chunk(settings)

    with pytest.raises(StreamNotFound) as exc:
        await use_case.execute("ghost", pcm(1.0))

    assert exc.value.code == "stream_not_found"


@pytest.mark.parametrize("bad_pcm", [b"", b"\x00", b"abc"])
async def test_malformed_audio_is_rejected(settings, bad_pcm):
    use_case, _, _, _, opener = build_process_chunk(settings)
    await opener.execute("s1")

    with pytest.raises(InvalidAudioChunk) as exc:
        await use_case.execute("s1", bad_pcm)

    assert exc.value.code == "invalid_audio"


async def test_first_chunk_yields_partial_text(settings):
    use_case, _, _, diarizer, opener = build_process_chunk(settings)
    await opener.execute("s1")

    outcome = await use_case.execute("s1", pcm(1.0))

    assert outcome.partial_text == "Olá"  # first scripted word
    assert outcome.utterances == []
    assert outcome.audio_end == pytest.approx(1.0)


async def test_diarization_runs_on_the_hop_cadence(settings):
    use_case, _, _, diarizer, opener = build_process_chunk(settings)
    await opener.execute("s1")

    await use_case.execute("s1", pcm(1.0))  # audio_end=1.0 >= min_s, hop => runs
    assert diarizer.runs_for("s1") == 1

    await use_case.execute("s1", pcm(0.2))  # below hop => no run
    assert diarizer.runs_for("s1") == 1

    await use_case.execute("s1", pcm(settings.diarization_hop_s))
    assert diarizer.runs_for("s1") == 2


async def test_diarization_is_delayed_until_the_minimum_window(settings):
    use_case, _, _, diarizer, opener = build_process_chunk(settings)
    await opener.execute("s1")

    await use_case.execute("s1", pcm(settings.diarization_min_s - 0.5))

    assert diarizer.runs_for("s1") == 0


async def test_completed_sentence_is_emitted_with_a_speaker(settings):
    use_case, _, _, _, opener = build_process_chunk(settings)
    await opener.execute("s1")

    emitted = []
    for _ in range(3):  # script: "Olá", "mundo", "."
        outcome = await use_case.execute("s1", pcm(1.0))
        emitted.extend(outcome.utterances)

    assert len(emitted) == 1
    utterance = emitted[0]
    assert utterance.text == "Olá mundo ."
    assert utterance.speaker is not None
    assert utterance.speaker.label.startswith("SPEAKER_")
    assert utterance.timestamp.end == pytest.approx(3.0)


async def test_transcriber_failure_propagates_as_transcription_failed(settings):
    use_case, _, transcriber, _, opener = build_process_chunk(settings)
    await opener.execute("s1")
    transcriber._fail_push = True  # noqa: SLF001 - test double controlled by the test

    with pytest.raises(TranscriptionFailed) as exc:
        await use_case.execute("s1", pcm(1.0))

    assert exc.value.code == "transcription_failed"


async def test_diarizer_failure_propagates_and_leaves_the_stream_intact(settings):
    use_case, registry, _, diarizer, opener = build_process_chunk(settings)
    await opener.execute("s1")
    diarizer.fail_with = "gpu exploded"

    with pytest.raises(DiarizationFailed) as exc:
        await use_case.execute("s1", pcm(2.0))

    assert exc.value.code == "diarization_failed"
    assert len(registry) == 1
    diarizer.fail_with = None
    # The stream recovers on the next chunk.
    outcome = await use_case.execute("s1", pcm(1.0))
    assert outcome.partial_text is not None
