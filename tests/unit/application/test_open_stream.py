"""Tests for the OpenStream use case."""

import pytest

from transcript.adapters.testing import FakeTranscriber
from transcript.application.dto import OpenedStream
from transcript.application.errors import (
    StreamAlreadyExists,
    StreamLimitReached,
    TranscriptionFailed,
)
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases import OpenStream


def build_open_stream(
    settings, registry: StreamRegistry | None = None
) -> tuple[OpenStream, StreamRegistry, FakeTranscriber]:
    registry = registry or StreamRegistry()
    transcriber = FakeTranscriber()
    use_case = OpenStream(
        registry,
        transcriber,
        max_streams=settings.max_streams,
        sample_rate=settings.sample_rate,
        audio_retention_s=settings.diarization_window_s,
    )
    return use_case, registry, transcriber


async def test_open_registers_stream_and_allocates_transcriber_state(settings):
    use_case, registry, transcriber = build_open_stream(settings)

    opened = await use_case.execute("s1")

    assert opened == OpenedStream(
        stream_id="s1",
        max_streams=settings.max_streams,
        sample_rate=settings.sample_rate,
    )
    assert len(registry) == 1
    assert transcriber.has_stream("s1")


async def test_opening_the_same_stream_twice_is_rejected(settings):
    use_case, _, _ = build_open_stream(settings)
    await use_case.execute("s1")

    with pytest.raises(StreamAlreadyExists) as exc:
        await use_case.execute("s1")

    assert exc.value.code == "stream_exists"


async def test_stream_limit_is_enforced(settings):
    use_case, registry, _ = build_open_stream(settings)

    for index in range(settings.max_streams):
        await use_case.execute(f"s{index}")

    with pytest.raises(StreamLimitReached) as exc:
        await use_case.execute("one-too-many")

    assert exc.value.code == "stream_limit"
    assert len(registry) == settings.max_streams


async def test_a_finished_stream_frees_its_slot(settings):
    use_case, registry, _ = build_open_stream(settings)
    for index in range(settings.max_streams):
        await use_case.execute(f"s{index}")
    registry.remove("s0")

    await use_case.execute("another")

    assert len(registry) == settings.max_streams


async def test_transcriber_failure_does_not_leave_a_ghost_stream(settings):
    registry = StreamRegistry()
    transcriber = FakeTranscriber(fail_open=True)
    use_case = OpenStream(
        registry,
        transcriber,
        max_streams=settings.max_streams,
        sample_rate=settings.sample_rate,
        audio_retention_s=settings.diarization_window_s,
    )

    with pytest.raises(TranscriptionFailed) as exc:
        await use_case.execute("s1")

    assert exc.value.code == "transcription_failed"
    assert len(registry) == 0
