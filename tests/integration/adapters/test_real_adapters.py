"""Smoke tests for the real adapters: Transformers streaming + pyannote community-1.

These need the heavy extras, a CUDA GPU and model downloads, so they only run
with ``RUN_GPU_TESTS=1``. Without the extras they skip gracefully.
"""

import os

import pytest

pytestmark = pytest.mark.gpu

_RUNNABLE = os.environ.get("RUN_GPU_TESTS") == "1"
requires_gpu_env = pytest.mark.skipif(
    not _RUNNABLE, reason="set RUN_GPU_TESTS=1 to run real-model tests"
)

SAMPLE_RATE = 16_000
ONE_SILENCE = b"\x00\x00" * SAMPLE_RATE  # 1 s of PCM16 silence
THREE_SILENCE = ONE_SILENCE * 3


@requires_gpu_env
async def test_nemotron_loads_transcribes_and_tracks_audio_time():
    pytest.importorskip("transformers", reason="transformers not installed (uv sync --extra asr)")
    pytest.importorskip("torch", reason="torch not installed (uv sync --extra asr)")
    from transcript.adapters.asr.nemotron_transcriber import NemotronTranscriber

    transcriber = NemotronTranscriber(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
        language="pt-BR",
        device="auto",
        chunk_ms=320,
        lookahead_s=1.4,
        sample_rate=SAMPLE_RATE,
    )
    await transcriber.load()
    try:
        assert transcriber.loaded
        await transcriber.open_stream("gpu-smoke")
        first = await transcriber.push_audio("gpu-smoke", THREE_SILENCE)
        assert isinstance(first, list)  # silence may decode to nothing — fine
        for word in first:
            assert 0.0 <= word.timestamp.start <= word.timestamp.end <= 3.0
        await transcriber.close_stream("gpu-smoke")
    finally:
        await transcriber.unload()
    assert not transcriber.loaded


@requires_gpu_env
async def test_pyannote_diarizes_a_short_window():
    pytest.importorskip("pyannote.audio", reason="pyannote.audio not installed")
    token = os.environ.get("TRANSCRIPT_HF_TOKEN")
    if not token:
        pytest.skip("TRANSCRIPT_HF_TOKEN is required for the gated community-1 pipeline")
    from transcript.adapters.diarization.pyannote_diarizer import PyannoteDiarizer

    diarizer = PyannoteDiarizer(
        pipeline_name="pyannote/speaker-diarization-community-1",
        token=token,
        device="auto",
    )
    await diarizer.load()
    try:
        assert diarizer.loaded
        turns = await diarizer.diarize(
            stream_id="gpu-smoke", pcm=THREE_SILENCE, sample_rate=SAMPLE_RATE, offset=10.0
        )
        assert isinstance(turns, list)
        for turn in turns:
            assert 10.0 <= turn.timestamp.start <= turn.timestamp.end <= 13.0
    finally:
        await diarizer.unload()
