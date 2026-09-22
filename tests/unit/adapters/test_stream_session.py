"""Tests for the streaming session buffer mechanics (no model needed)."""

import numpy as np
import pytest

from transcript.adapters.asr.nemotron_transcriber import ChunkPlanner, _StreamSession


def make_session() -> _StreamSession:
    planner = ChunkPlanner(
        first_samples=4040,
        per_chunk_samples=5520,
        first_mel=25,
        per_mel=32,
        hop_length=160,
        n_fft=512,
    )
    return _StreamSession("s1", planner, sample_rate=16_000)


def int16_bytes(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def test_append_tracks_sample_count():
    session = make_session()

    session.append_pcm(int16_bytes([0] * 1000))

    assert session.available == 1000
    assert session.samples_pushed == 1000


def test_materialize_returns_exact_window():
    session = make_session()
    session.append_pcm(int16_bytes(list(range(5000))))

    view = session.materialize(0, 4040)

    assert len(view) == 4040
    assert view[0] == pytest.approx(0.0)
    assert view[4039] == pytest.approx(4039 / 32768.0)


def test_materialize_pads_past_available_audio():
    session = make_session()
    session.append_pcm(int16_bytes([1000] * 4900))

    view = session.materialize(4800, 5520)

    assert len(view) == 720
    assert np.allclose(view[:100], 1000 / 32768.0)  # real audio
    assert (view[100:] == 0.0).all()  # zero-padded tail


def test_drop_before_trims_consumed_prefix():
    session = make_session()
    session.append_pcm(int16_bytes(list(range(5000))))

    session.drop_before(3744)

    assert session.available == 5000 - 3744
    view = session.materialize(3744, 4000)
    assert view[0] == pytest.approx(3744 / 32768.0)


def test_plan_materialize_drop_flows_like_the_pump():
    session = make_session()
    session.append_pcm(int16_bytes([7] * 15000))

    first = session.planner.plan(session.available, eos=False)
    assert first is not None and first.is_first
    view = session.materialize(first.start_sample, first.end_sample)
    assert len(view) == 4040
    session.drop_before(session.planner.last_start)

    second = session.planner.plan(session.available, eos=False)
    assert second is not None and not second.is_first
    assert (second.start_sample, second.end_sample) == (3744, 9264)
    view = session.materialize(second.start_sample, second.end_sample)
    assert len(view) == 5520
    assert (view == 7 / 32768.0).all()
