"""Tests for the pure helpers of the Nemotron adapter (no GPU/model needed)."""

import pytest

from transcript.adapters.asr.nemotron_transcriber import (
    ChunkPlanner,
    allocate_word_spans,
    pcm_bytes_to_float,
    right_context_frames,
    split_new_words,
    strip_language_tag,
)


@pytest.mark.parametrize(
    ("chunk_ms", "expected_frames"),
    [(80, 0), (320, 3), (560, 6), (1120, 13)],
)
def test_right_context_frames_follow_supported_lookaheads(chunk_ms, expected_frames):
    assert right_context_frames(chunk_ms) == expected_frames


def test_right_context_frames_rejects_unsupported_chunks():
    # 100 was never valid; 160 exists in NeMo but not in the Transformers port.
    with pytest.raises(ValueError, match="unsupported chunk_ms"):
        right_context_frames(100)
    with pytest.raises(ValueError, match="unsupported chunk_ms"):
        right_context_frames(160)


def test_strip_language_tag_removes_the_auto_detect_suffix():
    assert strip_language_tag("Bom dia. <pt-BR>") == "Bom dia."
    assert strip_language_tag("Bom dia. <pt-BR>   ") == "Bom dia."


def test_strip_language_tag_keeps_text_without_tags():
    assert strip_language_tag("Sem tag nenhuma.") == "Sem tag nenhuma."


def test_split_new_words_on_first_observation():
    assert split_new_words("", "Olá mundo") == ["Olá", "mundo"]


def test_split_new_words_returns_only_the_grown_suffix():
    assert split_new_words("Olá mundo", "Olá mundo das ideias") == ["das", "ideias"]


def test_split_new_words_is_empty_when_nothing_changed():
    assert split_new_words("Olá mundo.", "Olá mundo.") == []


def test_split_new_words_handles_punctuation_resegmentation():
    # Stateful decoding may finish a sentence by swapping the last character.
    assert split_new_words("tudo bem", "tudo bem?") == ["?"]


def test_allocate_word_spans_distributes_evenly():
    assert allocate_word_spans(4, 0.0, 4.0) == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def test_allocate_word_spans_degenerate_span_yields_zero_length_words():
    assert allocate_word_spans(2, 5.0, 5.0) == [(5.0, 5.0), (5.0, 5.0)]


def test_allocate_word_spans_without_words():
    assert allocate_word_spans(0, 0.0, 1.0) == []


def make_planner() -> ChunkPlanner:
    """Planner with the real lookahead-3 geometry (320 ms chunks)."""
    return ChunkPlanner(
        first_samples=4040,
        per_chunk_samples=5520,
        first_mel=25,
        per_mel=32,
        hop_length=160,
        n_fft=512,
    )


def test_planner_emits_the_first_window_once_covered():
    planner = make_planner()

    assert planner.plan(4039, eos=False) is None
    plan = planner.plan(4040, eos=False)

    assert plan is not None
    assert (plan.start_sample, plan.end_sample, plan.mel_frames) == (0, 4040, 25)
    assert plan.is_first and not plan.final


def test_planner_matches_the_documented_sliding_windows():
    planner = make_planner()
    planner.plan(4040, eos=False)

    # start = mel_idx * hop - n_fft // 2 = 25 * 160 - 256
    plan = planner.plan(9264, eos=False)

    assert plan is not None
    assert (plan.start_sample, plan.end_sample, plan.mel_frames) == (3744, 9264, 32)
    assert not plan.is_first and not plan.final

    plan = planner.plan(9264 + 5120, eos=False)

    assert plan is not None
    assert (plan.start_sample, plan.end_sample) == (8864, 14384)


def test_planner_pads_the_tail_only_at_eos():
    planner = make_planner()
    planner.plan(4040, eos=False)

    assert planner.plan(5000, eos=False) is None  # second window needs 9264

    plan = planner.plan(5000, eos=True)

    assert plan is not None
    assert (plan.start_sample, plan.end_sample) == (3744, 9264)
    assert plan.final


def test_planner_ends_quietly_when_nothing_new_at_eos():
    planner = make_planner()

    assert planner.plan(0, eos=True) is None  # stream closed with no audio

    planner.plan(4040, eos=False)
    planner.plan(9264, eos=False)

    assert planner.plan(8864, eos=True) is None  # available == next start: nothing new


def test_planner_first_window_pads_a_short_stream_at_eos():
    planner = make_planner()

    plan = planner.plan(1000, eos=True)

    assert plan is not None
    assert (plan.start_sample, plan.end_sample) == (0, 4040)
    assert plan.is_first and plan.final


def test_pcm_bytes_to_float_scales_and_clips():
    import struct

    pcm = struct.pack("<3h", 0, 32767, -32768)
    floats = pcm_bytes_to_float(pcm)

    assert floats[0] == 0.0
    assert floats[1] == pytest.approx(32767 / 32768.0)
    assert floats[2] == -1.0
