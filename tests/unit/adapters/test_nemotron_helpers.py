"""Tests for the pure helpers of the Nemotron adapter (no GPU/model needed)."""

from types import SimpleNamespace

import pytest

from transcript.adapters.asr.nemotron_transcriber import (
    _resolve_stft_params,
    allocate_word_spans,
    right_context_frames,
    split_new_words,
    strip_language_tag,
)
from transcript.application.errors import TranscriptionFailed


@pytest.mark.parametrize(
    ("chunk_ms", "expected_frames"),
    [(80, 0), (160, 1), (320, 3), (560, 6), (1120, 13)],
)
def test_right_context_frames_follow_nemo_supported_lookaheads(chunk_ms, expected_frames):
    assert right_context_frames(chunk_ms) == expected_frames


def test_right_context_frames_rejects_unsupported_chunks():
    with pytest.raises(ValueError, match="unsupported chunk_ms"):
        right_context_frames(100)


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


def test_resolve_stft_params_prefers_live_module_attributes():
    preprocessor = SimpleNamespace(
        hop_length=160, n_fft=512, featurizer=SimpleNamespace(hop_length=160, n_fft=512)
    )

    assert _resolve_stft_params(preprocessor, {}, 16_000) == (160, 512)


def test_resolve_stft_params_reads_sample_counts_from_config():
    preprocessor = SimpleNamespace(featurizer=None)
    cfg = {"hop_length": 160, "n_fft": 512}

    assert _resolve_stft_params(preprocessor, cfg, 16_000) == (160, 512)


def test_resolve_stft_params_derives_samples_from_second_based_keys():
    # Nemotron-style config: seconds instead of sample counts.
    preprocessor = SimpleNamespace(featurizer=None)
    cfg = {"sample_rate": 16000, "window_stride": 0.01, "window_size": 0.02}

    assert _resolve_stft_params(preprocessor, cfg, 16_000) == (160, 320)


def test_resolve_stft_params_falls_back_to_default_sample_rate():
    preprocessor = SimpleNamespace(featurizer=None)
    cfg = {"window_stride": 0.01, "window_size": 0.02}

    assert _resolve_stft_params(preprocessor, cfg, 16_000) == (160, 320)


def test_resolve_stft_params_fails_listing_available_keys():
    with pytest.raises(TranscriptionFailed, match="preprocessor config keys"):
        _resolve_stft_params(SimpleNamespace(featurizer=None), {"sample_rate": 16000}, 16_000)
