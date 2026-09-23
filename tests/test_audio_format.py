"""Testes de domínio: formato de áudio e timestamps."""

from src.domain.value_objects.audio_format import (
    BYTES_PER_SECOND,
    WAV_HEADER_SIZE,
    estimate_duration_sec,
)
from src.domain.value_objects.timestamp import round2


def test_estimate_duration():
    assert estimate_duration_sec(44) == 0.0
    assert estimate_duration_sec(10) == 0.0
    assert estimate_duration_sec(44 + BYTES_PER_SECOND) == 1.0
    assert estimate_duration_sec(44 + 32000 * 83.6) == 83.6


def test_wav_header_size():
    assert WAV_HEADER_SIZE == 44


def test_round2():
    assert round2(0.123) == 0.12
    assert round2(None) == 0.0
    assert round2("bad") == 0.0
