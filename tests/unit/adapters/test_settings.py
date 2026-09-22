"""Tests for Settings coercion: env vars always arrive as strings."""

import pytest

from transcript.adapters.config.settings import Settings


def test_chunk_ms_accepts_the_env_string_form():
    assert Settings(chunk_ms="320").chunk_ms == 320


def test_chunk_ms_accepts_int():
    assert Settings(chunk_ms=560).chunk_ms == 560


@pytest.mark.parametrize("bad", ["100", "abc", ""])
def test_chunk_ms_rejects_values_outside_the_supported_set(bad):
    with pytest.raises(ValueError, match="TRANSCRIPT_CHUNK_MS must be one of"):
        Settings(chunk_ms=bad)


def test_numeric_fields_coerce_from_env_strings():
    settings = Settings(port="8001", max_streams="2", lookahead_s="1.0")

    assert settings.port == 8001
    assert settings.max_streams == 2
    assert settings.lookahead_s == pytest.approx(1.0)
