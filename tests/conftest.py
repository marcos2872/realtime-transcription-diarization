"""Shared fixtures for the whole test suite."""

import pytest

from transcript.adapters.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings wired for hermetic tests: fake providers, fast diarization cadence."""
    return Settings(
        asr_provider="fake",
        diarization_provider="fake",
        language="pt-BR",
        max_streams=4,
        diarization_min_s=1.0,
        diarization_hop_s=1.0,
        diarization_overlap_s=1.0,
        diarization_window_s=5.0,
        emit_partials=True,
    )
