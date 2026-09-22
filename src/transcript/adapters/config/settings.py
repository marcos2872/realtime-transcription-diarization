"""Runtime settings, sourced from environment variables with the ``TRANSCRIPT_`` prefix."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ASRT_PROVIDER = Literal["nemotron", "fake"]
_DiarIZATION_PROVIDER = Literal["pyannote", "fake"]


class Settings(BaseSettings):
    """Everything the composition root needs to wire the application.

    All fields can be overridden via environment variables, e.g.
    ``TRANSCRIPT_MAX_STREAMS=4`` or ``TRANSCRIPT_HF_TOKEN=hf_...``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    max_streams: int = Field(default=4, ge=1, le=64)
    sample_rate: int = Field(default=16_000, ge=8_000)
    language: str = "auto"  # e.g. "pt-BR" or "auto" (language-ID prompt)
    emit_partials: bool = True  # server default; a connection may override with ?partials=

    # --- ASR (transcription) ---
    asr_provider: _ASRT_PROVIDER = "nemotron"
    asr_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    chunk_ms: Literal[80, 160, 320, 560, 1120] = 320  # streaming chunk (latency/accuracy trade-off)
    lookahead_s: float = Field(
        default=1.4,
        ge=0.0,
        description="Right-context latency used to anchor approximate word timestamps.",
    )

    # --- diarization ---
    diarization_provider: _DiarIZATION_PROVIDER = "pyannote"
    diarization_pipeline: str = "pyannote/speaker-diarization-community-1"
    diarization_window_s: float = Field(default=30.0, ge=1.0)
    diarization_hop_s: float = Field(default=5.0, ge=0.5)
    diarization_overlap_s: float = Field(default=7.0, ge=0.0)
    diarization_min_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum audio before the first diarization run (avoid tiny windows).",
    )

    # --- environment / credentials ---
    device: str = "auto"  # auto | cuda | cpu
    hf_token: str | None = None  # required by the gated pyannote community-1 pipeline

    @model_validator(mode="after")
    def _diarization_geometry_must_be_coherent(self) -> Settings:
        # Each diarization run reports speaker turns for
        # [previous_run_end - overlap, current_end]; that region must fit inside
        # the audio window we actually feed the model, or we would emit turns
        # for audio we never processed.
        if self.diarization_overlap_s >= self.diarization_window_s:
            raise ValueError("TRANSCRIPT_DIARIZATION_OVERLAP_S must be < window")
        if self.diarization_hop_s + self.diarization_overlap_s > self.diarization_window_s:
            raise ValueError("TRANSCRIPT_DIARIZATION_HOP_S + overlap must fit within the window")
        return self

    @property
    def bytes_per_second(self) -> int:
        """PCM16 mono byte rate for the configured sample rate."""
        return self.sample_rate * 2


def load_settings() -> Settings:
    """Load settings from the environment (the only place that reads env vars)."""
    return Settings()
