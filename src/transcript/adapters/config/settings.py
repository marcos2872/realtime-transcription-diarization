"""Runtime settings, sourced from environment variables with the ``TRANSCRIPT_`` prefix."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ASRT_PROVIDER = Literal["nemotron", "fake"]
_DiarIZATION_PROVIDER = Literal["pyannote", "fake"]

# Streaming chunk sizes the Transformers port supports (latency/accuracy trade-off).
# Note: 160 ms exists in NeMo but the Transformers port only supports [3, 0, 6, 13].
CHUNK_MS_OPTIONS = (80, 320, 560, 1120)


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
    chunk_ms: int = Field(
        default=320,
        description="Streaming chunk size in ms (latency/accuracy trade-off).",
    )
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
    diarization_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description=(
            "VBx pre-clustering threshold: lower splits voices more aggressively "
            "(helps when two speakers get merged into one), higher merges more."
        ),
    )
    diarization_min_speakers: int | None = Field(
        default=None,
        ge=1,
        description="Force at least this many speakers (e.g. 2 for a known 2-person call).",
    )
    diarization_max_speakers: int | None = Field(
        default=None,
        ge=1,
        description="Cap the speaker count (prevents one voice splitting into many).",
    )

    # --- environment / credentials ---
    device: str = "auto"  # auto | cuda | cpu
    hf_token: str | None = None  # required by the gated pyannote community-1 pipeline

    @field_validator("chunk_ms", mode="before")
    @classmethod
    def _coerce_chunk_ms(cls, value: object) -> int:
        # Env vars always arrive as strings; Literal[int, ...] would reject
        # '320', so coerce here and keep the friendly allowed-values check.
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"TRANSCRIPT_CHUNK_MS must be one of {list(CHUNK_MS_OPTIONS)}, got {value!r}"
            ) from None
        if number not in CHUNK_MS_OPTIONS:
            raise ValueError(
                f"TRANSCRIPT_CHUNK_MS must be one of {list(CHUNK_MS_OPTIONS)}, got {value!r}"
            )
        return number

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
        if (
            self.diarization_min_speakers is not None
            and self.diarization_max_speakers is not None
            and self.diarization_min_speakers > self.diarization_max_speakers
        ):
            raise ValueError("TRANSCRIPT_DIARIZATION_MIN_SPEAKERS must be <= MAX_SPEAKERS")
        return self

    @property
    def bytes_per_second(self) -> int:
        """PCM16 mono byte rate for the configured sample rate."""
        return self.sample_rate * 2


def load_settings() -> Settings:
    """Load settings from the environment (the only place that reads env vars)."""
    return Settings()
