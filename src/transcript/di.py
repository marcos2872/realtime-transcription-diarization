"""Composition root: the only module allowed to know concrete adapters.

Everything else talks to the domain ports; this module picks the
implementations from settings and wires the use cases together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from transcript import __version__
from transcript.adapters.config.settings import Settings
from transcript.adapters.testing import FakeDiarizer, FakeTranscriber
from transcript.application.stream_registry import StreamRegistry
from transcript.application.use_cases import CloseStream, OpenStream, ProcessChunk
from transcript.domain.ports import Diarizer, Transcriber

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Static service description served by ``GET /``."""

    name: str
    version: str
    language: str
    sample_rate: int
    max_streams: int
    asr_provider: str
    asr_model: str
    chunk_ms: int
    diarization_provider: str
    diarization_pipeline: str
    device: str


@dataclass
class Application:
    """The wired application: use cases + registry + provider lifecycle."""

    settings: Settings
    registry: StreamRegistry
    open_stream: OpenStream
    process_chunk: ProcessChunk
    close_stream: CloseStream
    info: ServerInfo
    _providers: list = field(default_factory=list, repr=False)

    async def startup(self) -> None:
        """Load heavy providers (models); roll back on the first failure."""
        loaded: list = []
        try:
            for provider in self._providers:
                load = getattr(provider, "load", None)
                if load is None:
                    continue  # lightweight providers (e.g. fakes) need no lifecycle
                await load()
                loaded.append(provider)
        except Exception:
            for provider in reversed(loaded):
                try:
                    await provider.unload()
                except Exception:
                    logger.exception("failed to unload %s during startup rollback", provider)
            raise

    async def shutdown(self) -> None:
        """Unload providers in reverse order; failures are logged, never raised."""
        for provider in reversed(self._providers):
            unload = getattr(provider, "unload", None)
            if unload is None:
                continue
            try:
                await unload()
            except Exception:
                logger.exception("failed to unload %s", provider)


def resolve_device(preference: str) -> str:
    """Turn the ``auto`` device preference into a concrete device for reporting."""
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_application(settings: Settings) -> Application:
    transcriber: Transcriber = _build_transcriber(settings)
    diarizer: Diarizer = _build_diarizer(settings)

    registry = StreamRegistry()
    # Retention covers the widest diarization window plus one hop of slack.
    audio_retention_s = settings.diarization_window_s + settings.diarization_hop_s
    open_stream = OpenStream(
        registry,
        transcriber,
        max_streams=settings.max_streams,
        sample_rate=settings.sample_rate,
        audio_retention_s=audio_retention_s,
    )
    process_chunk = ProcessChunk(
        registry,
        transcriber,
        diarizer,
        sample_rate=settings.sample_rate,
        window_s=settings.diarization_window_s,
        hop_s=settings.diarization_hop_s,
        overlap_s=settings.diarization_overlap_s,
        min_s=settings.diarization_min_s,
    )
    close_stream = CloseStream(
        registry,
        transcriber,
        diarizer,
        sample_rate=settings.sample_rate,
        window_s=settings.diarization_window_s,
        overlap_s=settings.diarization_overlap_s,
    )

    info = ServerInfo(
        name="transcript-service",
        version=__version__,
        language=settings.language,
        sample_rate=settings.sample_rate,
        max_streams=settings.max_streams,
        asr_provider=settings.asr_provider,
        asr_model=settings.asr_model,
        chunk_ms=settings.chunk_ms,
        diarization_provider=settings.diarization_provider,
        diarization_pipeline=settings.diarization_pipeline,
        device=resolve_device(settings.device),
    )
    return Application(
        settings=settings,
        registry=registry,
        open_stream=open_stream,
        process_chunk=process_chunk,
        close_stream=close_stream,
        info=info,
        _providers=[transcriber, diarizer],
    )


def _build_transcriber(settings: Settings) -> Transcriber:
    if settings.asr_provider == "fake":
        return FakeTranscriber(sample_rate=settings.sample_rate)
    from transcript.adapters.asr.nemotron_transcriber import NemotronTranscriber

    return NemotronTranscriber(
        model_name=settings.asr_model,
        language=settings.language,
        device=settings.device,
        chunk_ms=settings.chunk_ms,
        lookahead_s=settings.lookahead_s,
        sample_rate=settings.sample_rate,
    )


def _build_diarizer(settings: Settings) -> Diarizer:
    if settings.diarization_provider == "fake":
        return FakeDiarizer()
    from transcript.adapters.diarization.pyannote_diarizer import PyannoteDiarizer

    return PyannoteDiarizer(
        pipeline_name=settings.diarization_pipeline,
        token=settings.hf_token,
        device=settings.device,
        threshold=settings.diarization_threshold,
        min_speakers=settings.diarization_min_speakers,
        max_speakers=settings.diarization_max_speakers,
    )
