"""Use case: transcrição em lote (``POST /transcribe``).

Orquestra: transcrever via dispatcher → diarizar (opcional, fails
open) → montar ``TranscriptData``. Sem FastAPI, sem arquivos: recebe
bytes e ports injetados (Dependency Inversion — nunca importa
infraestrutura).
"""

from __future__ import annotations

import logging
import os
import uuid

from src.application.ports import AudioTempFiles, DiarizerPort, TranscriberPort
from src.domain.entities.transcript import TranscriptData, build_transcript
from src.domain.value_objects.audio_format import estimate_duration_sec

logger = logging.getLogger(__name__)


async def transcribe_audio(
    audio_bytes: bytes,
    language: str,
    diarize: bool,
    session_id: str | None,
    transcriber: TranscriberPort,
    tmp_files: AudioTempFiles,
    diarizer: DiarizerPort | None = None,
) -> TranscriptData:
    """Transcreve ``audio_bytes`` (WAV) e devolve o agregado de domínio."""
    sid = session_id or uuid.uuid4().hex[:12]
    tmp_path = tmp_files.save(audio_bytes)
    try:
        duration_sec = estimate_duration_sec(os.path.getsize(tmp_path))
        segments = await transcriber.dispatch(tmp_path, language)
        if diarize and diarizer is not None:
            try:
                diarization = diarizer.diarize(tmp_path)
                segments = diarizer.assign_speakers(segments, diarization)
            except Exception as exc:
                logger.warning("Diarização falhou (continuando sem): %s", exc)
        return build_transcript(sid, segments, duration_sec, language)
    finally:
        tmp_files.remove(tmp_path)
