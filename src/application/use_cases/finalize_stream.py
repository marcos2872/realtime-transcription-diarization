"""Use case: finalização de streaming (``stop``).

Transcreve cada canal acumulado, aplica ``apply_stop_rules``
(``mic`` → ``Eu``; ``system`` sem diarize → ``Sistema``) e monta o
``TranscriptData`` final. A diarização do canal ``system`` sempre
re-roda do zero no áudio completo (autoritativa); o cache parcial
só é usado como fallback se a nova execução falhar. Falha total
faz fallback para genérico, nunca 500.
"""

from __future__ import annotations

import logging

from src.application.ports import AudioTempFiles, DiarizerPort, TranscriberPort
from src.domain.entities.transcript import TranscriptData, build_transcript
from src.domain.services.speaker_rules import apply_stop_rules

logger = logging.getLogger(__name__)


async def finalize_stream(
    session,
    transcriber: TranscriberPort,
    tmp_files: AudioTempFiles,
    diarizer: DiarizerPort | None = None,
) -> TranscriptData:
    """Transcreve todos os canais da sessão e devolve a transcrição."""
    all_segments: list[dict] = []
    for channel in session.channels:
        wav_data = session.get_audio(channel)
        if wav_data is None:
            continue
        tmp_path = tmp_files.save(wav_data)
        try:
            segments = await transcriber.dispatch(tmp_path, session.language)
            if channel == "system" and session.diarize and diarizer is not None:
                segments = await _diarize_channel(session, diarizer, tmp_path, segments)
            segments = apply_stop_rules(segments, channel, session.diarize)
            all_segments.extend(segments)
        finally:
            tmp_files.remove(tmp_path)
    return build_transcript(
        session.id, all_segments, session.duration_sec, session.language
    )


async def _diarize_channel(session, diarizer: DiarizerPort, tmp_path: str,
                           segments: list[dict]) -> list[dict]:
    """Diariza do zero (autoritativo); cai para o cache parcial se falhar."""
    hints = {
        "min_speakers": getattr(session, "min_speakers", None),
        "max_speakers": getattr(session, "max_speakers", None),
    }
    try:
        diarization = diarizer.diarize(tmp_path, **hints)
        return diarizer.assign_speakers(segments, diarization)
    except Exception as exc:
        logger.warning("Diarização final falhou (%s); tentando cache parcial", exc)
    try:
        cached = session.cached_diarization("system")
        if cached is None:
            raise RuntimeError("sem cache parcial")
        return diarizer.assign_speakers(segments, cached)
    except Exception as exc:
        logger.warning("Diarização final sem fallback: %s", exc)
        return segments
