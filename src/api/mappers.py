"""Mappers wire ↔ domínio — conversão isolada em um só lugar.

Rotas nunca montam ``TranscriptionResult`` manualmente: convertem o
``TranscriptData`` do use case via ``to_wire``.
"""

from __future__ import annotations

from src.api.schemas import Segment, TranscriptionResult
from src.domain.entities.transcript import TranscriptData


def to_wire(result: TranscriptData) -> TranscriptionResult:
    """Converte agregado de domínio → schema de wire (camelCase)."""
    return TranscriptionResult(
        sessionId=result.session_id,
        segments=[
            Segment(
                speaker=s.speaker,
                text=s.text,
                tStart=s.t_start,
                tEnd=s.t_end,
            )
            for s in result.segments
        ],
        participants=list(result.participants),
        durationSec=result.duration_sec,
        language=result.language,
    )


def to_domain(result: TranscriptionResult) -> TranscriptData:
    """Converte schema de wire → agregado de domínio (ex: ``/refine``)."""
    from src.domain.entities.transcript import SegmentData

    return TranscriptData(
        session_id=result.sessionId,
        segments=[
            SegmentData(speaker=s.speaker, text=s.text, t_start=s.tStart, t_end=s.tEnd)
            for s in result.segments
        ],
        participants=list(result.participants),
        duration_sec=result.durationSec,
        language=result.language,
    )
