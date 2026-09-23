"""Entidade Transcrição — agregado de segmentos com invariantes.

Puro: sem Pydantic, sem FastAPI. A camada de API converte
``TranscriptData`` para o schema de wire (camelCase).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.services.speaker_rules import GENERIC_SPEAKER
from src.domain.value_objects.timestamp import round2


@dataclass(frozen=True)
class SegmentData:
    """Um trecho transcrito com janela temporal."""

    speaker: str
    text: str
    t_start: float
    t_end: float

    def normalized(self) -> SegmentData:
        """Retorna cópia com timestamps arredondados e texto limpo."""
        return SegmentData(
            speaker=self.speaker or GENERIC_SPEAKER,
            text=self.text.strip(),
            t_start=round2(self.t_start),
            t_end=round2(self.t_end),
        )


@dataclass(frozen=True)
class TranscriptData:
    """Resultado de transcrição: segmentos + participantes + metadados."""

    session_id: str
    segments: list[SegmentData] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    language: str = "pt"


def build_transcript(
    session_id: str,
    segments: list[dict],
    duration_sec: float,
    language: str,
) -> TranscriptData:
    """Constrói um ``TranscriptData`` a partir de segmentos brutos.

    Regras (por que existem):
    - ``speaker`` ausente/vazio vira ``"Locutor"`` genérico — o wire
      nunca recebe string vazia;
    - ``participants`` preserva ordem de primeira aparição, sem
      duplicatas — usado pelo front para listar locutores.
    """
    normalized = [
        SegmentData(
            speaker=s.get("speaker") or GENERIC_SPEAKER,
            text=(s.get("text") or "").strip(),
            t_start=round2(s.get("tStart", 0.0)),
            t_end=round2(s.get("tEnd", 0.0)),
        )
        for s in segments
    ]
    participants = list(dict.fromkeys(s.speaker for s in normalized))
    return TranscriptData(
        session_id=session_id,
        segments=normalized,
        participants=participants,
        duration_sec=round2(duration_sec),
        language=language,
    )
