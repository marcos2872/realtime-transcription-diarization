"""Regras de locutor — vocabulário ubíquo do domínio.

Por que estes valores:
- ``"Eu"``: canal ``mic`` é sempre o usuário local, sem diarização;
- ``"Sistema"``: canal ``system`` sem diarização (fallback que nunca
  gera 500 — diarização falhou = genérico, não erro);
- ``"Pessoa N"``: rótulo estável da diarização pyannote, mapeado por
  ordem de primeira aparição;
- ``"Locutor"``: genérico quando nada se sabe (batch sem diarize).
"""

from __future__ import annotations

MIC_SPEAKER = "Eu"
SYSTEM_SPEAKER = "Sistema"
GENERIC_SPEAKER = "Locutor"


def person_label(index: int) -> str:
    """``1`` → ``"Pessoa 1"``. Índice base-1, como na diarização."""
    return f"Pessoa {index}"


def normalize_speaker(speaker: str | None, fallback: str = GENERIC_SPEAKER) -> str:
    """Devolve ``speaker`` ou ``fallback`` quando vazio/ausente."""
    return speaker if speaker else fallback


def apply_stop_rules(
    segments: list[dict],
    channel: str,
    diarize: bool,
) -> list[dict]:
    """Aplica a regra de locutor do ``stop`` de streaming.

    - ``mic`` → sempre ``"Eu"`` (diarização irrelevante no microfone);
    - ``system`` + ``diarize=True`` → mantém o locutor da diarização
      (chamador já atribuiu; aqui só normaliza vazios);
    - demais casos → ``"Sistema"`` quando genérico/vazio.
    """
    result = []
    for seg in segments:
        speaker = seg.get("speaker") or GENERIC_SPEAKER
        if channel == "mic":
            speaker = MIC_SPEAKER
        elif channel == "system" and diarize:
            speaker = normalize_speaker(speaker)
        elif speaker == GENERIC_SPEAKER:
            speaker = SYSTEM_SPEAKER
        result.append({**seg, "speaker": speaker})
    return result
