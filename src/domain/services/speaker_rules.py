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


# Sobreposição mínima (s) para considerar que um label cru do run atual
# é a mesma voz de um Pessoa já rotulado. Abaixo disso, vira Pessoa novo.
MIN_STABLE_OVERLAP_SEC = 0.5


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Duração da sobreposição entre duas janelas (0 se disjuntas)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def match_raw_label(
    t_start: float, t_end: float, diarization: list[dict]
) -> str | None:
    """Label cru cuja janela contém o ponto médio; senão o mais próximo."""
    mid = (t_start + t_end) / 2.0
    best: str | None = None
    best_dist = float("inf")
    for dseg in diarization:
        ds, de = dseg["tStart"], dseg["tEnd"]
        if ds <= mid <= de:
            return dseg["speaker"]
        dist = min(abs(mid - ds), abs(mid - de))
        if dist < best_dist:
            best_dist, best = dist, dseg["speaker"]
    return best


def resolve_stable_labels(
    diarization: list[dict],
    timeline: list[dict],
    next_idx: int,
) -> tuple[dict[str, str], int]:
    """Mapeia labels crus do run atual → ``Pessoa N`` estável.

    Cada label cru herda o Pessoa com maior sobreposição temporal no
    histórico já rotulado — resolve a permutação de clusters entre
    execuções do pyannote. Sem sobreposição suficiente, ganha um
    Pessoa novo. Devolve ``(mapa, próximo índice)``.
    """
    mapping: dict[str, str] = {}
    for dseg in diarization:
        raw = dseg["speaker"]
        if raw in mapping:
            continue
        overlap_by_person: dict[str, float] = {}
        for past in timeline:
            ov = overlap(dseg["tStart"], dseg["tEnd"], past["tStart"], past["tEnd"])
            if ov > 0:
                overlap_by_person[past["speaker"]] = (
                    overlap_by_person.get(past["speaker"], 0.0) + ov
                )
        best = max(overlap_by_person, key=overlap_by_person.get, default=None)
        if best is not None and overlap_by_person[best] >= MIN_STABLE_OVERLAP_SEC:
            mapping[raw] = best
        else:
            mapping[raw] = person_label(next_idx)
            next_idx += 1
    return mapping, next_idx


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
