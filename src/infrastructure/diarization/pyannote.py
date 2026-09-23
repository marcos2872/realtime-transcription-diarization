"""Diarização de locutores via pyannote.audio.

Uso opcional — só carrega se o HF_TOKEN estiver configurado e
a flag `diarize=True` for passada na requisição.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Cache global do pipeline (carregado sob demanda)
_pipeline: Any = None


def _load_pipeline() -> Any:
    """Carrega o pipeline pyannote/speaker-diarization-3.1 com cache."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    hf_token = settings.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN não configurado. Defina HF_TOKEN no .env "
            "para usar diarização (é necessário aceitar os termos "
            "de uso em https://huggingface.co/pyannote/speaker-diarization-3.1)"
        )

    import torch
    from huggingface_hub import login as hf_login

    # pyannote.audio v4 usa `token` (o `use_auth_token` da v3 foi removido
    # no upstream — o monkey-patch antigo foi deletado junto no upgrade).
    # Faz login antes de carregar o pipeline (repositórios gated).
    hf_login(token=hf_token)

    logger.info("Carregando pyannote/speaker-diarization-3.1 ...")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,  # ← passa token explicitamente
    )

    # Move para GPU se disponível
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipeline.to(device)
        logger.info("Pyannote pipeline movido para GPU")
    else:
        logger.info("Pyannote pipeline rodando em CPU")

    _pipeline = pipeline
    return pipeline


def diarize(
    audio_path: str,
) -> list[dict[str, Any]]:
    """Executa diarização em um arquivo WAV.

    Returns:
        Lista de dicts: {speaker, tStart, tEnd}
    """
    import torch
    from pyannote.core import Annotation

    pipeline = _load_pipeline()

    logger.info("Diarizando %s ...", audio_path)
    output: Annotation = pipeline(audio_path)  # type: ignore

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in output.itertracks(yield_label=True):
        segments.append({
            "speaker": str(speaker),
            "tStart": round(float(turn.start), 2),
            "tEnd": round(float(turn.end), 2),
        })

    logger.info("Diarização concluída: %d segmentos, %d locutores",
                len(segments),
                len(set(s["speaker"] for s in segments)))
    return segments


def assign_speakers(
    transcript_segments: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atribui locutores de diarização aos segmentos de transcrição.

    Para cada segmento de transcrição, encontra o locutor cuja janela
    contém o ponto médio do segmento. Se não houver sobreposição,
    usa o locutor mais próximo.

    Retorna nova lista de segmentos com speaker atualizado.
    """
    if not diarization:
        return transcript_segments

    # Mapa de labels pyannote -> "Pessoa N"
    label_map: dict[str, str] = {}
    next_idx = 1

    def _label(speaker_id: str) -> str:
        nonlocal next_idx
        if speaker_id not in label_map:
            label_map[speaker_id] = f"Pessoa {next_idx}"
            next_idx += 1
        return label_map[speaker_id]

    def _find_speaker(t_start: float, t_end: float) -> str:
        mid = (t_start + t_end) / 2.0
        best = None
        best_dist = float("inf")

        for dseg in diarization:
            ds, de, spk = dseg["tStart"], dseg["tEnd"], dseg["speaker"]
            if ds <= mid <= de:
                return _label(spk)
            dist = min(abs(mid - ds), abs(mid - de))
            if dist < best_dist:
                best_dist = dist
                best = spk

        return _label(best) if best else "Pessoa 1"

    result = []
    for seg in transcript_segments:
        speaker = _find_speaker(seg.get("tStart", 0), seg.get("tEnd", 0))
        result.append({**seg, "speaker": speaker})

    return result


class PyannoteDiarizer:
    """Adapter ``DiarizerPort`` sobre as funções deste módulo.

    Permite injetar diarização nos use cases sem importar funções
    globais (Dependency Inversion). Falhas de pipeline propagam
    exceção — o use case decide o fallback (nunca 500).
    """

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        return diarize(audio_path)

    def assign_speakers(
        self,
        transcript_segments: list[dict[str, Any]],
        diarization: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return assign_speakers(transcript_segments, diarization)
