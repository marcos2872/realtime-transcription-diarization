"""Use case: refinamento via LLM (``POST /refine``).

Delega ao ``RefinerPort`` (fails open) e reconstrói o agregado com
os mesmos ``sessionId``/``durationSec``/``language`` originais.
"""

from __future__ import annotations

from src.application.ports import RefinerPort
from src.domain.entities.transcript import TranscriptData, build_transcript


async def refine_transcript(
    transcript: TranscriptData,
    model: str | None,
    prompt: str | None,
    default_model: str,
    refiner: RefinerPort,
) -> tuple[TranscriptData, str]:
    """Refina e devolve ``(transcrição refinada, modelo usado)``."""
    raw = [
        {
            "speaker": s.speaker,
            "text": s.text,
            "tStart": s.t_start,
            "tEnd": s.t_end,
        }
        for s in transcript.segments
    ]
    chosen = model or default_model
    refined = await refiner.refine(
        segments=raw,
        language=transcript.language,
        model=chosen,
        prompt=prompt,
    )
    return (
        build_transcript(
            transcript.session_id, refined, transcript.duration_sec, transcript.language
        ),
        chosen,
    )
