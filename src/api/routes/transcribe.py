"""Rota de transcrição em lote (``POST /transcribe``)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.deps import get_dispatcher, get_settings
from src.api.mappers import to_wire
from src.api.schemas import TranscriptionResult
from src.application.use_cases.transcribe_audio import transcribe_audio
from src.infrastructure.audio.wav import WavTempFiles
from src.infrastructure.diarization.pyannote import PyannoteDiarizer

router = APIRouter(tags=["transcribe"])


@router.post(
    "/transcribe",
    response_model=TranscriptionResult,
    summary="Transcrever arquivo WAV",
    description="Envia um WAV (16kHz, 16-bit, mono) e recebe a transcrição "
    "completa. Com `diarize=true`, identifica locutores (`Pessoa N`); "
    "sem `HF_TOKEN`, faz fallback para `Locutor` genérico sem erro.",
    responses={
        200: {"description": "Transcrição concluída."},
        400: {"description": "Áudio inválido ou ausente."},
        413: {"description": "Arquivo excede `MAX_FILE_SIZE_MB`."},
    },
    operation_id="transcribeAudio",
)
async def transcribe(
    audio: UploadFile = File(..., description="Arquivo WAV 16kHz 16-bit mono."),
    language: str = Form("pt", description="Idioma (ISO 639-1).", examples=["pt"]),
    sessionId: str | None = Form(None, description="ID opcional para rastreamento."),
    diarize: bool = Form(False, description="Executar diarização (requer `HF_TOKEN`)."),
    minSpeakers: int | None = Form(None, description="Piso de locutores."),
    maxSpeakers: int | None = Form(None, description="Teto de locutores."),
    dispatcher=Depends(get_dispatcher),
    settings=Depends(get_settings),
) -> TranscriptionResult:
    """Transcrição por lote."""
    audio_bytes = await audio.read()
    result = await transcribe_audio(
        audio_bytes=audio_bytes,
        language=language,
        diarize=diarize,
        session_id=sessionId,
        transcriber=dispatcher,
        tmp_files=WavTempFiles(),
        diarizer=PyannoteDiarizer(),
        min_speakers=minSpeakers,
        max_speakers=maxSpeakers,
    )
    return to_wire(result)
