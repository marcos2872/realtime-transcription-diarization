"""Rotas de streaming (``POST /stream/...``, ``GET /stream/.../events``).

Finas por design: validam, chamam o use case ``finalize_stream`` no
``stop`` e convertem via ``to_wire``. Regras de locutor vivem no
domínio (``apply_stop_rules``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.deps import get_dispatcher, get_sessions
from src.api.mappers import to_wire
from src.api.schemas import AudioChunk, StreamAction, TranscriptionResult
from src.api.sse.events import partial_event_generator
from src.application.use_cases.finalize_stream import finalize_stream
from src.infrastructure.audio.wav import WavTempFiles
from src.infrastructure.diarization.pyannote import PyannoteDiarizer

router = APIRouter(tags=["streaming"])


@router.post(
    "/stream/{session_id}",
    summary="Iniciar ou finalizar sessão de streaming",
    description="`action=start` cria a sessão; `action=stop` transcreve o "
    "áudio acumulado (`mic` → `Eu`; `system` sem diarize → `Sistema`) e "
    "remove a sessão. `sessionId` vai no body **e** no path.",
    responses={
        200: {"description": "Sessão criada ou transcrição final."},
        400: {"description": "`action` inválida."},
        404: {"description": "Sessão não encontrada no `stop`."},
    },
    operation_id="streamAction",
)
async def stream_action(
    session_id: str,
    action: StreamAction,
    sessions=Depends(get_sessions),
    dispatcher=Depends(get_dispatcher),
):
    """Gerencia sessões de streaming."""
    if action.action == "start":
        session = sessions.create(
            session_id=session_id,
            language=action.language,
            channels=action.channels,
            diarize=action.diarize,
            min_speakers=action.minSpeakers,
            max_speakers=action.maxSpeakers,
        )
        return {"ok": True, "sessionId": session.id}

    if action.action == "stop":
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Sessão não encontrada")
        session.close()
        result = await finalize_stream(
            session,
            transcriber=dispatcher,
            tmp_files=WavTempFiles(),
            diarizer=PyannoteDiarizer(),
        )
        sessions.remove(session_id)
        return to_wire(result)

    raise HTTPException(400, "action deve ser 'start' ou 'stop'")


@router.post(
    "/stream/{session_id}/audio",
    summary="Enviar chunk de áudio",
    description="PCM Int16LE 16kHz mono em base64 para uma sessão ativa.",
    responses={
        200: {"description": "Chunk aceito."},
        404: {"description": "Sessão inexistente ou finalizada."},
    },
    operation_id="streamAudio",
)
async def stream_audio(
    session_id: str,
    chunk: AudioChunk,
    sessions=Depends(get_sessions),
):
    """Recebe um chunk de áudio para uma sessão de streaming."""
    session = sessions.get(session_id)
    if not session or session.closed:
        raise HTTPException(404, "Sessão não encontrada ou já finalizada")
    await session.add_audio(chunk.channel, chunk.seq, chunk.data)
    return {"ok": True, "seq": chunk.seq}


@router.get(
    "/stream/{session_id}/events",
    summary="Eventos parciais via SSE",
    description="Emite `partial` a cada ~3s com o áudio novo desde o último "
    "flush; ao fechar, emite `heartbeat/closed`. Documentação do payload: "
    "`StreamEvent`.",
    responses={
        200: {"description": "Stream SSE de parciais."},
        404: {"description": "Sessão não encontrada."},
    },
    operation_id="streamEvents",
)
async def stream_events(
    session_id: str,
    request: Request,
    sessions=Depends(get_sessions),
    dispatcher=Depends(get_dispatcher),
):
    """SSE: escuta eventos parciais da transcrição."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Sessão não encontrada")
    return EventSourceResponse(partial_event_generator(session, request, dispatcher))
