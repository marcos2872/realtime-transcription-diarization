import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.azure_ws import (
    SpeakerMapper,
    StreamConfig,
    build_phrases,
    bytes_to_ticks,
    error_msg,
    hypothesis_msg,
    phrase_msg,
)
from src.config import settings
from src.dispatcher import dispatcher
from src.refiner import Refiner
from src.schemas import (
    AudioChunk,
    HealthResponse,
    PartialResult,
    RefineRequest,
    RefinedResult,
    Segment,
    StreamAction,
    TranscriptionResult,
)
from src.session import session_manager

# ── Logging ──
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Refiner ──
refiner = Refiner()


# ── Lifespan ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o dispatcher (carrega Whisper nas GPUs) ao iniciar."""
    logger.info("Iniciando servidor STT API...")
    await dispatcher.start()
    logger.info(
        "Servidor pronto: %d GPU(s) | %s",
        len(dispatcher._transcribers),
        settings.refine_base_url,
    )
    yield
    logger.info("Parando servidor...")
    await dispatcher.stop()


app = FastAPI(
    title="STT API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS liberado para o front de teste em web/ (browser → API direto).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──


def _save_upload_to_temp(audio: UploadFile | bytes, suffix: str = ".wav") -> str:
    """Salva um upload em arquivo temporário e retorna o caminho."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if isinstance(audio, UploadFile):
            content = audio.file.read()
        else:
            content = audio
        tmp.write(content)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise


def _build_transcription_result(
    session_id: str,
    segments: list[dict],
    duration_sec: float,
    language: str,
) -> TranscriptionResult:
    """Constrói um TranscriptionResult a partir de segmentos brutos."""
    participants = list(dict.fromkeys(
        s.get("speaker", "Locutor") for s in segments
    ))
    return TranscriptionResult(
        sessionId=session_id,
        segments=[
            Segment(
                speaker=s.get("speaker", "Locutor"),
                text=s.get("text", ""),
                tStart=s.get("tStart", 0.0),
                tEnd=s.get("tEnd", 0.0),
            )
            for s in segments
        ],
        participants=participants,
        durationSec=duration_sec,
        language=language,
    )


# ── Rotas ──


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        gpus=settings.whisper_gpus_list,
        whisperLoaded=len(dispatcher._transcribers) > 0,
        refineEndpoint=settings.refine_base_url,
        activeSessions=session_manager.active_count,
    )


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("pt"),
    sessionId: str | None = Form(None),
    diarize: bool = Form(False),
):
    """Transcrição por lote — envia um arquivo WAV e recebe a transcrição completa.

    Se `diarize=true`, executa diarização (pyannote) para identificar
    diferentes locutores. Requer HF_TOKEN configurado no .env.
    """
    sid = sessionId or uuid.uuid4().hex[:12]

    # Salva em temp
    audio_bytes = await audio.read()
    tmp_path = _save_upload_to_temp(audio_bytes)
    try:
        # Estima duração pelo tamanho do arquivo
        file_size = os.path.getsize(tmp_path)
        duration_sec = round((file_size - 44) / 32000, 2) if file_size > 44 else 0

        # Transcreve via dispatcher (fila + round-robin entre GPUs)
        segments = await dispatcher.dispatch(tmp_path, language)

        # Diarização opcional
        if diarize:
            try:
                logger.info("Executando diarização...")
                from src.diarizer import assign_speakers, diarize as run_diarize
                diarization = run_diarize(tmp_path)
                segments = assign_speakers(segments, diarization)
            except Exception as exc:
                logger.warning("Diarização falhou (continuando sem): %s", exc)

        result = _build_transcription_result(sid, segments, duration_sec, language)
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/stream/{session_id}")
async def stream_action(
    session_id: str,
    action: StreamAction,
):
    """Gerencia sessões de streaming.

    - action=start: inicia uma nova sessão
    - action=stop: finaliza e transcreve a sessão
    """
    if action.action == "start":
        session = session_manager.create(
            session_id=session_id,
            language=action.language,
            channels=action.channels,
            diarize=action.diarize,
        )
        return {"ok": True, "sessionId": session.id}

    elif action.action == "stop":
        session = session_manager.get(session_id)
        if not session:
            raise HTTPException(404, "Sessão não encontrada")

        session.close()
        duration = session.duration_sec

        # Transcreve cada canal
        all_segments = []
        for channel in session.channels:
            wav_data = session.get_audio(channel)
            if wav_data is None:
                continue
            tmp_path = _save_upload_to_temp(wav_data)
            try:
                segments = await dispatcher.dispatch(tmp_path, session.language)

                # Mic → "Eu"; System → diarização ou "Sistema"
                if channel == "mic":
                    for seg in segments:
                        seg["speaker"] = "Eu"
                elif channel == "system" and session.diarize:
                    try:
                        logger.info("Diarizando áudio acumulado do sistema...")
                        from src.diarizer import assign_speakers, diarize as run_diarize
                        # Usa cache de diarização parcial se disponível,
                        # senão roda do zero no áudio completo
                        diarization = session._cached_diarization.get("system")
                        if diarization is None:
                            diarization = run_diarize(tmp_path)
                        segments = assign_speakers(segments, diarization)
                    except Exception as exc:
                        logger.warning("Diarização final falhou: %s", exc)
                        for seg in segments:
                            if not seg.get("speaker") or seg["speaker"] == "Locutor":
                                seg["speaker"] = "Sistema"
                else:
                    for seg in segments:
                        if not seg.get("speaker") or seg["speaker"] == "Locutor":
                            seg["speaker"] = "Sistema"

                all_segments.extend(segments)
            finally:
                os.unlink(tmp_path)

        result = _build_transcription_result(
            session_id, all_segments, duration, session.language
        )

        session_manager.remove(session_id)
        return result

    raise HTTPException(400, "action deve ser 'start' ou 'stop'")


@app.post("/stream/{session_id}/audio")
async def stream_audio(session_id: str, chunk: AudioChunk):
    """Recebe um chunk de áudio para uma sessão de streaming."""
    session = session_manager.get(session_id)
    if not session or session.closed:
        raise HTTPException(404, "Sessão não encontrada ou já finalizada")

    await session.add_audio(chunk.channel, chunk.seq, chunk.data)
    return {"ok": True, "seq": chunk.seq}


@app.get("/stream/{session_id}/events")
async def stream_events(session_id: str, request: Request):
    """SSE endpoint: escuta eventos parciais da transcrição.

    Envia resultados parciais a cada ~3s enquanto o áudio está sendo
    acumulado. Quando a sessão é finalizada, envia o resultado final.
    """
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Sessão não encontrada")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_partial: dict[str, list[dict]] = {}
        try:
            while not session.closed:
                if await request.is_disconnected():
                    break

                # Para cada canal, tenta transcrever ácido novo
                for channel in session.channels:
                    try:
                        segments = await session.flush_to_partial(channel, dispatcher)
                    except Exception:
                        segments = None

                    if segments:
                        # Filtra só segmentos novos (não repetidos)
                        prev = last_partial.get(channel, [])
                        new_segs = [
                            s for s in segments
                            if s not in prev
                        ]
                        if new_segs:
                            last_partial[channel] = segments
                            for seg in new_segs:
                                yield {
                                    "event": "partial",
                                    "data": PartialResult(
                                        channel=channel,
                                        speaker=seg.get("speaker", "Locutor"),
                                        text=seg.get("text", ""),
                                        tStart=seg.get("tStart", 0.0),
                                        tEnd=seg.get("tEnd", 0.0),
                                        isFinal=False,
                                    ).model_dump_json(),
                                }

                await asyncio.sleep(3)

            # Sessão fechada — envia heartbeat para o cliente saber
            yield {"event": "heartbeat", "data": "closed"}

        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


@app.websocket("/speech/stream")
async def speech_stream(websocket: WebSocket):
    """Streaming via WebSocket com mensagens em formato estilo Azure STT.

    Protocolo (sem SDK — qualquer app HTTP/WS pode implementar):
      C→S texto: {"type":"config","language":"pt","locale":"pt-BR",
                  "diarize":false,"maxSpeakers":4,
                  "interimIntervalMs":1000,"words":true}
      C→S binário: PCM Int16LE 16kHz mono (ou texto {"type":"audio","data":"base64..."})
      C→S texto: {"type":"end"} — finaliza e dispara a transcrição completa
      S→C texto: {"path":"turn.start"}
      S→C texto: {"path":"speech.hypothesis","Text":...,"Offset":...,"Duration":...}
                 (interim, mutável, sem locutor — como no Azure)
      S→C texto: {"path":"speech.phrase","RecognitionStatus":"Success|NoMatch",
                  "DisplayText":...,"Offset":...,"Duration":...,
                  "Speaker":1,"Locale":...,"Words":[...]}
                 (`Speaker` int só com diarização; `Words` se `words:true`)
      S→C texto: {"path":"turn.end"} — em seguida a conexão é fechada (1000)
    """
    await websocket.accept()

    # ── 1) Config (primeira mensagem precisa ser texto JSON) ──
    try:
        raw_cfg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        cfg = StreamConfig.model_validate_json(raw_cfg)
        if cfg.type != "config":
            raise ValueError("primeira mensagem deve ter type=config")
    except (ValueError, KeyError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        try:
            await websocket.send_json(error_msg("InvalidParameterValue", str(exc)))
            await websocket.close(code=1008)
        except Exception:
            pass
        return
    except WebSocketDisconnect:
        return

    session = session_manager.create(
        session_id=uuid.uuid4().hex[:12],
        language=cfg.language,
        channels=["audio"],
        diarize=False,  # interim nunca diariza (como no Azure); diarização só no final
    )
    seq = 0
    await websocket.send_json({"path": "turn.start"})

    # ── 2) Loop de interim (parciais a cada interimIntervalMs) ──
    async def _interim_loop() -> None:
        interval = cfg.interimIntervalMs / 1000.0
        while True:
            await asyncio.sleep(interval)
            try:
                pos_before = session.transcribed_pos("audio")
                segments = await session.flush_to_partial("audio", dispatcher)
            except Exception:
                continue
            if not segments:
                continue
            pos_after = session.transcribed_pos("audio")
            text = " ".join(s.get("text", "") for s in segments).strip()
            if not text:
                continue
            try:
                await websocket.send_json(hypothesis_msg(
                    text,
                    bytes_to_ticks(pos_before),
                    bytes_to_ticks(pos_after - pos_before),
                ))
            except Exception:
                break

    interim_task = asyncio.create_task(_interim_loop())

    # ── 3) Recepção de áudio até {"type":"end"} ──
    ended = False
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                data: bytes = message["bytes"]
                if data:
                    await session.add_pcm("audio", seq, data)
                    seq += 1
            elif message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(error_msg(
                        "InvalidRequestBodyFormat", "mensagem texto deve ser JSON"))
                    continue
                mtype = payload.get("type")
                if mtype == "audio" and payload.get("data"):
                    try:
                        pcm = base64.b64decode(payload["data"])
                    except Exception:
                        await websocket.send_json(error_msg(
                            "InvalidParameterValue", "campo data não é base64 válido"))
                        continue
                    if pcm:
                        await session.add_pcm("audio", seq, pcm)
                        seq += 1
                elif mtype == "end":
                    ended = True
                    break
                else:
                    await websocket.send_json(error_msg(
                        "InvalidParameterValue",
                        "mensagem texto deve ser {type:audio,data} ou {type:end}"))
    except WebSocketDisconnect:
        pass
    finally:
        interim_task.cancel()
        try:
            await interim_task
        except asyncio.CancelledError:
            pass

    if not ended:
        # Cliente desconectou sem "end" — só limpa a sessão.
        session_manager.remove(session.id)
        return

    # ── 4) Transcrição final do áudio acumulado ──
    try:
        wav_data = session.get_audio("audio")
        if wav_data is None or len(wav_data) <= 44:
            await websocket.send_json(phrase_msg("NoMatch", "", 0, 0, cfg.locale))
        else:
            tmp_path = _save_upload_to_temp(wav_data)
            try:
                segments = await dispatcher.dispatch(
                    tmp_path, session.language, word_timestamps=cfg.words)
            finally:
                os.unlink(tmp_path)

            mapper = None
            if cfg.diarize and segments:
                try:
                    from src.diarizer import assign_speakers, diarize as run_diarize
                    tmp_dz = _save_upload_to_temp(wav_data)
                    try:
                        diarization = run_diarize(tmp_dz)
                    finally:
                        os.unlink(tmp_dz)
                    segments = assign_speakers(segments, diarization)
                    mapper = SpeakerMapper(cfg.maxSpeakers)
                except Exception as exc:
                    logger.warning("Diarização final (WS) falhou: %s", exc)
                    mapper = None

            phrases = build_phrases(segments, cfg.locale, mapper, cfg.words)
            if not phrases:
                await websocket.send_json(phrase_msg("NoMatch", "", 0, 0, cfg.locale))
            for phrase in phrases:
                await websocket.send_json(phrase)

        await websocket.send_json({"path": "turn.end"})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("Erro na finalização do WS: %s", exc)
        try:
            await websocket.send_json(error_msg("InternalServerError", str(exc)))
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        session_manager.remove(session.id)


@app.post("/refine", response_model=RefinedResult)
async def refine(req: RefineRequest):
    """Refina uma transcrição usando LLM (OpenAI-compatible)."""
    segments = [s.model_dump() for s in req.transcription.segments]

    model = req.model or settings.refine_model
    refined_segments = await refiner.refine(
        segments=segments,
        language=req.transcription.language,
        model=model,
        prompt=req.prompt,
    )

    result = _build_transcription_result(
        session_id=req.transcription.sessionId,
        segments=refined_segments,
        duration_sec=req.transcription.durationSec,
        language=req.transcription.language,
    )

    return RefinedResult(
        refined=result,
        model=model,
        tokensUsed=0,  # TODO: expor do refiner
    )


# ── Main ──

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
