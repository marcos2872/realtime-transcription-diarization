"""WebSocket ``/speech/stream`` — streaming estilo Azure STT.

Protocolo (sem SDK — qualquer app HTTP/WS implementa):
  C→S texto: ``{"type":"config","language":"pt","locale":"pt-BR",
              "diarize":false,"maxSpeakers":4,
              "interimIntervalMs":1000,"words":true}``
  C→S binário: PCM Int16LE 16kHz mono (ou texto
              ``{"type":"audio","data":"base64..."}``)
  C→S texto: ``{"type":"end"}`` — finaliza e transcreve o acumulado
  S→C texto: ``{"path":"turn.start"}``
  S→C texto: ``{"path":"speech.hypothesis",...}`` (interim mutável,
              sem locutor — como no Azure)
  S→C texto: ``{"path":"speech.phrase",...}`` (uma por segmento;
              ``Speaker`` int só com diarização)
  S→C texto: ``{"path":"turn.end"}`` — conexão fecha (1000)

Não aparece no OpenAPI (limitação do padrão) — documentado aqui e
em ``docs/usage.md#websocket-estilo-azure``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.ws.azure_protocol import (
    SpeakerMapper,
    StreamConfig,
    build_phrases,
    bytes_to_ticks,
    error_msg,
    hypothesis_msg,
    phrase_msg,
)
from src.infrastructure.audio.wav import save_upload_to_temp, unlink_quietly

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/speech/stream")
async def speech_stream(websocket: WebSocket):
    """Streaming via WebSocket com mensagens em formato estilo Azure STT."""
    await websocket.accept()
    dispatcher = websocket.app.state.dispatcher
    sessions = websocket.app.state.sessions

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

    session = sessions.create(
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
        sessions.remove(session.id)
        return

    # ── 4) Transcrição final do áudio acumulado ──
    try:
        wav_data = session.get_audio("audio")
        if wav_data is None or len(wav_data) <= 44:
            await websocket.send_json(phrase_msg("NoMatch", "", 0, 0, cfg.locale))
        else:
            tmp_path = save_upload_to_temp(wav_data)
            try:
                segments = await dispatcher.dispatch(
                    tmp_path, session.language, word_timestamps=cfg.words)
            finally:
                unlink_quietly(tmp_path)

            mapper = None
            if cfg.diarize and segments:
                try:
                    from src.infrastructure.diarization.pyannote import (
                        assign_speakers,
                        diarize as run_diarize,
                    )
                    tmp_dz = save_upload_to_temp(wav_data)
                    try:
                        diarization = run_diarize(tmp_dz, max_speakers=cfg.maxSpeakers)
                    finally:
                        unlink_quietly(tmp_dz)
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
        sessions.remove(session.id)
