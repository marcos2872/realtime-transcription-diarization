"""WebSocket transport: one connection = one live transcription stream.

Message discipline:
- binary frames carry PCM16 mono audio at the negotiated sample rate; odd
  trailing bytes are carried over into the next frame (never dropped).
- text frames must be JSON objects: ``{"type": "end"}`` (flush + close),
  ``{"type": "ping"}`` (answers ``{"type": "pong"}``). Anything else yields a
  non-fatal ``invalid_message`` error and the connection stays alive.
- diarization failures are non-fatal (the stream keeps transcribing);
  transcription failures are fatal (stream closed).
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress

from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from transcript.adapters.api.schemas import (
    ClosedEvent,
    ErrorEvent,
    PartialEvent,
    PongEvent,
    ReadyEvent,
    utterance_event,
)
from transcript.application.errors import DiarizationFailed, UseCaseError

logger = logging.getLogger(__name__)

_WS_OK = 1000
_WS_CANNOT_OPEN = 4009
_WS_INTERNAL_ERROR = 1011


class _ClientGone(Exception):
    """The client disconnected mid-conversation; stop sending, still clean up."""


async def handle_stream(websocket: WebSocket, stream_id: str, partials: bool | None) -> None:
    application = websocket.app.state.application
    settings = application.settings
    emit_partials = settings.emit_partials if partials is None else partials

    await websocket.accept()
    try:
        opened = await application.open_stream.execute(stream_id)
    except UseCaseError as exc:
        await _send(websocket, ErrorEvent(code=exc.code, message=exc.message, fatal=True))
        await _close(websocket, _WS_CANNOT_OPEN)
        return

    await _send(
        websocket,
        ReadyEvent(
            stream_id=stream_id,
            language=settings.language,
            sample_rate=opened.sample_rate,
            max_streams=opened.max_streams,
            partials=emit_partials,
        ),
    )

    carry = b""
    last_partial: str | None = None
    client_gone = False
    fatal = False
    end_requested = False
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                client_gone = True
                break
            if message.get("bytes") is not None:
                data = carry + message["bytes"]
                carry = data[-1:] if len(data) % 2 else b""
                pcm = data[:-1] if len(data) % 2 else data
                if not pcm:
                    continue
                try:
                    outcome = await application.process_chunk.execute(stream_id, pcm)
                except DiarizationFailed as exc:
                    await _send(
                        websocket, ErrorEvent(code=exc.code, message=exc.message, fatal=False)
                    )
                    continue
                except UseCaseError as exc:
                    await _send(
                        websocket, ErrorEvent(code=exc.code, message=exc.message, fatal=True)
                    )
                    fatal = True
                    break
                if (
                    emit_partials
                    and outcome.partial_text is not None
                    and outcome.partial_text != last_partial
                ):
                    last_partial = outcome.partial_text
                    await _send(
                        websocket, PartialEvent(stream_id=stream_id, text=outcome.partial_text)
                    )
                for utterance in outcome.utterances:
                    await _send(websocket, utterance_event(stream_id, utterance))
            elif message.get("text") is not None:
                control = _parse_control(message["text"])
                if control is None:
                    await _send(
                        websocket,
                        ErrorEvent(
                            code="invalid_message",
                            message='expected a JSON object like {"type": "end"}',
                            fatal=False,
                        ),
                    )
                    continue
                message_type = control.get("type")
                if message_type == "end":
                    end_requested = True
                    break
                if message_type == "ping":
                    await _send(websocket, PongEvent())
                    continue
                await _send(
                    websocket,
                    ErrorEvent(
                        code="invalid_message",
                        message=f"unknown message type {message_type!r}",
                        fatal=False,
                    ),
                )
    except (WebSocketDisconnect, _ClientGone):
        client_gone = True

    if client_gone or fatal:
        await _cleanup(application, stream_id)  # flush + free the slot, report nowhere
        if fatal and not client_gone:
            await _close(websocket, _WS_INTERNAL_ERROR)
        return

    # Graceful end (client asked, or the loop broke without error): flush, emit,
    # then tell the client the stream is closed.
    try:
        closed = await application.close_stream.execute(stream_id)
        for utterance in closed.utterances:
            await _send(websocket, utterance_event(stream_id, utterance))
        await _send(websocket, ClosedEvent(stream_id=stream_id))
        if end_requested:
            await _close(websocket, _WS_OK)
    except UseCaseError as exc:
        await _send(websocket, ErrorEvent(code=exc.code, message=exc.message, fatal=True))
        await _close(websocket, _WS_INTERNAL_ERROR)
    except (WebSocketDisconnect, _ClientGone):
        await _cleanup(application, stream_id)


def _parse_control(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def _send(websocket: WebSocket, event: BaseModel) -> None:
    try:
        await websocket.send_json(event.model_dump(mode="json"))
    except (RuntimeError, WebSocketDisconnect) as exc:
        raise _ClientGone from exc


async def _close(websocket: WebSocket, code: int) -> None:
    if websocket.client_state is WebSocketState.CONNECTED:
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=code)


async def _cleanup(application, stream_id: str) -> None:
    """Free the stream slot no matter what; this must never raise."""
    try:
        await application.close_stream.execute(stream_id)
    except UseCaseError as exc:
        logger.info("stream %s already gone during cleanup: %s", stream_id, exc.code)
    except Exception:
        logger.exception("unexpected failure closing stream %s during cleanup", stream_id)
