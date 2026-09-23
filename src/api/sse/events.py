"""Gerador SSE de parciais — extraído do antigo ``main.stream_events``.

Emite um evento ``partial`` por segmento novo a cada ciclo (~3s) e
``heartbeat/closed`` ao final. Erros de flush parcial são ignorados
(flush seguinte tenta de novo); desconexão do cliente encerra o loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import Request

from src.api.schemas import PartialResult

logger = logging.getLogger(__name__)

PARTIAL_INTERVAL_SEC = 3.0


async def partial_event_generator(session, request: Request, dispatcher) -> AsyncGenerator[dict, None]:
    """Gera eventos SSE para uma sessão até ``session.closed``."""
    last_partial: dict[str, list[dict]] = {}
    try:
        while not session.closed:
            if await request.is_disconnected():
                break
            for channel in session.channels:
                try:
                    segments = await session.flush_to_partial(channel, dispatcher)
                except Exception:
                    segments = None
                if not segments:
                    continue
                prev = last_partial.get(channel, [])
                new_segs = [s for s in segments if s not in prev]
                if not new_segs:
                    continue
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
            await asyncio.sleep(PARTIAL_INTERVAL_SEC)
        yield {"event": "heartbeat", "data": "closed"}
    except asyncio.CancelledError:
        pass
