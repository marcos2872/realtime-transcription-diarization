"""Wire-format events of the WebSocket API (the API contract, in code)."""

from __future__ import annotations

from pydantic import BaseModel

from transcript.domain.entities import Utterance


class ReadyEvent(BaseModel):
    type: str = "ready"
    stream_id: str
    language: str
    sample_rate: int
    max_streams: int
    partials: bool


class PartialEvent(BaseModel):
    type: str = "partial"
    stream_id: str
    text: str


class UtteranceEvent(BaseModel):
    type: str = "utterance"
    stream_id: str
    utterance_id: int
    start: float
    end: float
    speaker: str
    text: str


class ErrorEvent(BaseModel):
    type: str = "error"
    code: str
    message: str
    fatal: bool


class ClosedEvent(BaseModel):
    type: str = "closed"
    stream_id: str


class PongEvent(BaseModel):
    type: str = "pong"


def utterance_event(stream_id: str, utterance: Utterance) -> UtteranceEvent:
    speaker = utterance.speaker.label if utterance.speaker is not None else "UNKNOWN"
    return UtteranceEvent(
        stream_id=stream_id,
        utterance_id=utterance.id,
        start=utterance.timestamp.start,
        end=utterance.timestamp.end,
        speaker=speaker,
        text=utterance.text,
    )
