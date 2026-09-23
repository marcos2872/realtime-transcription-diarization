from pydantic import BaseModel
from typing import Literal


# ── Transcrição ──

class Segment(BaseModel):
    speaker: str
    text: str
    tStart: float
    tEnd: float


class TranscriptionResult(BaseModel):
    sessionId: str
    segments: list[Segment]
    participants: list[str]
    durationSec: float
    language: str


# ── Streaming ──

class StreamAction(BaseModel):
    sessionId: str
    action: Literal["start", "stop"]
    language: str = "pt"
    channels: list[str] = ["mic", "system"]
    diarize: bool = True


class AudioChunk(BaseModel):
    sessionId: str
    channel: Literal["mic", "system"]
    seq: int
    data: str  # base64 PCM Int16LE 16kHz


class PartialResult(BaseModel):
    channel: Literal["mic", "system"]
    speaker: str
    text: str
    tStart: float
    tEnd: float
    isFinal: bool = False


class StreamEvent(BaseModel):
    event: Literal["partial", "final", "error"]
    data: PartialResult | TranscriptionResult


# ── Refinamento ──

class RefineRequest(BaseModel):
    transcription: TranscriptionResult
    model: str | None = None
    prompt: str | None = None


class RefinedResult(BaseModel):
    refined: TranscriptionResult
    model: str
    tokensUsed: int


# ── Health ──

class HealthResponse(BaseModel):
    status: str
    gpus: list[str]
    whisperLoaded: bool
    refineEndpoint: str
    activeSessions: int
