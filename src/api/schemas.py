"""Schemas de wire (Pydantic) — formato externo estável da API.

Compatibilidade preservada: nomes camelCase (``sessionId``,
``tStart``/``tEnd``, ``durationSec``), ``sessionId`` exigido no body
**e** no path de streaming. Mudanças de wire são BREAKING CHANGE.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Transcrição ──


class Segment(BaseModel):
    """Um trecho transcrito."""

    speaker: str = Field(description="Locutor (`Eu`, `Sistema`, `Pessoa N` ou `Locutor`).",
                         examples=["Pessoa 1"])
    text: str = Field(description="Texto transcrito do trecho.", examples=["Olá, tudo bem?"])
    tStart: float = Field(description="Início do trecho em segundos.", examples=[6.5])
    tEnd: float = Field(description="Fim do trecho em segundos.", examples=[9.8])


class TranscriptionResult(BaseModel):
    """Resultado completo de uma transcrição."""

    sessionId: str = Field(description="ID da sessão/transcrição.", examples=["abc123def456"])
    segments: list[Segment] = Field(default_factory=list, description="Segmentos ordenados por tempo.")
    participants: list[str] = Field(default_factory=list, description="Locutores em ordem de aparição.")
    durationSec: float = Field(description="Duração estimada do áudio em segundos.", examples=[83.6])
    language: str = Field(description="Idioma (ISO 639-1).", examples=["pt"])


# ── Streaming ──


class StreamAction(BaseModel):
    """Ação de streaming. ``sessionId`` vai no body **e** no path."""

    sessionId: str = Field(description="ID da sessão (repetir o do path).", examples=["sess-001"])
    action: Literal["start", "stop"] = Field(description="`start` cria a sessão; `stop` finaliza e transcreve.")
    language: str = Field(default="pt", description="Idioma (ISO 639-1).", examples=["pt"])
    channels: list[str] = Field(default=["mic", "system"],
                                description="Canais a capturar (`mic`, `system`).")
    diarize: bool = Field(default=True, description="Diarizar o canal `system` (requer `HF_TOKEN`).")


class AudioChunk(BaseModel):
    """Chunk de áudio. ``sessionId`` vai no body **e** no path."""

    sessionId: str = Field(description="ID da sessão (repetir o do path).", examples=["sess-001"])
    channel: Literal["mic", "system"] = Field(description="Canal de destino do chunk.")
    seq: int = Field(description="Número de sequência do chunk.", examples=[1])
    data: str = Field(description="PCM Int16LE 16kHz mono codificado em base64.")


class PartialResult(BaseModel):
    """Resultado parcial emitido via SSE."""

    channel: Literal["mic", "system"] = Field(description="Canal de origem.")
    speaker: str = Field(description="Locutor do trecho.", examples=["Pessoa 1"])
    text: str = Field(description="Texto parcial.", examples=["..."])
    tStart: float = Field(description="Início em segundos.", examples=[0.0])
    tEnd: float = Field(description="Fim em segundos.", examples=[2.5])
    isFinal: bool = Field(default=False, description="Sempre `false` em parciais.")


class StreamEvent(BaseModel):
    """Evento de streaming (documentação do payload SSE)."""

    event: Literal["partial", "final", "error"] = Field(description="Tipo do evento.")
    data: PartialResult | TranscriptionResult = Field(description="Payload do evento.")


# ── Refinamento ──


class RefineRequest(BaseModel):
    """Pedido de refinamento via LLM."""

    transcription: TranscriptionResult = Field(description="Transcrição a refinar.")
    model: str | None = Field(default=None,
                              description="Modelo LLM (padrão: `REFINE_MODEL`).",
                              examples=["Qwen2.5-7B-Instruct"])
    prompt: str | None = Field(default=None,
                               description="Instruções extras (substituem o prompt padrão).")


class RefinedResult(BaseModel):
    """Transcrição refinada."""

    refined: TranscriptionResult = Field(description="Transcrição após refinamento.")
    model: str = Field(description="Modelo usado.", examples=["Qwen2.5-7B-Instruct"])
    tokensUsed: int = Field(description="Tokens usados (0 = não exposto pelo servidor llama.cpp).")


# ── Health ──


class HealthResponse(BaseModel):
    """Status do servidor."""

    status: str = Field(description="`ok` quando o servidor responde.", examples=["ok"])
    gpus: list[str] = Field(description="GPUs configuradas.", examples=[["cuda:0", "cuda:1"]])
    whisperLoaded: bool = Field(description="Se há transcribers carregados.")
    refineEndpoint: str = Field(description="Endpoint OpenAI-compatible do refine.",
                                examples=["http://llama-refine:8080/v1"])
    activeSessions: int = Field(description="Sessões de streaming ativas.", examples=[0])
