"""Ports — interfaces que o domínio exige do mundo externo.

Adapters em ``src/infrastructure`` implementam estes protocolos.
Use cases dependem só daqui (Dependency Inversion).
"""

from __future__ import annotations

from typing import Protocol


class TranscriberPort(Protocol):
    """Transcreve um arquivo WAV e devolve segmentos brutos."""

    async def dispatch(
        self,
        audio_path: str,
        language: str = "pt",
        word_timestamps: bool = False,
    ) -> list[dict]:
        ...


class DiarizerPort(Protocol):
    """Diarização opcional (pyannote). Falha nunca deve gerar 500."""

    def diarize(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[dict]:
        ...

    def assign_speakers(
        self,
        transcript_segments: list[dict],
        diarization: list[dict],
    ) -> list[dict]:
        ...


class RefinerPort(Protocol):
    """Refinamento via LLM. Falha retorna o original (fails open)."""

    async def refine(
        self,
        segments: list[dict],
        language: str = "pt",
        model: str | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        ...


class SessionStore(Protocol):
    """Sessões de streaming ativas (criar/obter/remover/contar)."""

    def create(
        self,
        session_id: str | None = None,
        language: str = "pt",
        channels: list[str] | None = None,
        diarize: bool = False,
    ) -> object:
        ...

    def get(self, session_id: str) -> object | None:
        ...

    def remove(self, session_id: str) -> None:
        ...

    @property
    def active_count(self) -> int:
        ...


class AudioTempFiles(Protocol):
    """Arquivos temporários para entregar áudio aos adapters.

    Implementado por ``infrastructure.audio.wav.WavTempFiles`` e
    injetado nos use cases (os use cases nunca importam
    infraestrutura diretamente).
    """

    def save(self, data: bytes) -> str:
        """Salva bytes em temp e devolve o caminho."""
        ...

    def remove(self, path: str) -> None:
        """Remove o temp, ignorando erros."""
        ...
