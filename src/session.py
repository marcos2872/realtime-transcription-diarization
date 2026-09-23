import asyncio
import logging
import os
import tempfile
import time
import uuid
import io
import wave
import struct
import base64
from typing import Callable, Awaitable, TYPE_CHECKING

from src.config import settings
from src.schemas import PartialResult, TranscriptionResult, Segment

if TYPE_CHECKING:
    from src.dispatcher import Dispatcher

logger = logging.getLogger(__name__)

OnPartial = Callable[[PartialResult], Awaitable[None]]
OnFinal = Callable[[TranscriptionResult], Awaitable[None]]

# ── Helpers para WAV parcial ──


def _build_wav_from_pcm(pcm_data: bytes) -> bytes:
    """Constrói um WAV completo a partir de PCM bruto."""
    buf = io.BytesIO()
    data_size = len(pcm_data)
    # Header WAV
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))   # PCM
    buf.write(struct.pack('<H', 1))   # mono
    buf.write(struct.pack('<I', 16000))
    buf.write(struct.pack('<I', 32000))
    buf.write(struct.pack('<H', 2))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm_data)
    return buf.getvalue()


class Session:
    """Representa uma sessão de streaming ativa."""

    def __init__(self, session_id: str, language: str, channels: list[str], diarize: bool = False):
        self.id = session_id
        self.language = language
        self.channels = channels
        self.diarize = diarize
        self.created_at = time.time()
        self._buffers: dict[str, io.BytesIO] = {}
        self._pcm_buffers: dict[str, list[bytes]] = {}  # PCM bruto acumulado
        self._last_transcribed_pos: dict[str, int] = {}  # bytes já transcritos
        self._last_diarized_pos: dict[str, int] = {}  # bytes já diarizados
        self._cached_diarization: dict[str, list[dict]] = {}  # diarização em cache
        self._seq: dict[str, int] = {}
        self._closed = False
        self._lock = asyncio.Lock()

    async def add_audio(self, channel: str, seq: int, pcm_base64: str):
        """Adiciona um chunk de áudio à sessão."""
        async with self._lock:
            if channel not in self._buffers:
                self._buffers[channel] = io.BytesIO()
                self._pcm_buffers[channel] = []
                self._last_transcribed_pos[channel] = 0
                self._seq[channel] = 0
                self._write_wav_header(self._buffers[channel])

            self._seq[channel] = max(self._seq[channel], seq)
            pcm_data = base64.b64decode(pcm_base64)
            self._buffers[channel].write(pcm_data)
            self._pcm_buffers[channel].append(pcm_data)

    async def add_pcm(self, channel: str, seq: int, pcm_data: bytes):
        """Adiciona PCM bruto (já decodificado) à sessão.

        Equivalente a `add_audio`, mas sem o passo base64 — usado pelo
        endpoint WebSocket, que recebe frames binários.
        """
        async with self._lock:
            if channel not in self._buffers:
                self._buffers[channel] = io.BytesIO()
                self._pcm_buffers[channel] = []
                self._last_transcribed_pos[channel] = 0
                self._seq[channel] = 0
                self._write_wav_header(self._buffers[channel])

            self._seq[channel] = max(self._seq[channel], seq)
            self._buffers[channel].write(pcm_data)
            self._pcm_buffers[channel].append(pcm_data)

    def transcribed_pos(self, channel: str) -> int:
        """Bytes de PCM já marcados como transcritos (para cálculo de offset)."""
        return self._last_transcribed_pos.get(channel, 0)

    def _write_wav_header(self, buf: io.BytesIO):
        """Escreve header WAV placeholder (atualizado no final)."""
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 0))  # tamanho total (placeholder)
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<I', 16))          # chunk size
        buf.write(struct.pack('<H', 1))            # PCM
        buf.write(struct.pack('<H', 1))            # mono
        buf.write(struct.pack('<I', 16000))        # sample rate
        buf.write(struct.pack('<I', 32000))        # byte rate
        buf.write(struct.pack('<H', 2))            # block align
        buf.write(struct.pack('<H', 16))           # bits per sample
        buf.write(b'data')
        buf.write(struct.pack('<I', 0))            # data size (placeholder)

    def _finalize_wav(self, buf: io.BytesIO) -> bytes:
        """Atualiza headers WAV e retorna o áudio completo."""
        data_size = buf.tell() - 44  # 44 = tamanho do header
        buf.seek(4)
        buf.write(struct.pack('<I', 36 + data_size))
        buf.seek(40)
        buf.write(struct.pack('<I', data_size))
        return buf.getvalue()

    def get_audio(self, channel: str) -> bytes | None:
        """Retorna o WAV completo de um canal."""
        buf = self._buffers.get(channel)
        if buf is None:
            return None
        return self._finalize_wav(buf)

    def get_partial_pcm(self, channel: str) -> bytes | None:
        """Retorna apenas o PCM novo (não transcrito ainda) de um canal."""
        pcm_data = b"".join(self._pcm_buffers.get(channel, []))
        pos = self._last_transcribed_pos.get(channel, 0)
        new_data = pcm_data[pos:]
        if len(new_data) < 8000:  # mínimo ~0.5s para transcrever
            return None
        return new_data

    def mark_transcribed(self, channel: str):
        """Marca todo o PCM atual como já transcrito."""
        pcm_data = b"".join(self._pcm_buffers.get(channel, []))
        self._last_transcribed_pos[channel] = len(pcm_data)

    async def flush_to_partial(
        self, channel: str, dispatcher: 'Dispatcher'
    ) -> list[dict] | None:
        """Transcreve o áudio acumulado não transcrito e retorna segmentos.

        Só transcreve se houver pelo menos 1s de áudio novo.
        Se `self.diarize=True` e channel='system', roda diarização
        no áudio acumulado completo para identificar locutores.
        Após transcrever, marca a posição como já processada.
        """
        async with self._lock:
            pcm_data = self.get_partial_pcm(channel)
            if pcm_data is None:
                return None

            pos = self._last_transcribed_pos.get(channel, 0)
            wav_bytes = _build_wav_from_pcm(pcm_data)
            self.mark_transcribed(channel)

        # 1) Transcreve o chunk novo com Whisper
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp.write(wav_bytes)
            tmp.close()
            segments = await dispatcher.dispatch(tmp.name, self.language)
        finally:
            os.unlink(tmp.name)

        # 2) Se diarize ativo, aplica diarização no áudio acumulado
        if self.diarize and channel == "system" and segments:
            try:
                await self._apply_diarization_to_segments(segments, pos)
            except Exception as exc:
                logger.warning("Diarização parcial falhou: %s", exc)

        return segments

    async def _apply_diarization_to_segments(
        self, segments: list[dict], pcm_offset: int
    ):
        """Roda diarização no áudio acumulado e atribui locutores.

        Args:
            segments: segmentos do Whisper (timestamps relativos ao chunk)
            pcm_offset: posição em bytes deste chunk no áudio acumulado
        """
        from src.diarizer import assign_speakers, diarize as run_diarize

        # Concatena o PCM acumulado completo
        full_pcm = b"".join(self._pcm_buffers.get("system", []))
        if not full_pcm:
            return

        # Cria WAV temporário com o áudio acumulado
        wav_bytes = _build_wav_from_pcm(full_pcm)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp.write(wav_bytes)
            tmp.close()
            diarization = run_diarize(tmp.name)
        finally:
            os.unlink(tmp.name)

        # Converte timestamps dos segmentos do Whisper (relativos ao chunk)
        # para a timeline do áudio acumulado completo
        offset_sec = pcm_offset / 32000.0
        for seg in segments:
            seg["tStart"] = round(seg.get("tStart", 0) + offset_sec, 2)
            seg["tEnd"] = round(seg.get("tEnd", 0) + offset_sec, 2)

        # Atribui locutores via assign_speakers
        assigned = assign_speakers(segments, diarization)
        # Substitui os segmentos originais pelos com speaker preenchido
        segments[:] = assigned

        # Cache (para uso futuro / finalização)
        self._cached_diarization["system"] = diarization

    @property
    def duration_sec(self) -> float:
        """Duração estimada em segundos (baseada no primeiro canal disponível)."""
        for channel in self.channels:
            buf = self._buffers.get(channel)
            if buf is not None:
                data_size = buf.tell() - 44
                return data_size / 32000  # 16kHz mono 16-bit
        return 0.0

    def is_expired(self, timeout_min: int = 10) -> bool:
        return time.time() - self.created_at > timeout_min * 60

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self):
        self._closed = True


class SessionManager:
    """Gerencia sessões de streaming ativas."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str | None = None, language: str = "pt",
               channels: list[str] | None = None, diarize: bool = False) -> Session:
        sid = session_id or uuid.uuid4().hex[:12]
        session = Session(sid, language, channels or ["mic", "system"], diarize=diarize)
        self._sessions[sid] = session
        logger.info("Sessão criada: %s (diarize=%s)", sid, diarize)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)
        logger.info("Sessão removida: %s", session_id)

    def cleanup_expired(self):
        """Remove sessões expiradas."""
        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired(settings.session_timeout_min)
        ]
        for sid in expired:
            self._sessions[sid].close()
            self._sessions.pop(sid, None)
        if expired:
            logger.info("Sessões expiradas removidas: %d", len(expired))

    @property
    def active_count(self) -> int:
        return len(self._sessions)


session_manager = SessionManager()
