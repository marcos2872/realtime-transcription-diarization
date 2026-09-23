"""Áudio WAV — construção, leitura e arquivos temporários.

Centraliza o que antes estava espalhado em ``main.py`` e
``session.py``: o formato canônico é PCM Int16LE 16kHz mono.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import tempfile

from fastapi import UploadFile

from src.domain.value_objects.audio_format import (
    BITS_PER_SAMPLE,
    BYTES_PER_SECOND,
    CHANNELS,
    SAMPLE_RATE,
    WAV_HEADER_SIZE,
)

logger = logging.getLogger(__name__)


def build_wav_from_pcm(pcm_data: bytes) -> bytes:
    """Constrói um WAV completo a partir de PCM bruto."""
    buf = io.BytesIO()
    data_size = len(pcm_data)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", CHANNELS))  # mono
    buf.write(struct.pack("<I", SAMPLE_RATE))
    buf.write(struct.pack("<I", BYTES_PER_SECOND))
    buf.write(struct.pack("<H", CHANNELS * BITS_PER_SAMPLE // 8))
    buf.write(struct.pack("<H", BITS_PER_SAMPLE))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)
    return buf.getvalue()


def write_wav_header(buf: io.BytesIO) -> None:
    """Escreve header WAV placeholder (atualizado ao finalizar)."""
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 0))  # tamanho total (placeholder)
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", CHANNELS))  # mono
    buf.write(struct.pack("<I", SAMPLE_RATE))
    buf.write(struct.pack("<I", BYTES_PER_SECOND))
    buf.write(struct.pack("<H", CHANNELS * BITS_PER_SAMPLE // 8))
    buf.write(struct.pack("<H", BITS_PER_SAMPLE))
    buf.write(b"data")
    buf.write(struct.pack("<I", 0))  # data size (placeholder)


def finalize_wav(buf: io.BytesIO) -> bytes:
    """Atualiza os headers WAV e retorna o áudio completo."""
    data_size = buf.tell() - WAV_HEADER_SIZE
    buf.seek(4)
    buf.write(struct.pack("<I", 36 + data_size))
    buf.seek(40)
    buf.write(struct.pack("<I", data_size))
    return buf.getvalue()


def save_bytes_to_temp(data: bytes, suffix: str = ".wav") -> str:
    """Salva bytes em arquivo temporário e retorna o caminho."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        unlink_quietly(tmp.name)
        raise

def save_upload_to_temp(audio: UploadFile | bytes, suffix: str = ".wav") -> str:
    """Salva um upload (FastAPI ou bytes) em arquivo temporário."""
    if isinstance(audio, bytes):
        return save_bytes_to_temp(audio, suffix)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(audio.file.read())
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        unlink_quietly(tmp.name)
        raise


def unlink_quietly(path: str) -> None:
    """Remove arquivo temporário, ignorando erros."""
    try:
        os.unlink(path)
    except OSError:
        pass


class WavTempFiles:
    """Implementação ``AudioTempFiles`` sobre arquivos ``.wav``."""

    def __init__(self, suffix: str = ".wav"):
        self.suffix = suffix

    def save(self, data: bytes) -> str:
        return save_bytes_to_temp(data, self.suffix)

    def remove(self, path: str) -> None:
        unlink_quietly(path)
