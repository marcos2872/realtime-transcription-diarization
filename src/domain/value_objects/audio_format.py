"""Formato de áudio canônico: WAV 16kHz 16-bit mono.

Por que fixo: o Whisper recebe sempre esse formato; duração é
estimada como ``(file_size - 44) / 32000`` sem parsear o header.
"""

SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SECOND = 32000  # 16000 Hz * 1 canal * 2 bytes
WAV_HEADER_SIZE = 44


def estimate_duration_sec(file_size_bytes: int) -> float:
    """Estima duração em segundos a partir do tamanho do WAV."""
    if file_size_bytes <= WAV_HEADER_SIZE:
        return 0.0
    return round((file_size_bytes - WAV_HEADER_SIZE) / BYTES_PER_SECOND, 2)
