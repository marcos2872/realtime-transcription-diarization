"""Protocolo WebSocket de streaming com formato estilo Azure STT.

Permite que qualquer aplicação que fale com a API do Azure Speech
diretamente (sem SDK) use este servidor: o cliente abre um WebSocket,
envia PCM 16kHz mono e recebe mensagens JSON com os mesmos nomes de
campo do Azure (`speech.hypothesis` / `speech.phrase`, `Offset` /
`Duration` em ticks de 100ns, `RecognitionStatus`, `Speaker`).

Formato resumido (ver README § WebSocket):

    C→S texto: {"type": "config", "language": "pt", "locale": "pt-BR",
                "diarize": true, "maxSpeakers": 4,
                "interimIntervalMs": 1000, "words": true}
    C→S binário: PCM Int16LE 16kHz mono (ou texto {"type":"audio","data":"base64..."})
    C→S texto: {"type": "end"}
    S→C texto: {"path": "turn.start"}
    S→C texto: {"path": "speech.hypothesis", "Text": ..., "Offset": ..., "Duration": ...}
    S→C texto: {"path": "speech.phrase", "RecognitionStatus": "Success",
                "DisplayText": ..., "Offset": ..., "Duration": ...,
                "Speaker": 1, "Locale": "pt-BR", "Words": [...]}
    S→C texto: {"path": "turn.end"}
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 1 tick do Azure = 100ns → 10^7 ticks por segundo.
TICKS_PER_SECOND = 10_000_000

# Bytes por segundo do PCM Int16LE 16kHz mono.
BYTES_PER_SECOND = 32000


def sec_to_ticks(seconds: float) -> int:
    """Converte segundos → ticks de 100ns (formato Azure)."""
    return int(round(seconds * TICKS_PER_SECOND))


def bytes_to_ticks(num_bytes: int) -> int:
    """Converte bytes de PCM (16kHz 16-bit mono) → ticks de 100ns."""
    return sec_to_ticks(num_bytes / BYTES_PER_SECOND)


class StreamConfig(BaseModel):
    """Mensagem `config` enviada pelo cliente ao abrir o WebSocket."""

    type: str = "config"
    language: str = "pt"
    locale: str = "pt-BR"
    diarize: bool = False
    maxSpeakers: int | None = None
    interimIntervalMs: int = Field(default=1000, ge=500, le=10000)
    words: bool = True


def hypothesis_msg(text: str, offset_ticks: int, duration_ticks: int) -> dict:
    """Mensagem interim (`speech.hypothesis`) — texto parcial mutável, sem locutor."""
    return {
        "path": "speech.hypothesis",
        "Text": text,
        "Offset": offset_ticks,
        "Duration": duration_ticks,
    }


def phrase_msg(
    status: str,
    text: str,
    offset_ticks: int,
    duration_ticks: int,
    locale: str,
    speaker: int | None = None,
    words: list[dict] | None = None,
) -> dict:
    """Mensagem final (`speech.phrase`) — uma por utterance/segmento.

    `speaker` (int) só é incluído quando a diarização está ativa,
    como no Azure (`speaker` só presente com diarização habilitada).
    """
    msg: dict = {
        "path": "speech.phrase",
        "RecognitionStatus": status,
        "DisplayText": text,
        "Offset": offset_ticks,
        "Duration": duration_ticks,
        "Locale": locale,
    }
    if speaker is not None:
        msg["Speaker"] = speaker
    if words is not None:
        msg["Words"] = words
    return msg


def error_msg(code: str, message: str) -> dict:
    """Mensagem de erro (formato `{code, message}` do Azure)."""
    return {"path": "error", "code": code, "message": message}


class SpeakerMapper:
    """Mapeia labels de locutor (`Pessoa N`) → ints estilo Azure.

    O Azure retorna `speaker` como inteiro sem ordem particular; aqui
    o id segue a ordem de primeira aparição. Se `max_speakers` for
    definido e houver mais locutores, os excedentes são mesclados no
    último id (aproximação documentada — o Azure pode combinar
    locutores nesse caso).
    """

    def __init__(self, max_speakers: int | None = None):
        self._ids: dict[str, int] = {}
        self._max = max_speakers if (max_speakers and max_speakers > 0) else None

    def get(self, label: str) -> int:
        """Retorna o id int do locutor, criando se necessário."""
        if label not in self._ids:
            new_id = len(self._ids) + 1
            if self._max is not None and new_id > self._max:
                new_id = self._max
            self._ids[label] = new_id
        return self._ids[label]


def build_phrases(
    segments: list[dict],
    locale: str,
    mapper: SpeakerMapper | None,
    include_words: bool,
) -> list[dict]:
    """Converte segmentos internos → lista de mensagens `speech.phrase`."""
    phrases = []
    for seg in segments:
        t_start = float(seg.get("tStart", 0.0))
        t_end = float(seg.get("tEnd", 0.0))
        offset = sec_to_ticks(t_start)
        duration = sec_to_ticks(max(0.0, t_end - t_start))
        speaker = mapper.get(seg.get("speaker", "")) if mapper else None
        words = None
        if include_words and seg.get("words"):
            words = [
                {
                    "text": w.get("text", ""),
                    "offsetMilliseconds": int(round(float(w.get("start", 0.0)) * 1000)),
                    "durationMilliseconds": int(round(
                        max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0)))
                        * 1000
                    )),
                }
                for w in seg["words"]
            ]
        phrases.append(phrase_msg(
            "Success", seg.get("text", ""), offset, duration,
            locale, speaker, words,
        ))
    return phrases
