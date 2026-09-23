import concurrent.futures
import logging
import re
import time

from faster_whisper import WhisperModel

from src.config import settings

logger = logging.getLogger(__name__)


def parse_device(device_str: str) -> tuple[str, int]:
    """Converte 'cuda:0' → ('cuda', 0). 'cpu' → ('cpu', 0)."""
    m = re.match(r"^cuda(?::(\d+))?$", device_str)
    if m:
        idx = int(m.group(1)) if m.group(1) else 0
        return "cuda", idx
    return "cpu", 0


class Transcriber:
    """Wrapper around faster-whisper para uma GPU específica.

    Cada instância mantém um executor DEDICADO de 1 thread para que todas as
    operações CUDA (load + transcribe) rodem na **mesma** thread, evitando o
    erro "CUDA failed with error invalid argument" causado pela troca de
    contexto CUDA entre threads diferentes.
    """

    def __init__(self, device_str: str):
        self.device_str = device_str
        self.device, self.device_index = parse_device(device_str)
        self._model: WhisperModel | None = None
        # Executor dedicado: 1 thread → mesmo contexto CUDA para load + transcribe
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def load(self):
        if self._model is not None:
            return
        logger.info(
            "Carregando Whisper %s em %s (index=%d) ...",
            settings.whisper_model, self.device, self.device_index,
        )
        t0 = time.time()

        compute = settings.whisper_compute
        if self.device == "cpu":
            compute = "int8"

        self._model = WhisperModel(
            settings.whisper_model,
            device=self.device,
            device_index=self.device_index,
            compute_type=compute,
            download_root="/app/models",
        )
        elapsed = time.time() - t0
        logger.info("Whisper carregado em %.1fs (%s:%d)", elapsed, self.device, self.device_index)

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self.load()
        return self._model  # type: ignore

    def transcribe(
        self, audio_path: str, language: str = "pt", word_timestamps: bool = False
    ) -> list[dict]:
        """Transcreve um arquivo WAV e retorna lista de segmentos.

        Se `word_timestamps=True`, cada segmento inclui `words` com
        `[{text, start, end}]` (usado pelo streaming WebSocket para
        preencher `Words` no formato Azure).
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=word_timestamps,
        )

        results = []
        for seg in segments:
            entry = {
                "speaker": "Locutor" if info.language != "pt" else "Locutor",
                "text": seg.text.strip(),
                "tStart": round(seg.start, 2),
                "tEnd": round(seg.end, 2),
            }
            if word_timestamps and seg.words:
                entry["words"] = [
                    {
                        "text": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                    }
                    for w in seg.words
                ]
            results.append(entry)

        return results

    def unload(self):
        """Descarrega o modelo da GPU (libera VRAM) e finaliza o executor."""
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
            logger.info("Whisper descarregado de %s:%d", self.device, self.device_index)
        self._executor.shutdown(wait=False)
