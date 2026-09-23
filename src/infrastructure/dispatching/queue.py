import asyncio
import concurrent.futures
import logging

from src.config import settings
from src.infrastructure.asr.faster_whisper import Transcriber

logger = logging.getLogger(__name__)


def _available_devices(requested: list[str]) -> list[str]:
    """Retorna apenas GPUs disponíveis. Se nenhuma GPU, fallback para cpu."""
    import torch
    available = [d for d in requested if d.startswith("cuda")]
    if available:
        # Filtra apenas GPUs que realmente existem
        num_gpus = torch.cuda.device_count()
        real_devices = []
        for d in available:
            idx = int(d.split(":")[1]) if ":" in d else 0
            if idx < num_gpus:
                real_devices.append(d)
            else:
                logger.warning("GPU %s não encontrada (total=%d)", d, num_gpus)
        if real_devices:
            return real_devices

    logger.warning("Nenhuma GPU disponível — usando CPU")
    return ["cpu"]

# Tipo para job da fila: (audio_path, language, future)
JobResult = list[dict]


class TranscriptionJob:
    def __init__(self, audio_path: str, language: str, word_timestamps: bool = False):
        self.audio_path = audio_path
        self.language = language
        self.word_timestamps = word_timestamps
        self.future: asyncio.Future[JobResult] = asyncio.get_event_loop().create_future()


def _transcribe_job(transcriber: Transcriber, job: TranscriptionJob) -> JobResult:
    """Executa a transcrição de um job.

    Roda dentro do executor DEDICADO do transcriber (mesma thread do
    load), preservando o contexto CUDA.
    """
    return transcriber.transcribe(job.audio_path, job.language, job.word_timestamps)


class Dispatcher:
    """Distribui requisições de transcrição entre as GPUs via round-robin.

    Cada GPU roda uma instância residente do Whisper.
    As requisições entram numa fila e são processadas uma por vez
    por cada worker (um worker por GPU).
    """

    def __init__(self):
        self._transcribers: list[Transcriber] = []
        self._queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._next_gpu = 0
        self._running = False

    async def start(self):
        """Carrega Whisper em cada GPU e inicia os workers."""
        if self._running:
            return
        self._running = True

        devices = _available_devices(settings.whisper_gpus_list)

        for device in devices:
            device = device.strip()
            if not device:
                continue
            transcriber = Transcriber(device)
            # Carrega o modelo no executor DEDICADO da GPU (mesma thread do worker,
            # necessário para consistência do contexto CUDA)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(transcriber._executor, transcriber.load)
            self._transcribers.append(transcriber)

            # Inicia um worker por GPU
            worker = asyncio.create_task(self._worker_loop(transcriber))
            self._workers.append(worker)
            logger.info("Worker iniciado para %s", device)

        logger.info(
            "Dispatcher pronto: %d GPU(s), %d worker(s)",
            len(self._transcribers), len(self._workers),
        )

    async def dispatch(
        self, audio_path: str, language: str = "pt", word_timestamps: bool = False
    ) -> JobResult:
        """Envia um arquivo de áudio para transcrição via fila.

        Retorna quando a transcrição estiver pronta.
        """
        job = TranscriptionJob(audio_path, language, word_timestamps)
        await self._queue.put(job)
        return await job.future

    async def _worker_loop(self, transcriber: Transcriber):
        """Loop do worker: pega jobs da fila e transcreve."""
        while self._running:
            try:
                job = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                logger.info(
                    "Transcrevendo %s em %s ...",
                    job.audio_path, transcriber.device_str,
                )
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    transcriber._executor,
                    _transcribe_job,
                    transcriber,
                    job,
                )
                job.future.set_result(result)
                logger.info(
                    "Transcrição concluída: %s (%d segmentos) em %s",
                    job.audio_path, len(result), transcriber.device_str,
                )
            except Exception as e:
                logger.error("Erro na transcrição %s: %s", job.audio_path, e)
                job.future.set_exception(e)
            finally:
                self._queue.task_done()

    async def stop(self):
        """Para todos os workers e descarrega modelos."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        for transcriber in self._transcribers:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(transcriber._executor, transcriber.unload)
        self._transcribers.clear()
        logger.info("Dispatcher parou")


# Instância global
dispatcher = Dispatcher()
