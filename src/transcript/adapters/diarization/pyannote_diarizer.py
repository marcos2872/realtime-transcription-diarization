"""pyannote speaker diarization adapter (community-1 pipeline).

Runs the whole-file pipeline on an in-memory ``{"waveform", "sample_rate"}``
dict (the documented offline usage), shifting the returned turns by ``offset``
so they line up with the transcriber's absolute timestamps. Heavy work runs on
a worker thread behind a lock — pyannote pipelines are not thread-safe.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any

import numpy as np

from transcript.application.errors import DiarizationFailed
from transcript.domain.value_objects import Speaker, SpeakerTurn, Timestamp

logger = logging.getLogger(__name__)

_SPEAKER_INDEX = re.compile(r"SPEAKER_(\d+)", re.IGNORECASE)


def parse_speaker(label: str) -> Speaker:
    """Map a pipeline speaker label to a domain Speaker (``SPEAKER_07`` -> numbered)."""
    match = _SPEAKER_INDEX.fullmatch((label or "").strip())
    if match:
        return Speaker.numbered(int(match.group(1)))
    cleaned = (label or "").strip()
    return Speaker(label=cleaned) if cleaned else Speaker.UNKNOWN


class PyannoteDiarizer:
    def __init__(self, *, pipeline_name: str, token: str | None, device: str = "auto") -> None:
        self._pipeline_name = pipeline_name
        self._token = token
        self._device_pref = device
        self._pipeline: Any = None
        self._resolved_device = device
        self._lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------

    async def load(self) -> None:
        await asyncio.to_thread(self._load_blocking)

    async def unload(self) -> None:
        await asyncio.to_thread(self._unload_blocking)

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def _load_blocking(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed — run `uv sync --extra diarization` "
                "(or use the Docker image), or set TRANSCRIPT_DIARIZATION_PROVIDER=fake"
            ) from exc
        if not self._token:
            raise RuntimeError(
                f"{self._pipeline_name} is gated: accept the conditions on Hugging Face "
                "and set TRANSCRIPT_HF_TOKEN"
            )

        device = self._device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TRANSCRIPT_DEVICE=cuda but no CUDA device is available")

        logger.info("loading %s on %s", self._pipeline_name, device)
        with self._lock:
            pipeline = Pipeline.from_pretrained(self._pipeline_name, token=self._token)
            if device == "cuda":
                pipeline.to(torch.device("cuda"))  # like the official README
            self._pipeline = pipeline
            self._resolved_device = device

    def _unload_blocking(self) -> None:
        with self._lock:
            self._pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- Diarizer port --------------------------------------------------

    async def diarize(
        self,
        *,
        stream_id: str,
        pcm: bytes,
        sample_rate: int,
        offset: float,
    ) -> list[SpeakerTurn]:
        if self._pipeline is None:
            raise DiarizationFailed("diarization pipeline is not loaded yet")
        if not pcm:
            return []
        try:
            return await asyncio.to_thread(self._diarize_blocking, pcm, sample_rate, offset)
        except DiarizationFailed:
            raise
        except Exception as exc:
            raise DiarizationFailed(str(exc)) from exc

    def _diarize_blocking(self, pcm: bytes, sample_rate: int, offset: float) -> list[SpeakerTurn]:
        import torch

        with self._lock:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            waveform = torch.from_numpy(samples).unsqueeze(0)  # [channel, samples]
            output = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
            annotation = getattr(output, "speaker_diarization", output)

            turns: list[SpeakerTurn] = []
            for item in annotation:
                if len(item) == 2:
                    segment, label = item
                elif len(item) == 3:
                    segment, _track, label = item
                else:  # fail loud: silently misreading turns would corrupt attribution
                    raise DiarizationFailed(f"unexpected diarization item: {item!r}")
                start = float(segment.start) + offset
                end = float(segment.end) + offset
                if end <= start:
                    continue
                turns.append(
                    SpeakerTurn(
                        timestamp=Timestamp(start=start, end=end),
                        speaker=parse_speaker(str(label)),
                    )
                )
            return turns
