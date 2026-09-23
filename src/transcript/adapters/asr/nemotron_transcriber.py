"""Nemotron streaming transcriber, backed by HuggingFace Transformers.

Implements the documented streaming pattern for
``nvidia/nemotron-3.5-asr-streaming-0.6b`` (``AutoModelForRNNT`` + ``generate``
with an ``input_features`` generator + ``TextIteratorStreamer``), adapted from
whole-file to live audio:

- each stream runs one ``model.generate`` call in a pump thread; the feature
  generator blocks on newly pushed audio instead of a file;
- a consumer thread drains the streamer into the session's live text;
- ``push_audio`` returns only the words grown since the previous call,
  timestamped inside ``[span_start, now - lookahead]`` (approximate);
- ``finish_stream`` signals end-of-stream (padding the tail), joins the pump
  and returns the final words.

Heavy work (``generate`` + processor calls) runs on worker threads so the
event loop never blocks while streams share the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from transcript.application.errors import TranscriptionFailed
from transcript.domain.value_objects import Timestamp, Word

logger = logging.getLogger(__name__)

# Chunk size (ms) -> Transformers right attention context. The Transformers
# port supports [3, 0, 6, 13]; there is no 160 ms operating point here
# (validated against processor.supported_num_lookahead_tokens).
LOOKAHEAD_TOKENS_BY_CHUNK_MS: dict[int, int] = {80: 0, 320: 3, 560: 6, 1120: 13}

# Suffix emitted in language auto-detect mode, appended after terminal punctuation.
# (The streamer already decodes with skip_special_tokens=True; kept as belt and braces.)
_LANGUAGE_TAG_SUFFIX = re.compile(r"\s*<[a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4})?>\s*$")

# Words attributed to a single growth step are capped to this span, so a long
# silence followed by one word does not smear that word over a minute of timeline.
MAX_WORD_SPAN_S = 8.0

# Generous but finite: finish() must not hang forever if generate stalls.
PUMP_JOIN_TIMEOUT_S = 60.0
CLOSE_JOIN_TIMEOUT_S = 5.0


# --- pure helpers (unit-tested, no model needed) ----------------------------


def right_context_frames(chunk_ms: int) -> int:
    """Map a streaming chunk size to the model's right attention context."""
    try:
        return LOOKAHEAD_TOKENS_BY_CHUNK_MS[chunk_ms]
    except KeyError:
        expected = ", ".join(str(ms) for ms in sorted(LOOKAHEAD_TOKENS_BY_CHUNK_MS))
        raise ValueError(f"unsupported chunk_ms {chunk_ms}; expected one of {expected}") from None


def strip_language_tag(text: str) -> str:
    """Remove the ``<xx-XX>`` language tag the model emits in auto-detect mode."""
    return _LANGUAGE_TAG_SUFFIX.sub("", text).rstrip()


def split_new_words(previous_text: str, current_text: str) -> list[str]:
    """Words the model produced since the last observation (longest common prefix)."""
    if not current_text:
        return []
    limit = min(len(previous_text), len(current_text))
    common = 0
    while common < limit and previous_text[common] == current_text[common]:
        common += 1
    delta = current_text[common:].strip()
    return delta.split() if delta else []


def allocate_word_spans(
    word_count: int, span_start: float, span_end: float
) -> list[tuple[float, float]]:
    """Split ``[span_start, span_end]`` evenly; degenerate spans yield zero-length words."""
    if word_count <= 0:
        return []
    width = max(0.0, span_end - span_start) / word_count
    return [(span_start + i * width, span_start + (i + 1) * width) for i in range(word_count)]


def pcm_bytes_to_float(pcm: bytes) -> np.ndarray:
    """PCM16 mono bytes -> float32 mono in [-1, 1]."""
    return (np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0).copy()


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One streaming window in absolute sample coordinates."""

    start_sample: int
    end_sample: int
    mel_frames: int
    is_first: bool
    final: bool  # tail padded with zeros (only possible at end-of-stream)


class ChunkPlanner:
    """Plans streaming windows mirroring the documented Transformers chunking.

    One larger first window, then fixed windows derived from mel progress with
    STFT left context (``start = mel_idx * hop - n_fft // 2``). All indices
    are absolute; the session translates them against its trimmed buffer.
    Window starts are strictly increasing, so everything before the last
    emitted start can be dropped.
    """

    def __init__(
        self,
        *,
        first_samples: int,
        per_chunk_samples: int,
        first_mel: int,
        per_mel: int,
        hop_length: int,
        n_fft: int,
    ) -> None:
        if first_samples <= 0 or per_chunk_samples <= 0 or per_mel <= 0:
            raise ValueError("chunk sizes must be positive")
        self._first_samples = first_samples
        self._per_chunk_samples = per_chunk_samples
        self._first_mel = first_mel
        self._per_mel = per_mel
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._first_done = False
        self._mel_idx = first_mel
        self.last_start = 0

    def plan(self, available_samples: int, *, eos: bool) -> ChunkPlan | None:
        """Next window fully covered by samples, or the padded tail at eos.

        ``available_samples`` is the ABSOLUTE total pushed (not the trimmed
        buffer length): windows are planned on the stream timeline and the
        session trims only behind the last emitted start.
        """
        if not self._first_done:
            if available_samples >= self._first_samples:
                return self._emit_first(final=False)
            if eos and available_samples > 0:
                return self._emit_first(final=True)
            return None
        start = self._mel_idx * self._hop_length - self._n_fft // 2
        end = start + self._per_chunk_samples
        if available_samples >= end:
            return self._emit_next(start, end, final=False)
        if eos and available_samples > start:
            return self._emit_next(start, end, final=True)
        return None

    def _emit_first(self, *, final: bool) -> ChunkPlan:
        self._first_done = True
        self.last_start = 0
        return ChunkPlan(
            start_sample=0,
            end_sample=self._first_samples,
            mel_frames=self._first_mel,
            is_first=True,
            final=final,
        )

    def _emit_next(self, start: int, end: int, *, final: bool) -> ChunkPlan:
        self._mel_idx += self._per_mel
        self.last_start = start
        return ChunkPlan(
            start_sample=start,
            end_sample=end,
            mel_frames=self._per_mel,
            is_first=False,
            final=final,
        )


# --- live session -----------------------------------------------------------


class _StreamSession:
    """Per-stream state: sample buffer, planner, pump/consumer threads, live text."""

    def __init__(self, stream_id: str, planner: ChunkPlanner, sample_rate: int) -> None:
        self.stream_id = stream_id
        self.planner = planner
        self.sample_rate = sample_rate
        self.condition = threading.Condition()
        self._chunks: list[np.ndarray] = []
        self._dropped = 0  # absolute index of _chunks[0][0]
        self.samples_pushed = 0
        self.live_text = ""
        self.emitted_text = ""
        self.span_start_s = 0.0
        self.eos = False
        self.abandoned = False
        self.error: BaseException | None = None
        self.streamer: Any = None
        self.pump_done = False
        self.pump_thread: threading.Thread | None = None
        self.consumer_thread: threading.Thread | None = None
        self.first_features: Any = None

    # -- buffer (call with condition held unless noted) --------------------

    @property
    def available(self) -> int:
        return self.samples_pushed - self._dropped

    def append_pcm(self, pcm: bytes) -> None:
        with self.condition:
            self._chunks.append(pcm_bytes_to_float(pcm))
            self.samples_pushed += len(pcm) // 2
            self.condition.notify_all()

    def materialize(self, start: int, end: int) -> np.ndarray:
        """Absolute [start, end) as float32, zero-padded past available audio."""
        with self.condition:
            local_start = start - self._dropped
            have = self.available - local_start
            parts: list[np.ndarray] = []
            remaining_start = max(0, local_start)
            remaining = max(0, min(end - start, have - max(0, -local_start)))
            offset = remaining_start
            for chunk in self._chunks:
                if offset >= len(chunk):
                    offset -= len(chunk)
                    continue
                take = min(len(chunk) - offset, remaining)
                if take <= 0:
                    break
                parts.append(chunk[offset : offset + take])
                remaining -= take
                offset = 0
                if remaining <= 0:
                    break
            view = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(view) < end - start:
                view = np.pad(view, (0, end - start - len(view)))
            return view

    def drop_before(self, index: int) -> None:
        with self.condition:
            cut = index - self._dropped
            if cut <= 0:
                return
            kept: list[np.ndarray] = []
            for chunk in self._chunks:
                if cut >= len(chunk):
                    cut -= len(chunk)
                    continue
                if cut > 0:
                    chunk = chunk[cut:]
                    cut = 0
                kept.append(chunk)
            dropped_now = index - self._dropped
            self._chunks = kept
            self._dropped = index
            assert dropped_now >= 0

    def set_eos(self) -> None:
        with self.condition:
            self.eos = True
            self.condition.notify_all()

    def set_abandoned(self) -> None:
        with self.condition:
            self.abandoned = True
            self.eos = True
            self.condition.notify_all()

    def set_error(self, exc: BaseException) -> None:
        with self.condition:
            if self.error is None:
                self.error = exc
            self.eos = True
            self.condition.notify_all()

    def append_live_text(self, text: str) -> None:
        with self.condition:
            self.live_text += text

    def check_error(self) -> None:
        with self.condition:
            exc = self.error
        if exc is not None:
            raise TranscriptionFailed(f"streaming failed: {exc}") from exc


# --- the adapter ------------------------------------------------------------


class NemotronTranscriber:
    """Transcriber port backed by Transformers streaming inference (Nemotron 3.5 ASR)."""

    def __init__(
        self,
        *,
        model_name: str,
        language: str,
        device: str,
        chunk_ms: int,
        lookahead_s: float,
        sample_rate: int,
    ) -> None:
        self._model_name = model_name
        self._language = language
        self._device_pref = device
        self._chunk_ms = chunk_ms
        self._lookahead_s = lookahead_s
        self._sample_rate = sample_rate

        self._model: Any = None
        self._processor: Any = None
        self._lookahead_tokens = right_context_frames(chunk_ms)  # validated early
        self._resolved_device = device
        self._lock = threading.Lock()
        self._sessions: dict[str, _StreamSession] = {}
        # The Transformers streaming mixin mutates instance state per generate()
        # (it pops prompt_ids into self._prompt_ids and monkey-patches
        # self.get_audio_features), so concurrent streams must not overlap
        # inside generate(). The GPU serializes kernels anyway; the lock only
        # makes the patch/unpatch window race-free.
        self._generate_lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------

    async def load(self) -> None:
        await asyncio.to_thread(self._load_blocking)

    async def unload(self) -> None:
        await asyncio.to_thread(self._unload_blocking)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def _load_blocking(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are not installed — run `uv sync --extra asr` "
                "(or use the Docker image), or set TRANSCRIPT_ASR_PROVIDER=fake"
            ) from exc

        logger.info(
            "loading %s (%s, chunk_ms=%d)", self._model_name, self._language, self._chunk_ms
        )
        processor = AutoProcessor.from_pretrained(self._model_name)
        supported = list(processor.supported_num_lookahead_tokens)
        if self._lookahead_tokens not in supported:
            raise TranscriptionFailed(
                f"chunk_ms={self._chunk_ms} needs lookahead {self._lookahead_tokens}, "
                f"unsupported by this model (supported: {supported})"
            )
        if self._language not in processor.prompt_dictionary:
            raise TranscriptionFailed(
                f"unsupported language {self._language!r} for {self._model_name}"
            )
        processor.set_num_lookahead_tokens(self._lookahead_tokens)

        device_map = self._resolve_device_map(torch)
        logger.info("device_map=%s for %s", device_map, self._model_name)
        model = AutoModelForRNNT.from_pretrained(self._model_name, device_map=device_map)
        model.eval()

        with self._lock:
            self._processor = processor
            self._model = model
            self._resolved_device = (
                "cuda" if torch.cuda.is_available() and self._device_pref != "cpu" else "cpu"
            )

    def _unload_blocking(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._model = None
            self._processor = None
        for session in sessions:
            session.set_abandoned()
            self._join_session(session, CLOSE_JOIN_TIMEOUT_S)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolve_device_map(self, torch: Any) -> str:
        if self._device_pref == "cpu":
            return "cpu"
        if self._device_pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("TRANSCRIPT_DEVICE=cuda but no CUDA device is available")
            return "cuda"
        # "auto": pin the whole model to the first GPU instead of letting
        # accelerate shard it. Proven on a 2x RTX 4000 Ada box: device_map="auto"
        # shards the 0.6B model across both cards (accelerate hooks active) and
        # the first streaming generate() dies in the prompt one_hot with a
        # ScatterGather index-out-of-bounds on valid prompt_ids ([101] < 128);
        # the identical call with device_map="cuda:0" streams fine. The model
        # (~2.5 GB) easily fits on one card, and all streams share it anyway.
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    # --- Transcriber port -----------------------------------------------

    async def open_stream(self, stream_id: str) -> None:
        await asyncio.to_thread(self._open_blocking, stream_id)

    async def push_audio(self, stream_id: str, pcm: bytes) -> list[Word]:
        return await asyncio.to_thread(self._push_blocking, stream_id, pcm)

    async def finish_stream(self, stream_id: str) -> list[Word]:
        return await asyncio.to_thread(self._finish_blocking, stream_id)

    async def close_stream(self, stream_id: str) -> None:
        session = self._sessions.pop(stream_id, None)  # idempotent by contract
        if session is not None:
            session.set_abandoned()
            await asyncio.to_thread(self._join_session, session, CLOSE_JOIN_TIMEOUT_S)

    def _open_blocking(self, stream_id: str) -> None:
        if self._model is None:
            raise TranscriptionFailed("ASR model is not loaded yet")
        with self._lock:
            if stream_id in self._sessions:
                raise TranscriptionFailed(f"transcriber already has stream {stream_id!r}")
            planner = ChunkPlanner(
                first_samples=int(self._processor.num_samples_first_audio_chunk),
                per_chunk_samples=int(self._processor.num_samples_per_audio_chunk),
                first_mel=int(self._processor.num_mel_frames_first_audio_chunk),
                per_mel=int(self._processor.num_mel_frames_per_audio_chunk),
                hop_length=int(self._processor.feature_extractor.hop_length),
                n_fft=int(self._processor.feature_extractor.n_fft),
            )
            session = _StreamSession(stream_id, planner, self._sample_rate)
            self._sessions[stream_id] = session
        session.pump_thread = threading.Thread(
            target=self._pump_target, args=(session,), name=f"asr-pump-{stream_id}", daemon=True
        )
        session.consumer_thread = threading.Thread(
            target=self._consume_target,
            args=(session,),
            name=f"asr-consumer-{stream_id}",
            daemon=True,
        )
        session.pump_thread.start()
        session.consumer_thread.start()

    def _push_blocking(self, stream_id: str, pcm: bytes) -> list[Word]:
        session = self._require_session(stream_id)
        session.append_pcm(pcm)
        session.check_error()
        return self._drain_new_words(session, final=False)

    def _finish_blocking(self, stream_id: str) -> list[Word]:
        session = self._require_session(stream_id)
        session.set_eos()
        self._join_session(session, PUMP_JOIN_TIMEOUT_S)
        session.check_error()
        return self._drain_new_words(session, final=True)

    def _require_session(self, stream_id: str) -> _StreamSession:
        if self._model is None:
            raise TranscriptionFailed("ASR model is not loaded yet")
        try:
            return self._sessions[stream_id]
        except KeyError:
            raise TranscriptionFailed(f"unknown transcriber stream {stream_id!r}") from None

    @staticmethod
    def _join_session(session: _StreamSession, timeout_s: float) -> None:
        for thread in (session.pump_thread, session.consumer_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout_s)

    # --- pump + consumer ------------------------------------------------

    def _pump_target(self, session: _StreamSession) -> None:
        """Run one generate() call fed by live audio (documented pattern, live source)."""
        try:
            import torch
            from transformers import TextIteratorStreamer

            first_inputs = self._wait_first_inputs(session)
            if first_inputs is None:
                return  # abandoned before any audio / error already recorded
            with session.condition:
                if session.abandoned or session.error is not None:
                    return
                first_features = session.first_features
                session.first_features = None
            session.streamer = TextIteratorStreamer(
                self._processor.tokenizer, skip_special_tokens=True
            )
            generate_kwargs: dict[str, Any] = {
                **first_inputs,
                "input_features": self._live_feature_generator(session, first_features),
                "streamer": session.streamer,
            }
            with torch.inference_mode(), self._generate_lock:
                self._model.generate(**generate_kwargs)
        except Exception as exc:  # fail loud: push/finish surface it, never stall silently
            logger.exception("streaming pump failed for stream %s", session.stream_id)
            session.set_error(exc)
        finally:
            session.pump_done = True
            streamer = session.streamer
            if streamer is not None:
                with suppress(Exception):  # generate() usually ends it already
                    streamer.end()

    def _wait_first_inputs(self, session: _StreamSession) -> Any | None:
        """Block until the first window is available; None when abandoned/failed."""
        while True:
            with session.condition:
                if session.abandoned or session.error is not None:
                    return None
                # Absolute coordinates: the planner works on the total timeline,
                # not on the trimmed buffer (see drop_before).
                plan = session.planner.plan(session.samples_pushed, eos=session.eos)
                if plan is None:
                    if session.eos:
                        return None  # ended with no audio at all
                    session.condition.wait(timeout=0.5)
                    continue
            inputs = self._features_for(session, plan, is_first=True)
            # Trim to the documented first-chunk mel width and stash it: the
            # feature generator must yield the first chunk itself.
            inputs["input_features"] = inputs["input_features"][:, : plan.mel_frames, :]
            session.first_features = inputs["input_features"]
            return inputs

    def _live_feature_generator(
        self, session: _StreamSession, first_features: Any
    ) -> Iterator[Any]:
        """Yield the first chunk, then subsequent ones (documented pattern)."""
        yield first_features
        while True:
            with session.condition:
                if session.abandoned or session.error is not None:
                    return
                plan = session.planner.plan(session.samples_pushed, eos=session.eos)
                if plan is None:
                    if session.eos:
                        return
                    session.condition.wait(timeout=0.5)
                    continue
            inputs = self._features_for(session, plan, is_first=False)
            yield inputs["input_features"]

    def _features_for(self, session: _StreamSession, plan: ChunkPlan, *, is_first: bool) -> Any:
        """Materialize a planned window and run the processor (documented call shape)."""
        pcm_view = session.materialize(plan.start_sample, plan.end_sample)
        session.drop_before(plan.start_sample)
        inputs = self._processor(
            pcm_view,
            sampling_rate=session.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=is_first,
            language=self._language,
            return_tensors="pt",
        )
        return inputs.to(self._model.device, dtype=self._model.dtype)

    def _consume_target(self, session: _StreamSession) -> None:
        """Drain the streamer into live text; ends when generate() finishes."""
        while True:
            streamer = session.streamer
            if streamer is None:
                if session.pump_done or session.abandoned:
                    return
                time.sleep(0.05)
                continue
            try:
                for text_chunk in streamer:
                    session.append_live_text(text_chunk)
                return
            except Exception:
                logger.exception("streamer consumer failed for stream %s", session.stream_id)
                return

    # --- word attribution -----------------------------------------------

    def _drain_new_words(self, session: _StreamSession, *, final: bool) -> list[Word]:
        with session.condition:
            live = session.live_text
        clean = strip_language_tag(live)
        word_texts = split_new_words(session.emitted_text, clean)
        session.emitted_text = clean
        if not word_texts:
            return []

        now_s = session.samples_pushed / self._sample_rate
        span_end = now_s if final else max(session.span_start_s, now_s - self._lookahead_s)
        # Cap each growth step so one late word cannot sprawl over a long silence.
        span_end = min(span_end, session.span_start_s + MAX_WORD_SPAN_S)

        spans = allocate_word_spans(len(word_texts), session.span_start_s, span_end)
        session.span_start_s = span_end
        return [
            Word(text=text, timestamp=Timestamp(start=start, end=end))
            for text, (start, end) in zip(word_texts, spans, strict=True)
        ]
