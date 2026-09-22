"""Nemotron streaming transcriber, backed by NVIDIA NeMo cache-aware inference.

The NeMo cache-aware model (``conformer_stream_step`` + ``CacheAwareStreamingAudioBuffer``)
expects mel-spectrogram *features*, so this adapter owns the whole GPU loop:

1. PCM16 chunks accumulate in :class:`_LiveAudioBuffer`.
2. Only complete, stride-aligned feature frames are extracted (with left
   context so every frame matches whole-file preprocessing — no STFT seams).
3. Each emitted frame is appended to NeMo's buffer and immediately decoded;
   only the text grown since the previous call is returned, split into words
   timestamped inside the real-time horizon ``[span_start, now - lookahead]``.

Everything NeMo touches runs on a worker thread (``asyncio.to_thread``) behind
a single lock, so the event loop never blocks while 4 streams share the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from transcript.application.errors import TranscriptionFailed
from transcript.domain.value_objects import Timestamp, Word

logger = logging.getLogger(__name__)

# NeMo accepts the right attention context (lookahead) in 80 ms encoder frames;
# the Nemotron model card documents {0, 1, 3, 6, 13} for {80, 160, 320, 560, 1120} ms.
RIGHT_CONTEXT_FRAMES_BY_CHUNK_MS: dict[int, int] = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}

# Suffix emitted in language auto-detect mode, appended after terminal punctuation.
_LANGUAGE_TAG_SUFFIX = re.compile(r"\s*<[a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4})?>\s*$")

# Words attributed to a single growth step are capped to this span, so a long
# silence followed by one word does not smear that word over a minute of timeline.
MAX_WORD_SPAN_S = 8.0


# --- pure helpers (unit-tested, no GPU/model needed) ------------------------


def right_context_frames(chunk_ms: int) -> int:
    """Map a streaming chunk size to NeMo's right attention context (80 ms frames)."""
    try:
        return RIGHT_CONTEXT_FRAMES_BY_CHUNK_MS[chunk_ms]
    except KeyError:
        expected = ", ".join(str(ms) for ms in sorted(RIGHT_CONTEXT_FRAMES_BY_CHUNK_MS))
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


# --- live audio feed --------------------------------------------------------


class _LiveAudioBuffer:
    """Seam-free, chunk-gated feed into NeMo's ``CacheAwareStreamingAudioBuffer``.

    NeMo's buffer preprocesses each appended piece independently, so raw PCM is
    accumulated here and extracted as mel features over a sliding window with
    ``n_fft`` samples of left context: every feature frame the model ever sees
    matches whole-file preprocessing exactly. Whole-file frame counts
    (``len / hop`` at the end of stream) are never assumed up front — the
    preprocessor output itself tells us how many frames a piece really holds.
    """

    def __init__(self, model: Any) -> None:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self._nemo = CacheAwareStreamingAudioBuffer(model=model)
        self._preprocessor = self._nemo.preprocessor  # dither/pad-free copy owned by the buffer
        cfg = model.cfg.preprocessor
        self._hop = int(cfg.hop_length)
        self._n_fft = int(cfg.n_fft)
        self._left_context_frames = -(-self._n_fft // self._hop)  # ceil(n_fft / hop)

        shift = self._nemo.streaming_cfg.shift_size
        self._shift = int(shift[1] if isinstance(shift, list) else shift)

        self._raw = bytearray()
        self._raw_start_sample = 0  # absolute sample index of _raw[0]
        self._written_frame = 0  # absolute frames already appended to NeMo's buffer
        self._device = self._nemo.get_model_device()

    @property
    def nemo_buffer(self) -> Any:
        return self._nemo

    def feed(self, pcm: bytes, *, final: bool = False) -> None:
        """Append PCM16 and extract every newly completable frame window.

        Non-final pushes extract only frames whose full STFT context has
        arrived (no look-ahead reads past what was pushed). The final call
        flushes the tail, whose right edge the preprocessor zero-pads — exactly
        like whole-file inference.
        """
        if pcm:
            self._raw += pcm
        raw_samples = len(self._raw) // 2
        if raw_samples == 0:
            return

        if final:
            target = self._written_frame + self._remaining_frames(raw_samples, final=True)
        else:
            available = self._raw_start_sample + self._remaining_frames(raw_samples, final=False)
            complete = (available - self._written_frame) // self._shift
            if complete <= 0:
                return
            target = self._written_frame + complete * self._shift

        if target > self._written_frame:
            self._extract_and_append(target)

    def _remaining_frames(self, raw_samples: int, *, final: bool) -> int:
        absolute_samples = self._raw_start_sample + raw_samples
        if final:
            highest = absolute_samples // self._hop  # whole-file frame count (inclusive)
        else:
            highest = (absolute_samples - self._n_fft // 2) // self._hop
        return max(0, highest + 1 - self._written_frame)

    def _extract_and_append(self, target: int) -> None:
        import torch

        window_start_sample = max(
            self._raw_start_sample, self._written_frame * self._hop - self._n_fft
        )
        byte_offset = (window_start_sample - self._raw_start_sample) * 2
        samples = np.frombuffer(bytes(self._raw[byte_offset:]), dtype="<i2").astype(np.float32)
        samples /= 32768.0
        if samples.size == 0:
            raise RuntimeError("live buffer window unexpectedly empty")

        signal = torch.from_numpy(samples).unsqueeze(0).to(self._device)
        length = torch.tensor([samples.shape[0]])
        with torch.inference_mode():
            features, _ = self._preprocessor(input_signal=signal, length=length)

        local_end = target - window_start_sample // self._hop
        if features.shape[-1] < local_end:
            raise RuntimeError(
                f"preprocessor returned {features.shape[-1]} frames, expected >= {local_end}"
            )
        piece = features[:, :, :local_end]
        stream_id = -1 if self._written_frame == 0 and self._raw_start_sample == 0 else 0
        self._nemo.append_processed_signal(piece, stream_id=stream_id)
        self._written_frame = target

        # Trim raw PCM no longer needed (keep left context for the next window).
        new_start_sample = max(0, target * self._hop - self._n_fft)
        cut = new_start_sample - self._raw_start_sample
        if cut > 0:
            del self._raw[: cut * 2]
            self._raw_start_sample = new_start_sample


@dataclass
class _StreamSession:
    """Per-stream decoding state: GPU caches plus the last emitted text."""

    buffer: _LiveAudioBuffer
    cache_last_channel: Any = None
    cache_last_time: Any = None
    cache_last_channel_len: Any = None
    previous_hypotheses: Any = None
    previous_pred_out: Any = None
    step: int = 0
    text: str = ""
    span_start_s: float = 0.0
    samples_pushed: int = 0
    extra: dict = field(default_factory=dict)


# --- the adapter ------------------------------------------------------------


class NemotronTranscriber:
    """Transcriber port backed by NVIDIA NeMo cache-aware streaming inference."""

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

        self._torch: Any = None
        self._model: Any = None
        self._resolved_device = device
        self._lock = threading.Lock()
        self._sessions: dict[str, _StreamSession] = {}

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
            from nemo.collections.asr.models import ASRModel
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        except ImportError as exc:
            raise RuntimeError(
                "NeMo is not installed — run `uv sync --extra asr` (or use the Docker image), "
                "or set TRANSCRIPT_ASR_PROVIDER=fake for a GPU-free run"
            ) from exc

        right = right_context_frames(self._chunk_ms)  # validated before touching the GPU
        device = self._resolve_device(torch)
        logger.info(
            "loading %s on %s (chunk_ms=%d, language=%s)",
            self._model_name,
            device,
            self._chunk_ms,
            self._language,
        )

        with self._lock:
            model = ASRModel.from_pretrained(self._model_name, map_location=device)
            if hasattr(model.encoder, "set_default_att_context_size"):
                model.encoder.set_default_att_context_size(att_context_size=[56, right])
            if hasattr(model, "set_inference_prompt"):
                model.set_inference_prompt(self._language or "auto")
            if hasattr(model, "change_decoding_strategy"):
                model.change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1))
            decoding = getattr(model, "decoding", None)
            if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
                try:
                    decoding.set_strip_lang_tags(True, lang_tag_pattern=None)
                except TypeError:
                    decoding.set_strip_lang_tags(True)  # older NeMo signature
            model.eval()
            self._torch = torch
            self._model = model
            self._resolved_device = str(device)

    def _unload_blocking(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolve_device(self, torch: Any) -> str:
        if self._device_pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self._device_pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TRANSCRIPT_DEVICE=cuda but no CUDA device is available")
        return self._device_pref

    # --- Transcriber port -----------------------------------------------

    async def open_stream(self, stream_id: str) -> None:
        await asyncio.to_thread(self._open_blocking, stream_id)

    async def push_audio(self, stream_id: str, pcm: bytes) -> list[Word]:
        return await asyncio.to_thread(self._push_blocking, stream_id, pcm)

    async def finish_stream(self, stream_id: str) -> list[Word]:
        return await asyncio.to_thread(self._finish_blocking, stream_id)

    async def close_stream(self, stream_id: str) -> None:
        self._sessions.pop(stream_id, None)  # idempotent by contract

    def _open_blocking(self, stream_id: str) -> None:
        if self._model is None:
            raise TranscriptionFailed("ASR model is not loaded yet")
        with self._lock:
            if stream_id in self._sessions:
                raise TranscriptionFailed(f"transcriber already has stream {stream_id!r}")
            buffer = _LiveAudioBuffer(self._model)
            channel, time, length = self._model.encoder.get_initial_cache_state(batch_size=1)
            self._sessions[stream_id] = _StreamSession(
                buffer=buffer,
                cache_last_channel=channel,
                cache_last_time=time,
                cache_last_channel_len=length,
            )

    def _push_blocking(self, stream_id: str, pcm: bytes) -> list[Word]:
        session = self._require_session(stream_id)
        with self._lock:
            session.samples_pushed += len(pcm) // 2
            session.buffer.feed(pcm)
            new_text = self._decode_pending_chunks(session)
            return self._extract_new_words(session, new_text, final=False)

    def _finish_blocking(self, stream_id: str) -> list[Word]:
        session = self._require_session(stream_id)
        with self._lock:
            session.buffer.feed(b"", final=True)
            new_text = self._decode_pending_chunks(session)
            return self._extract_new_words(session, new_text, final=True)

    def _require_session(self, stream_id: str) -> _StreamSession:
        if self._model is None:
            raise TranscriptionFailed("ASR model is not loaded yet")
        try:
            return self._sessions[stream_id]
        except KeyError:
            raise TranscriptionFailed(f"unknown transcriber stream {stream_id!r}") from None

    # --- GPU decoding loop ----------------------------------------------

    def _decode_pending_chunks(self, session: _StreamSession) -> str | None:
        """Run encoder/decoder steps for every buffered chunk; return latest text."""
        model = self._model
        buffer = session.buffer.nemo_buffer
        latest_text: str | None = None

        for chunk_audio, chunk_lengths in buffer:  # resumes where the last call stopped
            drop = (
                0
                if (session.step == 0 and not buffer.pad_and_drop_preencoded)
                else model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with self._torch.inference_mode():
                (
                    session.previous_pred_out,
                    hypotheses,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.previous_pred_out,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            session.step += 1
            hypothesis = hypotheses[-1]
            latest_text = getattr(hypothesis, "text", None) or str(hypothesis)

        return latest_text

    def _extract_new_words(
        self, session: _StreamSession, new_text: str | None, *, final: bool
    ) -> list[Word]:
        if new_text is None:
            return []
        clean = strip_language_tag(new_text)
        word_texts = split_new_words(session.text, clean)
        session.text = clean
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
