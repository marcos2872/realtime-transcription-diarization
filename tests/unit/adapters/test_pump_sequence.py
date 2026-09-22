"""End-to-end test of the pump/generator/streamer glue with fakes.

Reproduces the documented Transformers streaming contract: the feature
generator must yield the trimmed first chunk first (25 mel frames for
lookahead 3), then full subsequent chunks (32). A stubbed generate()
validates widths exactly like the real validator, so a sequencing
regression fails here instead of on the GPU.
"""

import queue
import sys
import threading
import types

import numpy as np
import pytest

from transcript.adapters.asr.nemotron_transcriber import (
    ChunkPlanner,
    NemotronTranscriber,
    _StreamSession,
)

FIRST_MEL = 25
PER_MEL = 32


class FakeBatch(dict):
    """Duck-typed BatchFeature: **-unpackable, .to()able, numpy-valued."""

    def to(self, *args, **kwargs):
        return self


class FakeProcessor:
    tokenizer = object()

    def __call__(self, pcm, **kwargs):
        width = 26 if kwargs.get("is_first_audio_chunk") else PER_MEL
        return FakeBatch(
            {
                "input_features": np.zeros((1, width, 128), dtype=np.float32),
                "prompt_ids": np.zeros((1,), dtype=np.int64),
                "num_lookahead_tokens": 3,
            }
        )


class FakeStreamer:
    """Minimal TextIteratorStreamer: put()/end() + blocking iteration."""

    _END = object()

    def __init__(self, *args, **kwargs):
        self._queue: queue.Queue = queue.Queue()

    def put(self, text: str) -> None:
        self._queue.put(text)

    def end(self) -> None:
        self._queue.put(self._END)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._END:
                return
            yield item


class FakeModel:
    device = "cpu"
    dtype = "float32"
    seen_widths: list[int]

    def __init__(self) -> None:
        self.seen_widths = []

    def generate(self, **kwargs):
        for features in kwargs["input_features"]:
            self.seen_widths.append(int(features.shape[1]))
        expected = [FIRST_MEL] + [PER_MEL] * (len(self.seen_widths) - 1)
        assert self.seen_widths == expected, f"chunk widths {self.seen_widths} != {expected}"
        streamer = kwargs["streamer"]
        streamer.put("Olá mundo. ")
        streamer.end()
        return "done"


@pytest.fixture
def stubbed_backend(monkeypatch):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.TextIteratorStreamer = FakeStreamer
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    import contextlib

    torch_stub = types.ModuleType("torch")
    torch_stub.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def make_adapter() -> NemotronTranscriber:
    adapter = NemotronTranscriber(
        model_name="stub",
        language="pt-BR",
        device="cpu",
        chunk_ms=320,
        lookahead_s=1.4,
        sample_rate=16_000,
    )
    adapter._processor = FakeProcessor()  # noqa: SLF001 - wiring fakes for the glue test
    adapter._model = FakeModel()  # noqa: SLF001
    return adapter


def make_session() -> _StreamSession:
    planner = ChunkPlanner(
        first_samples=4040,
        per_chunk_samples=5520,
        first_mel=FIRST_MEL,
        per_mel=PER_MEL,
        hop_length=160,
        n_fft=512,
    )
    return _StreamSession("s1", planner, sample_rate=16_000)


def test_pump_yields_first_chunk_then_subsequent_and_streams_text(stubbed_backend):
    adapter = make_adapter()
    session = make_session()
    # 10 000 samples cover [0,4040), [3744,9264) and a padded [8864,14384).
    session.append_pcm(np.zeros(10_000, dtype=np.int16).tobytes())
    session.set_eos()

    pump = threading.Thread(target=adapter._pump_target, args=(session,))  # noqa: SLF001
    consumer = threading.Thread(target=adapter._consume_target, args=(session,))  # noqa: SLF001
    pump.start()
    consumer.start()
    pump.join(timeout=10)
    consumer.join(timeout=10)

    assert not pump.is_alive()
    assert not consumer.is_alive()
    assert session.error is None
    assert session.live_text == "Olá mundo. "
    assert adapter._model.seen_widths == [FIRST_MEL, PER_MEL, PER_MEL]  # noqa: SLF001


def test_pump_with_no_audio_ends_without_generate(stubbed_backend):
    adapter = make_adapter()
    session = make_session()
    session.set_eos()  # closed before any audio arrived

    adapter._pump_target(session)  # noqa: SLF001 - direct call, no threads needed

    assert session.error is None
    assert session.pump_done
    assert adapter._model.seen_widths == []  # noqa: SLF001 - generate never ran
