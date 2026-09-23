"""Testes de aplicação: use cases com fakes (sem GPU/rede/disco real)."""

import asyncio

from src.application.use_cases.finalize_stream import finalize_stream
from src.application.use_cases.transcribe_audio import transcribe_audio


class FakeTranscriber:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    async def dispatch(self, audio_path, language="pt", word_timestamps=False):
        self.calls.append((audio_path, language))
        return [dict(s) for s in self._segments]


class FakeTmp:
    def __init__(self):
        self.removed = []

    def save(self, data: bytes) -> str:
        return "/tmp/fake.wav"

    def remove(self, path: str) -> None:
        self.removed.append(path)


class FakeDiarizer:
    def __init__(self, fail=False):
        self.fail = fail

    def diarize(self, audio_path):
        if self.fail:
            raise RuntimeError("sem HF_TOKEN")
        return [{"speaker": "SPEAKER_00", "tStart": 0.0, "tEnd": 10.0}]

    def assign_speakers(self, segments, diarization):
        return [{**s, "speaker": "Pessoa 1"} for s in segments]


class FakeSession:
    def __init__(self, channels=("mic", "system"), diarize=False):
        self.id = "sess-1"
        self.language = "pt"
        self.channels = list(channels)
        self.diarize = diarize

    def get_audio(self, channel):
        return b"RIFF" + b"\x00" * 100

    def cached_diarization(self, channel):
        return None

    @property
    def duration_sec(self):
        return 5.0


def run(coro):
    return asyncio.run(coro)


def _patch_size(monkeypatch):
    monkeypatch.setattr("os.path.getsize", lambda p: 44 + 32000 * 2)


def test_transcribe_sem_diarize(monkeypatch):
    _patch_size(monkeypatch)
    tmp = FakeTmp()
    out = run(transcribe_audio(
        b"wav", "pt", False, "abc",
        FakeTranscriber([{"speaker": "Locutor", "text": "oi", "tStart": 0.0, "tEnd": 1.0}]),
        tmp,
    ))
    assert out.session_id == "abc"
    assert out.participants == ["Locutor"]
    assert out.duration_sec == 2.0
    assert tmp.removed == ["/tmp/fake.wav"]


def test_transcribe_com_diarize(monkeypatch):
    _patch_size(monkeypatch)
    out = run(transcribe_audio(
        b"wav", "pt", True, None,
        FakeTranscriber([{"speaker": "Locutor", "text": "oi", "tStart": 0.0, "tEnd": 1.0}]),
        FakeTmp(), FakeDiarizer(),
    ))
    assert out.participants == ["Pessoa 1"]


def test_transcribe_diarize_falha_faz_fallback(monkeypatch):
    _patch_size(monkeypatch)
    out = run(transcribe_audio(
        b"wav", "pt", True, "s",
        FakeTranscriber([{"speaker": "Locutor", "text": "oi", "tStart": 0.0, "tEnd": 1.0}]),
        FakeTmp(), FakeDiarizer(fail=True),
    ))
    assert out.participants == ["Locutor"]


def test_finalize_stream_regras_locutor():
    session = FakeSession()
    out = run(finalize_stream(
        session,
        FakeTranscriber([{"speaker": "Locutor", "text": "x", "tStart": 0.0, "tEnd": 1.0}]),
        FakeTmp(),
    ))
    by_speaker = [s.speaker for s in out.segments]
    assert by_speaker == ["Eu", "Sistema"]
