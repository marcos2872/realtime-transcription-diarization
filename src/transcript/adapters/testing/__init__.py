"""Deterministic test doubles (also used by ``provider=fake``)."""

from transcript.adapters.testing.fake_diarizer import FakeDiarizer
from transcript.adapters.testing.fake_transcriber import FakeTranscriber

__all__ = ["FakeDiarizer", "FakeTranscriber"]
