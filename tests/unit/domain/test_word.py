"""Tests for the Word value object."""

import pytest

from transcript.domain.value_objects import Timestamp, Word


def test_word_reveals_text_and_span():
    word = Word(text="Olá", timestamp=Timestamp(start=0.0, end=0.4))

    assert word.text == "Olá"
    assert word.timestamp.duration == pytest.approx(0.4)


def test_blank_text_is_rejected():
    with pytest.raises(ValueError, match="text"):
        Word(text="   ", timestamp=Timestamp(start=0.0, end=0.4))
