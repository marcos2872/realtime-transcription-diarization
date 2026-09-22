"""Tests for the Speaker and SpeakerTurn value objects."""

from transcript.domain.value_objects import Speaker, SpeakerTurn, Timestamp


def test_numbered_speaker_is_zero_padded():
    assert Speaker.numbered(0).label == "SPEAKER_00"
    assert Speaker.numbered(7).label == "SPEAKER_07"
    assert Speaker.numbered(12).label == "SPEAKER_12"


def test_unknown_speaker_is_a_singleton_label():
    assert Speaker.UNKNOWN.label == "UNKNOWN"


def test_speaker_turn_bundles_timestamp_and_speaker():
    turn = SpeakerTurn(timestamp=Timestamp(start=0.0, end=3.0), speaker=Speaker.numbered(1))

    assert turn.speaker.label == "SPEAKER_01"
    assert turn.timestamp.duration == 3.0
