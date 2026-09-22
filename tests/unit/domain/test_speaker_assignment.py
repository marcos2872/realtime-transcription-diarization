"""Tests for the speaker assignment domain service."""

from transcript.domain.entities import Utterance
from transcript.domain.services import assign_speaker
from transcript.domain.value_objects import Speaker, SpeakerTurn, Timestamp


def _utterance(start: float, end: float) -> Utterance:
    return Utterance(id=0, timestamp=Timestamp(start=start, end=end), text="oi")


def _turn(start: float, end: float, index: int) -> SpeakerTurn:
    return SpeakerTurn(timestamp=Timestamp(start=start, end=end), speaker=Speaker.numbered(index))


def test_picks_the_speaker_with_the_largest_overlap():
    turns = [_turn(0.0, 4.0, 0), _turn(4.0, 10.0, 1)]

    assert assign_speaker(_utterance(4.5, 8.0), turns) == Speaker.numbered(1)
    assert assign_speaker(_utterance(0.5, 3.5), turns) == Speaker.numbered(0)


def test_falls_back_to_the_most_recent_speaker_right_before_the_utterance():
    turns = [_turn(0.0, 4.0, 0), _turn(4.0, 5.0, 1)]  # last turn ends at 5s
    utterance = _utterance(6.0, 9.0)  # silence gap, no direct overlap

    assert assign_speaker(utterance, turns) == Speaker.numbered(1)


def test_ignores_speakers_that_are_too_old_to_be_continuous():
    turns = [_turn(0.0, 1.0, 0)]
    utterance = _utterance(60.0, 63.0)  # far beyond the continuity margin

    assert assign_speaker(utterance, turns) == Speaker.UNKNOWN


def test_no_turns_at_all_yields_unknown():
    assert assign_speaker(_utterance(0.0, 1.0), []) == Speaker.UNKNOWN


def test_silence_gap_within_margin_keeps_the_previous_speaker():
    turns = [_turn(10.0, 12.0, 1)]
    utterance = _utterance(15.0, 17.0)  # 3s gap < 10s continuity margin

    assert assign_speaker(utterance, turns) == Speaker.numbered(1)
