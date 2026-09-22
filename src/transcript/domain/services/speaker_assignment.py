"""assign_speaker domain service: attribute an utterance to a speaker turn."""

from __future__ import annotations

from transcript.domain.entities.transcript import Utterance
from transcript.domain.value_objects.speaker import Speaker
from transcript.domain.value_objects.speaker_turn import SpeakerTurn

# After this many seconds of silence the previous speaker can no longer be assumed
# to be the same person (turn-taking happens during pauses in normal conversation).
CONTINUITY_MARGIN_S = 10.0


def assign_speaker(
    utterance: Utterance,
    turns: list[SpeakerTurn],
    continuity_margin_s: float = CONTINUITY_MARGIN_S,
) -> Speaker:
    """Pick the speaker for ``utterance``.

    Rules, in order:
    1. The speaker with the largest time overlap wins.
    2. On a silence gap, keep the most recent speaker that finished less than
       ``continuity_margin_s`` before the utterance started (people pause, then
       resume talking).
    3. Otherwise ``Speaker.UNKNOWN`` — better no answer than a wrong one.
    """
    if not turns:
        return Speaker.UNKNOWN

    best = max(turns, key=lambda turn: turn.timestamp.overlap_seconds(utterance.timestamp))
    if best.timestamp.overlap_seconds(utterance.timestamp) > 0:
        return best.speaker

    preceding = [
        turn
        for turn in turns
        if turn.timestamp.end <= utterance.timestamp.start
        and utterance.timestamp.start - turn.timestamp.end <= continuity_margin_s
    ]
    if preceding:
        return max(preceding, key=lambda turn: turn.timestamp.end).speaker

    return Speaker.UNKNOWN
