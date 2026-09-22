"""Word value object: a transcribed word with its approximate time span."""

from __future__ import annotations

from dataclasses import dataclass

from transcript.domain.value_objects.timestamp import Timestamp


@dataclass(frozen=True, slots=True)
class Word:
    """A single transcribed word and when it was (approximately) spoken."""

    text: str
    timestamp: Timestamp

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("word text must not be blank")

    @property
    def ends_sentence(self) -> bool:
        """True when the word closes a sentence (terminal punctuation)."""
        return self.text.rstrip().endswith((".", "?", "!", "…"))
