"""Timestamp value object: an immutable time span measured in seconds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Timestamp:
    """Half-open time span ``[start, end]`` in seconds, anchored to the stream timeline."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: Timestamp) -> bool:
        """True when the two spans share interior time (touching does not count)."""
        return self.start < other.end and other.start < self.end

    def overlap_seconds(self, other: Timestamp) -> float:
        """Length of the intersection between the two spans."""
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))
