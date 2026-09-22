"""Speaker value object: the label attributed to whoever is talking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Speaker:
    """A speaker label such as ``SPEAKER_00`` (or ``UNKNOWN`` when diarization is inconclusive)."""

    label: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("speaker label must not be empty")

    @classmethod
    def numbered(cls, index: int) -> Speaker:
        if index < 0:
            raise ValueError(f"speaker index must be >= 0, got {index}")
        return cls(label=f"SPEAKER_{index:02d}")

    UNKNOWN: ClassVar[Speaker]  # sentinel instance, assigned right after the class body


# The sentinel can only be built once the class exists.
Speaker.UNKNOWN = Speaker(label="UNKNOWN")
