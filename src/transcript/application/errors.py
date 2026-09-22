"""Application errors: one class per failure, each carrying an API-facing code."""

from __future__ import annotations


class UseCaseError(Exception):
    """Base class for expected application failures."""

    code = "use_case_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidStreamId(UseCaseError):
    code = "invalid_stream_id"


class StreamNotFound(UseCaseError):
    code = "stream_not_found"


class StreamAlreadyExists(UseCaseError):
    code = "stream_exists"


class StreamLimitReached(UseCaseError):
    code = "stream_limit"


class InvalidAudioChunk(UseCaseError):
    code = "invalid_audio"


class TranscriptionFailed(UseCaseError):
    code = "transcription_failed"


class DiarizationFailed(UseCaseError):
    code = "diarization_failed"
