"""Shared helpers translating adapter failures into application errors."""

from __future__ import annotations

import transcript.application.errors as errors
from transcript.domain.entities import AudioStream
from transcript.domain.ports import Diarizer, Transcriber
from transcript.domain.value_objects import Timestamp, Word


async def open_transcriber(transcriber: Transcriber, stream_id: str) -> None:
    try:
        await transcriber.open_stream(stream_id)
    except errors.UseCaseError:
        raise
    except Exception as exc:  # adapter/infra failure — never swallowed
        raise errors.TranscriptionFailed(str(exc)) from exc


async def push_audio(transcriber: Transcriber, stream_id: str, pcm: bytes) -> list[Word]:
    try:
        return await transcriber.push_audio(stream_id, pcm)
    except errors.UseCaseError:
        raise
    except Exception as exc:
        raise errors.TranscriptionFailed(str(exc)) from exc


async def finish_audio(transcriber: Transcriber, stream_id: str) -> list[Word]:
    try:
        return await transcriber.finish_stream(stream_id)
    except errors.UseCaseError:
        raise
    except Exception as exc:
        raise errors.TranscriptionFailed(str(exc)) from exc


async def run_diarization(
    stream: AudioStream,
    diarizer: Diarizer,
    *,
    sample_rate: int,
    window_s: float,
    overlap_s: float,
) -> bool:
    """Diarize the stream tail and record the fresh region.

    Geometry: the reported region is ``[max(window_start, last_end - overlap),
    audio_end]`` — always inside the audio window fed to the model, always
    contiguous with the previous region (settings validate hop + overlap <= window).
    Returns ``False`` when there is nothing new to decide.
    """
    if stream.audio_end <= 0:
        return False

    pcm, window_start = stream.audio_window(window_s)
    if not pcm:
        return False

    region_start = max(window_start, stream.last_diarization_end - overlap_s, 0.0)
    region_end = stream.audio_end
    if region_end <= region_start:
        return False

    try:
        turns = await diarizer.diarize(
            stream_id=stream.id,
            pcm=pcm,
            sample_rate=sample_rate,
            offset=window_start,
        )
    except errors.UseCaseError:
        raise
    except Exception as exc:
        raise errors.DiarizationFailed(str(exc)) from exc

    stream.record_diarization(turns, Timestamp(start=region_start, end=region_end))
    return True
