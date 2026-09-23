"""Testes de domínio: agregado de transcrição."""

from src.domain.entities.transcript import build_transcript


def test_build_transcript_participants_ordem():
    transcript = build_transcript(
        "s1",
        [
            {"speaker": "Pessoa 1", "text": " a ", "tStart": 0.0, "tEnd": 1.0},
            {"speaker": "Pessoa 2", "text": "b", "tStart": 1.0, "tEnd": 2.0},
            {"speaker": "Pessoa 1", "text": "c", "tStart": 2.0, "tEnd": 3.0},
        ],
        3.0,
        "pt",
    )
    assert transcript.session_id == "s1"
    assert transcript.participants == ["Pessoa 1", "Pessoa 2"]
    assert transcript.segments[0].text == "a"
    assert transcript.language == "pt"


def test_build_transcript_speaker_ausente_vira_locutor():
    transcript = build_transcript(
        "s2",
        [{"text": "x", "tStart": 0, "tEnd": 1}],
        1.0,
        "pt",
    )
    assert transcript.segments[0].speaker == "Locutor"
    assert transcript.participants == ["Locutor"]


def test_build_transcript_arredonda():
    transcript = build_transcript(
        "s3",
        [{"speaker": "Eu", "text": "x", "tStart": 0.123, "tEnd": 1.987}],
        1.999,
        "pt",
    )
    assert transcript.segments[0].t_start == 0.12
    assert transcript.duration_sec == 2.0
