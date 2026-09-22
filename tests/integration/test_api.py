"""End-to-end tests of the WebSocket API using the fake providers (no GPU)."""

from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from transcript.di import build_application
from transcript.main import create_app


@pytest.fixture
def client(settings):
    application = build_application(settings)
    with TestClient(create_app(application)) as test_client:
        yield test_client


def test_health_reports_providers_and_zero_streams(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_streams"] == 0
    assert body["max_streams"] == 4
    assert body["asr_provider"] == "fake"
    assert body["diarization_provider"] == "fake"
    assert body["language"] == "pt-BR"


def test_root_exposes_service_identity(client):
    body = client.get("/").json()

    assert body["name"] == "transcript-service"
    assert body["version"]
    assert body["language"] == "pt-BR"


def test_openapi_is_available(client):
    assert client.get("/openapi.json").status_code == 200


def test_websocket_happy_path_emits_partials_utterances_and_close(client):
    with client.websocket_connect("/ws/streams/s1") as ws:
        ready = ws.receive_json()
        assert ready == {
            "type": "ready",
            "stream_id": "s1",
            "language": "pt-BR",
            "sample_rate": 16_000,
            "max_streams": 4,
            "partials": True,
        }

        one_second = b"\x00\x00" * 16_000

        ws.send_bytes(one_second)  # word 1: "Olá"
        partial = ws.receive_json()
        assert partial == {"type": "partial", "stream_id": "s1", "text": "Olá"}

        ws.send_bytes(one_second)  # word 2: "mundo"
        partial = ws.receive_json()
        assert partial["text"] == "Olá mundo"

        ws.send_bytes(one_second)  # word 3: "." -> sentence closed
        utterance = ws.receive_json()
        assert utterance["type"] == "utterance"
        assert utterance["stream_id"] == "s1"
        assert utterance["utterance_id"] == 0
        assert utterance["text"] == "Olá mundo ."
        assert utterance["speaker"].startswith("SPEAKER_")
        assert utterance["start"] == pytest.approx(0.0)
        assert utterance["end"] == pytest.approx(3.0)

        ws.send_text('{"type": "end"}')
        closed = ws.receive_json()
        assert closed == {"type": "closed", "stream_id": "s1"}

    # Slot must be free again after the connection ends.
    assert client.get("/health").json()["active_streams"] == 0


def test_websocket_partials_can_be_disabled_per_connection(client):
    with client.websocket_connect("/ws/streams/s2?partials=false") as ws:
        ready = ws.receive_json()
        assert ready["partials"] is False


def test_duplicate_stream_gets_a_fatal_error(client):
    with (
        client.websocket_connect("/ws/streams/s1"),
        client.websocket_connect("/ws/streams/s1") as ws_second,
    ):
        error = ws_second.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "stream_exists"
        assert error["fatal"] is True
        with pytest.raises(WebSocketDisconnect):
            ws_second.receive_json()


def test_stream_limit_is_enforced(client, settings):
    with ExitStack() as stack:
        for index in range(settings.max_streams):
            stack.enter_context(client.websocket_connect(f"/ws/streams/s{index}"))

        with client.websocket_connect("/ws/streams/one-too-many") as ws_extra:
            error = ws_extra.receive_json()
            assert error["code"] == "stream_limit"
            assert error["fatal"] is True


def test_blank_stream_id_is_rejected(client):
    with client.websocket_connect("/ws/streams/%20") as ws:
        error = ws.receive_json()
        assert error["code"] == "invalid_stream_id"
        assert error["fatal"] is True


def test_ping_pong_and_invalid_messages_are_non_fatal(client):
    with client.websocket_connect("/ws/streams/s3") as ws:
        ws.receive_json()  # ready

        ws.send_text("i am not json")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_message"
        assert error["fatal"] is False

        ws.send_text('{"type": "ping"}')
        assert ws.receive_json() == {"type": "pong"}


def test_odd_sized_audio_chunks_are_resampled_across_frames(client):
    with client.websocket_connect("/ws/streams/s4") as ws:
        ws.receive_json()  # ready

        first, second = 16_001, 15_999  # 1 byte of carry across the two frames
        ws.send_bytes(b"\x00\x00" * (first // 2) + b"\x00")  # odd: 1 byte carried
        partial = ws.receive_json()
        assert partial["text"] == "Olá"  # first full second still transcribed

        ws.send_bytes(b"\x00" + b"\x00\x00" * (second // 2))  # completes the byte
        partial = ws.receive_json()
        assert partial["text"] == "Olá mundo"

        ws.send_text('{"type": "end"}')
        # Closing flushes the pending sentence ("Olá mundo", no punctuation yet)
        # before reporting the stream as closed.
        utterance = ws.receive_json()
        assert utterance["type"] == "utterance"
        assert utterance["text"] == "Olá mundo"
        assert utterance["speaker"].startswith("SPEAKER_")
        assert ws.receive_json()["type"] == "closed"


def test_disconnecting_cleans_the_stream_slot(client):
    with client.websocket_connect("/ws/streams/s5") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 8_000)
        ws.receive_json()  # partial
        # Leaving the context closes the socket abruptly.

    assert client.get("/health").json()["active_streams"] == 0


def test_stream_shorter_than_first_diarization_window_still_emits_a_speaker(client, settings):
    """Very short stream: closes before the first hop cadence => final pass at close."""
    with client.websocket_connect("/ws/streams/s6") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 4_000)  # 0.25s, below diarization_min_s
        partial = ws.receive_json()
        assert partial["type"] == "partial"

        ws.send_text('{"type": "end"}')
        utterance = ws.receive_json()
        assert utterance["type"] == "utterance"
        # Final diarization at close does run (turns are empty) => real speaker.
        assert utterance["speaker"].startswith("SPEAKER_")
        assert ws.receive_json()["type"] == "closed"
