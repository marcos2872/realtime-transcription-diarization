"""Testes do protocolo Azure (pula se pydantic não instalado)."""

import pytest

pydantic = pytest.importorskip("pydantic")

from src.api.ws.azure_protocol import (
    SpeakerMapper,
    bytes_to_ticks,
    sec_to_ticks,
)


def test_sec_to_ticks():
    assert sec_to_ticks(1.0) == 10_000_000
    assert sec_to_ticks(0.5) == 5_000_000


def test_bytes_to_ticks():
    assert bytes_to_ticks(32000) == 10_000_000


def test_speaker_mapper_ordem_aparicao():
    mapper = SpeakerMapper()
    assert mapper.get("Pessoa 1") == 1
    assert mapper.get("Pessoa 2") == 2
    assert mapper.get("Pessoa 1") == 1


def test_speaker_mapper_max_mescla():
    mapper = SpeakerMapper(max_speakers=2)
    mapper.get("Pessoa 1")
    mapper.get("Pessoa 2")
    assert mapper.get("Pessoa 3") == 2
