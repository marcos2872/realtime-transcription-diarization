"""Testes de domínio: regras de locutor (puro, sem GPU/rede)."""

from src.domain.services.speaker_rules import (
    GENERIC_SPEAKER,
    MIC_SPEAKER,
    SYSTEM_SPEAKER,
    apply_stop_rules,
    normalize_speaker,
    person_label,
)


def test_person_label_base1():
    assert person_label(1) == "Pessoa 1"
    assert person_label(3) == "Pessoa 3"


def test_normalize_fallback():
    assert normalize_speaker("") == GENERIC_SPEAKER
    assert normalize_speaker(None) == GENERIC_SPEAKER
    assert normalize_speaker("Pessoa 2") == "Pessoa 2"


def test_mic_sempre_eu():
    segs = [{"speaker": "Pessoa 9", "text": "oi", "tStart": 0.0, "tEnd": 1.0}]
    assert apply_stop_rules(segs, "mic", True)[0]["speaker"] == MIC_SPEAKER
    assert apply_stop_rules(segs, "mic", False)[0]["speaker"] == MIC_SPEAKER


def test_system_sem_diarize_vira_sistema():
    segs = [{"speaker": "Locutor", "text": "x", "tStart": 0.0, "tEnd": 1.0}]
    assert apply_stop_rules(segs, "system", False)[0]["speaker"] == SYSTEM_SPEAKER


def test_system_com_diarize_preserva():
    segs = [{"speaker": "Pessoa 2", "text": "x", "tStart": 0.0, "tEnd": 1.0}]
    assert apply_stop_rules(segs, "system", True)[0]["speaker"] == "Pessoa 2"


def test_nao_mutar_original():
    segs = [{"speaker": "Locutor", "text": "x", "tStart": 0.0, "tEnd": 1.0}]
    out = apply_stop_rules(segs, "mic", False)
    assert segs[0]["speaker"] == "Locutor"
    assert out[0]["speaker"] == MIC_SPEAKER
