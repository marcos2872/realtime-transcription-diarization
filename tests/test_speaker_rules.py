"""Testes de domínio: regras de locutor (puro, sem GPU/rede)."""

from src.domain.services.speaker_rules import (
    GENERIC_SPEAKER,
    MIC_SPEAKER,
    SYSTEM_SPEAKER,
    apply_stop_rules,
    match_raw_label,
    normalize_speaker,
    overlap,
    person_label,
    resolve_stable_labels,
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


def test_overlap():
    assert overlap(0, 10, 5, 15) == 5
    assert overlap(0, 5, 5, 10) == 0
    assert overlap(0, 5, 10, 15) == 0


def test_match_raw_label_meio_e_proximo():
    dz = [{"speaker": "S0", "tStart": 0.0, "tEnd": 10.0}]
    assert match_raw_label(2.0, 4.0, dz) == "S0"
    assert match_raw_label(10.0, 12.0, dz) == "S0"
    assert match_raw_label(0.0, 1.0, []) is None


def test_resolve_estavel_sob_permutacao():
    # Flush 1: S0 = pessoa A (0-10). Vira Pessoa 1.
    dz1 = [{"speaker": "S0", "tStart": 0.0, "tEnd": 10.0}]
    m1, idx = resolve_stable_labels(dz1, [], 1)
    assert m1 == {"S0": "Pessoa 1"}
    timeline = [{"speaker": "Pessoa 1", "tStart": 0.0, "tEnd": 10.0}]

    # Flush 2: pyannote permuta (S1 = A 0-10, S0 = B 10-20).
    # S0 agora é voz nova (sem sobreposição) → Pessoa 2.
    dz2 = [
        {"speaker": "S1", "tStart": 0.0, "tEnd": 10.0},
        {"speaker": "S0", "tStart": 10.0, "tEnd": 20.0},
    ]
    m2, idx = resolve_stable_labels(dz2, timeline, idx)
    assert m2 == {"S1": "Pessoa 1", "S0": "Pessoa 2"}
    timeline.append({"speaker": "Pessoa 2", "tStart": 10.0, "tEnd": 15.0})

    # Flush 3: permuta de volta (S0 = A, S1 = B). Estabilidade mantida
    # (mapeamento fresco daria S1 → Pessoa 1, errado para a voz B).
    dz3 = [
        {"speaker": "S0", "tStart": 0.0, "tEnd": 10.0},
        {"speaker": "S1", "tStart": 10.0, "tEnd": 20.0},
    ]
    m3, idx = resolve_stable_labels(dz3, timeline, idx)
    assert m3 == {"S0": "Pessoa 1", "S1": "Pessoa 2"}
    assert idx == 3
