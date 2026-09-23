#!/usr/bin/env python3
"""
Cliente de teste para API de transcrição.
Uso: python test_api.py [--audio caminho.wav] [--url http://localhost:8000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# ── Configurações padrão (edite ou use --audio / --url) ──
SERVER_URL = "http://100.68.95.21:4321"
AUDIO_PATH = "/home/marcos/Documentos/rec/2026-07-29_09-02-01/system.wav"
LANGUAGE = "pt"
# ─────────────────────────────────────────────────────────


def transcribe(audio_path: str, server_url: str, language: str = "pt", diarize: bool = False) -> dict:
    """Envia áudio para /transcribe e retorna a transcrição."""
    print(f"\n▶ Enviando {audio_path} para {server_url}/transcribe ...")

    with open(audio_path, "rb") as f:
        files = {"audio": (Path(audio_path).name, f, "audio/wav")}
        data = {"language": language, "diarize": str(diarize).lower()}

        t0 = time.time()
        resp = requests.post(f"{server_url}/transcribe", files=files, data=data, timeout=600)
        elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"✗ Erro {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)

    result = resp.json()
    print(f"✓ Transcrição concluída em {elapsed:.1f}s")
    print(f"  Sessão:    {result['sessionId']}")
    print(f"  Duração:   {result['durationSec']:.1f}s")
    print(f"  Idioma:    {result['language']}")
    print(f"  Locutores: {', '.join(result['participants'])}")
    print(f"  Segmentos: {len(result['segments'])}")
    return result


def refine(transcription: dict, server_url: str) -> dict:
    """Envia transcrição para /refine e retorna o resultado refinado."""
    print(f"\n▶ Refinando via {server_url}/refine ...")

    t0 = time.time()
    resp = requests.post(
        f"{server_url}/refine",
        json={"transcription": transcription},
        timeout=300,
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"✗ Erro no refinamento {resp.status_code}: {resp.text[:200]}")
        return transcription  # fallback: retorna original

    result = resp.json()
    print(f"✓ Refinamento concluído em {elapsed:.1f}s")
    print(f"  Modelo: {result.get('model', '?')}")
    return result.get("refined", transcription)


def show_result(transcription: dict, refined: dict | None = None):
    """Exibe a transcrição formatada no terminal."""
    data = refined or transcription
    segments = data.get("segments", [])

    print("\n" + "=" * 60)
    print("TRANSCRIÇÃO".center(60))
    print("=" * 60)

    for seg in segments:
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "")
        ts = seg.get("tStart", 0)
        te = seg.get("tEnd", 0)
        print(f"\n  [{speaker}] ({ts:.1f}s – {te:.1f}s)")
        print(f"  {text}")

    print("\n" + "=" * 60)

    if refined:
        print("✓ Transcrição refinada exibida acima")
    else:
        print("ℹ Transcrição bruta (refinamento não foi aplicado)")

    print("=" * 60 + "\n")


def health_check(server_url: str) -> bool:
    """Verifica se o servidor está respondendo."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            print(f"✓ Servidor OK: {info['status']}")
            print(f"  GPUs:     {', '.join(info['gpus'])}")
            print(f"  Whisper:  {'carregado' if info['whisperLoaded'] else 'não carregado'}")
            print(f"  Refine:   {info['refineEndpoint']}")
            print(f"  Sessões:  {info['activeSessions']}")
            return True
        else:
            print(f"✗ Servidor respondeu com status {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"✗ Não foi possível conectar em {server_url}")
        print(f"  Verifique se o servidor está rodando")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cliente de teste para API de transcrição")
    parser.add_argument("--audio", default=AUDIO_PATH, help="Caminho para arquivo WAV")
    parser.add_argument("--url", default=SERVER_URL, help="URL do servidor STT API")
    parser.add_argument("--lang", default=LANGUAGE, help="Idioma (pt, en, etc)")
    parser.add_argument("--no-refine", action="store_true", help="Pular refinamento")
    parser.add_argument("--diarize", action="store_true", help="Ativar diarização (identificar locutores)")
    parser.add_argument("--only-health", action="store_true", help="Só testar health check")
    args = parser.parse_args()

    print("=" * 60)
    print("   STT API — Cliente de Teste".center(60))
    print("=" * 60)
    print(f"  Servidor: {args.url}")
    print(f"  Áudio:    {args.audio}")
    print(f"  Idioma:   {args.lang}")
    print(f"  Diarizar: {'sim' if args.diarize else 'não'}")

    # Health check
    print()
    if not health_check(args.url):
        sys.exit(1)

    if args.only_health:
        return

    # Verifica se o arquivo de áudio existe
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"\n✗ Arquivo não encontrado: {args.audio}")
        sys.exit(1)

    # Transcrição
    transcription = transcribe(str(audio_path), args.url, args.lang, args.diarize)

    # Refinamento
    refined = None
    if not args.no_refine:
        refined = refine(transcription, args.url)

    # Exibe resultado
    show_result(transcription, refined)


if __name__ == "__main__":
    main()
