#!/usr/bin/env python3
"""
Cliente de teste WEBSOCKET para streaming com formato estilo Azure STT.

Conecta em WS /speech/stream, envia config + PCM binário com pacing
em tempo real e exibe hypothesis (interim) e phrase (final).

Uso:
  uv run test_ws.py --audio ~/audio.wav --url http://servidor:4321 --diarize
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

SERVER_URL = "http://192.168.3.81:4321"
AUDIO_PATH = "/home/marcos/Documentos/rec/2026-07-29_15-00-56/audio.wav"
LANGUAGE = "pt"
LOCALE = "pt-BR"
CHUNK_MS = 500
PACE_ENABLED = True


def read_wav_pcm(path: str) -> bytes:
    """Lê WAV 16kHz 16-bit e retorna PCM mono."""
    import wave
    import array
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            sys.exit(f"✗ WAV deve ser 16kHz 16-bit, tem "
                     f"{wf.getframerate()}Hz/{wf.getsampwidth()*8}-bit")
        raw = wf.readframes(wf.getnframes())
        if wf.getnchannels() > 1:
            samples = array.array("h", raw)
            mono = array.array("h")
            nch = wf.getnchannels()
            for i in range(0, len(samples), nch):
                mono.append(sum(samples[i:i + nch]) // nch)
            raw = mono.tobytes()
            print(f"  Convertido de {nch} canais para mono")
        return raw


async def run(url: str, audio_path: str, language: str, locale: str,
              diarize: bool, chunk_ms: int):
    """Executa o fluxo WS completo e retorna as phrases finais."""
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.rstrip("/") + "/speech/stream"
    pcm = read_wav_pcm(audio_path)
    print(f"  Duração: {len(pcm) / 32000:.1f}s | Servidor: {ws_url}")

    phrases: list[dict] = []
    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "config", "language": language, "locale": locale,
            "diarize": diarize, "interimIntervalMs": 1000, "words": True,
        }))
        msg = json.loads(await ws.recv())
        assert msg.get("path") == "turn.start", msg
        print("✓ turn.start")

        chunk_bytes = int(16000 * chunk_ms / 1000) * 2
        chunks = [pcm[i:i + chunk_bytes]
                  for i in range(0, len(pcm), chunk_bytes)]

        async def sender():
            t0 = asyncio.get_event_loop().time()
            for i, ch in enumerate(chunks):
                await ws.send(ch)
                if PACE_ENABLED:
                    expected = (i + 1) * chunk_ms / 1000.0
                    wait = expected - (asyncio.get_event_loop().time() - t0)
                    if wait > 0:
                        await asyncio.sleep(wait)
            await ws.send(json.dumps({"type": "end"}))
            print(f"\n✓ {len(chunks)} chunks enviados + end")

        async def receiver():
            async for raw in ws:
                m = json.loads(raw)
                path = m.get("path")
                if path == "speech.hypothesis":
                    print(f"  … interim [{m['Offset']/1e7:.1f}s]: {m['Text']}")
                elif path == "speech.phrase":
                    phrases.append(m)
                    spk = f" | speaker={m['Speaker']}" if "Speaker" in m else ""
                    print(f"  ◆ final [{m['Offset']/1e7:.1f}s]: "
                          f"{m['DisplayText']}{spk}")
                elif path == "turn.end":
                    print("✓ turn.end")
                    break
                elif path == "error":
                    print(f"✗ erro: {m}")
                    break

        await asyncio.gather(sender(), receiver())
    return phrases


def main():
    """Ponto de entrada: argumentos CLI e execução do teste WS."""
    parser = argparse.ArgumentParser(description="Cliente WebSocket Azure-style")
    parser.add_argument("--audio", default=AUDIO_PATH)
    parser.add_argument("--url", default=SERVER_URL)
    parser.add_argument("--lang", default=LANGUAGE)
    parser.add_argument("--locale", default=LOCALE)
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"✗ Arquivo não encontrado: {args.audio}")

    print("=" * 60)
    print("   STT API — Cliente WebSocket (formato Azure)".center(60))
    print("=" * 60)
    phrases = asyncio.run(run(args.url, args.audio, args.lang,
                              args.locale, args.diarize, args.chunk_ms))
    print(f"\nTotal: {len(phrases)} phrases finais")


if __name__ == "__main__":
    main()
