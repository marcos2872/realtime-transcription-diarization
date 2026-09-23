#!/usr/bin/env python3
"""
Cliente de teste STREAMING para API de transcrição.

Envia o áudio em chunks simulando fluxo ao vivo, enquanto exibe
a transcrição PARCIAL em tempo real via SSE (Server-Sent Events).

Taxa de envio:
  Cada chunk representa chunk_ms de áudio e é espaçado
  por exatamente chunk_ms/1000 segundos (pacing real-time).
  A taxa de bytes é sempre 32000 bytes/s (PCM 16kHz 16bits mono).

Uso:
  python test_streaming.py [--audio caminho.wav] [--url http://servidor:4321]
  python test_streaming.py --only-health
  python test_streaming.py --help
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import requests


# ═══════════════════════════════════════════════════════════
#  CONFIGURAÇÕES EDITÁVEIS
# ═══════════════════════════════════════════════════════════

# ── Servidor ──
SERVER_URL = "http://192.168.3.81:4321"      # URL do servidor STT API

# ── Áudio ──
AUDIO_PATH = "/home/marcos/Documentos/rec/2026-07-29_15-00-56/audio.wav"  # WAV 16kHz mono 16-bit

# ── Transcrição ──
LANGUAGE = "pt"                                # Idioma (pt, en, es, etc)
DIARIZE = True                                 # Ativar diarização pyannote
NO_REFINE = False                               # Pular refinamento via LLM

# ── Streaming ──
CHUNK_MS = 500                                  # ms por chunk (~8KB cada)
PACE_ENABLED = True                             # Pacing em tempo real

# ── Timeouts (segundos) ──
FLUSH_INTERVAL = 3                              # Intervalo entre transcrições parciais via SSE
SSE_TIMEOUT = 120                               # Timeout da conexão SSE
STOP_TIMEOUT = 600                              # Timeout da transcrição final
CHUNK_TIMEOUT = 30                              # Timeout por chunk enviado
SESSION_START_TIMEOUT = 10                      # Timeout para iniciar sessão
HEALTH_TIMEOUT = 5                              # Timeout do health check

# ── Avançado ──
PACE_TOLERANCE_MS = 10                          # ms de tolerância no pacing
# ═══════════════════════════════════════════════════════════

# ── SSE parser minimal (sem dependência externa) ──

class _SSEEvent:
    """Evento SSE simples: event (tipo) + data (payload JSON)."""

    def __init__(self):
        self.event = ""
        self.data = ""


def _sse_iter_events(response):
    """Gera eventos SSE um por um à medida que chegam do servidor.

    Args:
        response: Resposta HTTP com stream=True.

    Yields:
        _SSEEvent a cada bloco event/data completo.
    """
    current = _SSEEvent()
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if current.event or current.data:
                yield current
            current = _SSEEvent()
            continue
        if line.startswith("event: "):
            current.event = line[7:]
        elif line.startswith("data: "):
            current.data = line[6:]


def read_wav_pcm(path: str) -> tuple[bytes, int, int]:
    """Lê um arquivo WAV e retorna os dados PCM brutos + metadados.

    Valida que o arquivo é 16kHz, 16-bit (sampwidth=2).
    Suporta mono e estéreo (conversão manual é feita em stream_audio).

    Args:
        path: Caminho para o arquivo .wav.

    Returns:
        Tupla (raw_pcm_bytes, sample_rate, num_channels).

    Raises:
        SystemExit se o formato for inválido.
    """
    import wave
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            print(f"✗ WAV deve ser 16-bit, tem {wf.getsampwidth()*8}-bit")
            sys.exit(1)
        if wf.getframerate() != 16000:
            print(f"✗ WAV deve ser 16kHz, tem {wf.getframerate()}Hz")
            sys.exit(1)
        raw = wf.readframes(wf.getnframes())
        return raw, wf.getframerate(), wf.getnchannels()


# ── Estado compartilhado entre threads ──
_live_segments: dict[str, list[dict]] = {}  # channel -> [segments]
_live_lock = threading.Lock()
_sse_error: str | None = None


def _sse_listener(server_url: str, session_id: str):
    """Thread que escuta eventos SSE e atualiza _live_segments.

    Executada em background durante o streaming. Processa eventos:
    - "partial": segmento de transcrição parcial (exibe em tempo real)
    - "heartbeat": keepalive do servidor; se "closed", encerra
    - "error": erro reportado pelo servidor

    Args:
        server_url: URL base do servidor STT API.
        session_id: ID da sessão de streaming.
    """
    global _live_segments, _sse_error
    url = f"{server_url}/stream/{session_id}/events"
    try:
        resp = requests.get(url, stream=True, timeout=SSE_TIMEOUT)
        for event in _sse_iter_events(resp):
            if event.event == "partial":
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                channel = data.get("channel", "system")
                seg = {
                    "speaker": data.get("speaker", "Locutor"),
                    "text": data.get("text", ""),
                    "tStart": data.get("tStart", 0.0),
                    "tEnd": data.get("tEnd", 0.0),
                }
                with _live_lock:
                    if channel not in _live_segments:
                        _live_segments[channel] = []
                    if seg not in _live_segments[channel]:
                        _live_segments[channel].append(seg)
                        _print_live_segment(seg)
            elif event.event == "heartbeat":
                if event.data == "closed":
                    break
            elif event.event == "error":
                _sse_error = event.data
                break
    except Exception as exc:
        if not _sse_error:
            _sse_error = str(exc)


def _print_live_segment(seg: dict):
    """Exibe um segmento recém-chegado no terminal durante o streaming.

    Chamado pela thread SSE listener assim que um evento "partial"
    é recebido. O print aparece intercalado com a barra de progresso.

    Args:
        seg: Dict com speaker, text, tStart, tEnd.
    """
    speaker = seg.get("speaker", "?")
    text = seg.get("text", "")
    ts = seg.get("tStart", 0)
    te = seg.get("tEnd", 0)
    print(f"\n  🎤 [{speaker}] ({ts:.1f}s – {te:.1f}s)")
    print(f"     {text}")


def _print_divider(char: str = "─", width: int = 60):
    """Imprime uma linha divisória no terminal.

    Args:
        char: Caractere para a linha.
        width: Largura em caracteres.
    """
    print(char * width)


def stream_audio(
    audio_path: str,
    server_url: str,
    language: str = "pt",
    diarize: bool = False,
    chunk_ms: int = 500,
):
    """Envia áudio em streaming via API e mostra transcrição AO VIVO.

    Fluxo:
      1. Inicia sessão no servidor (POST /stream/{id} com action=start)
      2. Conecta SSE listener em background (recebe parciais)
      3. Envia chunks com pacing em tempo real (POST /stream/{id}/audio)
      4. Finaliza sessão (POST /stream/{id} com action=stop)
      5. Exibe transcrição final consolidada

    O pacing simula o fluxo de áudio ao vivo: cada chunk de `chunk_ms` ms
    é enviado com intervalo de exatamente `chunk_ms/1000` segundos.
    Desligue com PACE_ENABLED=False para enviar o mais rápido possível.

    Args:
        audio_path: Caminho do arquivo WAV (16kHz, 16-bit, mono).
        server_url: URL base do servidor STT API.
        language: Código do idioma (pt, en, es).
        diarize: Se True, envia flag para diarização.
        chunk_ms: Tamanho de cada chunk em milissegundos.

    Returns:
        Dict com a transcrição final (sessionId, segments, etc).
    """
    session_id = uuid.uuid4().hex[:12]
    print(f"Sessão: {session_id}")

    # Lê WAV e converte para mono se necessário
    raw_pcm, sr, channels = read_wav_pcm(audio_path)
    if channels > 1:
        import array
        samples = array.array("h", raw_pcm)
        mono = array.array("h")
        for i in range(0, len(samples), channels):
            ch = samples[i:i + channels]
            mono.append(sum(ch) // channels)
        raw_pcm = mono.tobytes()
        print(f"  Convertido de {channels} canais para mono")

    total_samples = len(raw_pcm) // 2
    duration_sec = total_samples / 16000
    print(f"  Duração: {duration_sec:.1f}s ({total_samples} samples)")

    # 1. Inicia sessão
    print("\n▶ Iniciando sessão de streaming...")
    resp = requests.post(
        f"{server_url}/stream/{session_id}",
        json={
            "sessionId": session_id, "action": "start",
            "language": language, "channels": ["system"],
            "diarize": diarize,
        },
        timeout=SESSION_START_TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"✗ Erro ao iniciar sessão: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    print("✓ Sessão iniciada")

    # 2. Inicia SSE listener em background
    print("\n▶ Conectando SSE para transcrição ao vivo...")
    sse_thread = threading.Thread(
        target=_sse_listener,
        args=(server_url, session_id),
        daemon=True,
    )
    sse_thread.start()
    time.sleep(0.5)  # dá tempo de conectar

    # 3. Envia áudio em chunks
    chunk_samples = int(sr * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    seq = 0
    total_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
    t_start = time.time()

    pace_label = "com pacing em tempo real" if PACE_ENABLED else "sem pacing (mais rápido)"
    print(f"\n▶ Enviando {total_chunks} chunks ({chunk_ms}ms cada) {pace_label}...\n")

    chunk_duration_s = chunk_ms / 1000.0
    t_chunk_start = time.time()

    for offset in range(0, len(raw_pcm), chunk_bytes):
        pcm_chunk = raw_pcm[offset:offset + chunk_bytes]
        if len(pcm_chunk) < 4:
            seq += 1
            continue
        pcm_b64 = base64.b64encode(pcm_chunk).decode("ascii")

        resp = requests.post(
            f"{server_url}/stream/{session_id}/audio",
            json={
                "sessionId": session_id,
                "channel": "system",
                "seq": seq,
                "data": pcm_b64,
            },
            timeout=CHUNK_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"\n✗ Erro no chunk {seq}: {resp.status_code}")
            break

        seq += 1
        progress = min(offset / len(raw_pcm) * 100, 100)
        elapsed = time.time() - t_start
        print(
            f"\r  [{progress:5.1f}%] chunk {seq}/{total_chunks} "
            f"({elapsed:.1f}s)   ",
            end="", flush=True,
        )

        # Pacing: espera o tempo real do áudio (simula fluxo ao vivo)
        if PACE_ENABLED:
            expected_elapsed = seq * chunk_duration_s
            actual_elapsed = time.time() - t_chunk_start
            sleep_needed = expected_elapsed - actual_elapsed
            if sleep_needed > (PACE_TOLERANCE_MS / 1000.0):
                time.sleep(sleep_needed)

    print(f"\n\n✓ Streaming concluído: {seq} chunks em {time.time() - t_start:.1f}s")

    # 4. Mostra resumo parcial antes de finalizar
    with _live_lock:
        total_live = sum(len(segs) for segs in _live_segments.values())
    print(f"  Segmentos ao vivo recebidos: {total_live}")

    if _sse_error:
        print(f"  ⚠ Erro no SSE: {_sse_error}")

    # 5. Finaliza sessão (dispara transcrição completa)
    print("\n▶ Finalizando sessão e transcrevendo áudio completo...")
    t0 = time.time()
    resp = requests.post(
        f"{server_url}/stream/{session_id}",
        json={
            "sessionId": session_id, "action": "stop",
            "language": language, "channels": ["system"],
        },
        timeout=STOP_TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"✗ Erro ao finalizar: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)

    result = resp.json()
    elapsed = time.time() - t0
    print(f"✓ Transcrição final em {elapsed:.1f}s")
    print(f"  Sessão:    {result['sessionId']}")
    print(f"  Duração:   {result['durationSec']:.1f}s")
    print(f"  Idioma:    {result['language']}")
    print(f"  Locutores: {', '.join(result['participants'])}")
    print(f"  Segmentos: {len(result['segments'])}")

    return result


def refine(transcription: dict, server_url: str) -> dict:
    """Envia transcrição para /refine e retorna o resultado refinado pelo LLM.

    O servidor usa Qwen 2.5 7B (ou modelo configurado) via llama.cpp
    para corrigir pontuação, maiúsculas, fluência e remover hesitações.

    Args:
        transcription: Dict com a transcrição (sessionId, segments, etc).
        server_url: URL base do servidor STT API.

    Returns:
        Dict com a transcrição refinada (campo "refined" na resposta)
        ou a transcrição original se o refinamento falhar.
    """
    print(f"\n▶ Refinando via {server_url}/refine ...")

    t0 = time.time()
    resp = requests.post(
        f"{server_url}/refine",
        json={"transcription": transcription},
        timeout=STOP_TIMEOUT,
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"✗ Erro no refinamento {resp.status_code}: {resp.text[:200]}")
        return transcription

    result = resp.json()
    print(f"✓ Refinamento concluído em {elapsed:.1f}s")
    print(f"  Modelo: {result.get('model', '?')}")
    return result.get("refined", transcription)


def show_transcription(data: dict, title: str = "TRANSCRIÇÃO"):
    """Exibe a transcrição formatada no terminal.

    Cada segmento mostra: [Locutor] (tStart – tEnd) + texto.

    Args:
        data: Dict com chave "segments" (lista de segmentos).
        title: Título opcional para o bloco (ex: "TRANSCRIÇÃO FINAL").
    """
    segments = data.get("segments", [])

    _print_divider("=")
    print(title.center(60))
    _print_divider("=")

    for seg in segments:
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "")
        ts = seg.get("tStart", 0)
        te = seg.get("tEnd", 0)
        print(f"\n  [{speaker}] ({ts:.1f}s – {te:.1f}s)")
        print(f"  {text}")

    _print_divider("=")
    print(f"Total: {len(segments)} segmentos")
    _print_divider("=")


def health_check(server_url: str) -> bool:
    """Verifica se o servidor STT API está respondendo.

    Faz GET /health e exibe status, GPUs disponíveis,
    se Whisper está carregado e endpoint de refinamento.

    Args:
        server_url: URL base do servidor.

    Returns:
        True se servidor respondeu com 200, False caso contrário.
    """
    try:
        resp = requests.get(f"{server_url}/health", timeout=HEALTH_TIMEOUT)
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
    """Ponto de entrada: processa argumentos e executa o teste de streaming.

    Uso:
        python test_streaming.py
            (usa SERVER_URL, AUDIO_PATH, LANGUAGE das configs)
        python test_streaming.py --url http://192.168.3.81:4321 --audio ~/audio.wav --lang pt
        python test_streaming.py --only-health          # só health check
        python test_streaming.py --no-refine             # pula refinamento LLM
        python test_streaming.py --diarize               # ativa diarização
        python test_streaming.py --chunk-ms 200          # chunks de 200ms
    """
    parser = argparse.ArgumentParser(description="Cliente STREAMING para API de transcrição")
    parser.add_argument("--audio", default=AUDIO_PATH, help="Caminho para arquivo WAV (16kHz mono)")
    parser.add_argument("--url", default=SERVER_URL, help="URL do servidor STT API")
    parser.add_argument("--lang", default=LANGUAGE, help="Idioma (pt, en, etc)")
    parser.add_argument("--diarize", action="store_true", help="Ativar diarização")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS, help="Tamanho do chunk em ms")
    parser.add_argument("--no-refine", action="store_true", help="Pular refinamento")
    parser.add_argument("--only-health", action="store_true", help="Só testar health check")
    args = parser.parse_args()

    print("=" * 60)
    print("   STT API — Cliente de Streaming".center(60))
    print("=" * 60)
    print(f"  Servidor: {args.url}")
    print(f"  Áudio:    {args.audio}")
    print(f"  Idioma:   {args.lang}")
    print(f"  Diarizar: {'sim' if args.diarize else 'não'}")
    print(f"  Chunk:    {args.chunk_ms}ms")
    print(f"  SSE:      resultados parciais a cada ~{FLUSH_INTERVAL}s")
    print(f"  Pacing:   {'sim' if PACE_ENABLED else 'não'}")

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

    # Streaming com transcrição ao vivo
    print(f"\n{'─' * 60}")
    print("   TRANSCRIÇÃO AO VIVO".center(60))
    print(f"{'─' * 60}")
    print("   (segmentos aparecem conforme o servidor processa)")
    print(f"{'─' * 60}\n")

    transcription = stream_audio(
        str(audio_path), args.url, args.lang, args.diarize, args.chunk_ms,
    )

    # Exibe transcrição final completa
    show_transcription(transcription, "TRANSCRIÇÃO FINAL")

    # Refinamento
    if not args.no_refine:
        refined = refine(transcription, args.url)
        show_transcription(refined, "TRANSCRIÇÃO REFINADA")
    else:
        print("ℹ Pulado refinamento (--no-refine)")


if __name__ == "__main__":
    main()
