"use client";

import { useRef, useState } from "react";
import { transcribeBatch, type TranscriptionResult } from "../lib/api";
import { MicCapture, concatInt16, encodeWav, fileToWav16kMono } from "../lib/audio";
import SegmentList from "./SegmentList";
import type { TabProps } from "./common";

/** Aba 1: transcrição por lote (POST /transcribe) com diarização opcional. */
export default function BatchTab({ settings, addLog, onResult }: TabProps) {
  const [recording, setRecording] = useState(false);
  const [wavUrl, setWavUrl] = useState<string | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const capRef = useRef<MicCapture | null>(null);
  const partsRef = useRef<Int16Array[]>([]);
  const wavRef = useRef<Blob | null>(null);

  function setWav(blob: Blob) {
    wavRef.current = blob;
    setWavUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(blob);
    });
  }

  async function toggleRecord() {
    if (recording) {
      const rest = await capRef.current!.stop();
      capRef.current = null;
      setRecording(false);
      const all = concatInt16([...partsRef.current, rest]);
      partsRef.current = [];
      setWav(encodeWav(all));
      addLog(`mic gravado: ${(all.length / 16000).toFixed(1)}s`);
    } else {
      const cap = new MicCapture();
      partsRef.current = [];
      cap.onChunk = (c) => partsRef.current.push(c.pcm16);
      await cap.start();
      capRef.current = cap;
      setRecording(true);
    }
  }

  async function onFile(f: File | undefined) {
    if (!f) return;
    const wav = await fileToWav16kMono(f);
    setWav(wav);
    addLog(`upload: ${f.name} → WAV 16k mono (${(wav.size / 1024).toFixed(0)} KB)`);
  }

  async function send() {
    if (!wavRef.current || busy) return;
    setBusy(true);
    try {
      addLog(`POST /transcribe (diarize=${settings.diarize})`);
      const r = await transcribeBatch(
        settings.baseUrl, wavRef.current, settings.language, settings.diarize);
      setResult(r);
      onResult(r);
      addLog("→ /transcribe", { segments: r.segments.length, participants: r.participants });
    } catch (e) {
      addLog("✗ /transcribe", String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h3>Batch — POST /transcribe</h3>
      <div className="row">
        <button onClick={toggleRecord}>{recording ? "■ Parar mic" : "● Gravar do mic"}</button>
        <label className="file">
          ou enviar arquivo
          <input type="file" accept="audio/*" hidden
            onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        <button onClick={send} disabled={!wavUrl || busy || recording}>
          {busy ? "Transcrevendo…" : "Transcrever"}
        </button>
      </div>
      {recording && <p className="warn">● gravando…</p>}
      {wavUrl && <audio controls src={wavUrl} className="audio" />}
      {result && (
        <>
          <p className="muted">
            {result.segments.length} segmentos · {result.durationSec.toFixed(1)}s ·
            locutores: {result.participants.join(", ")}
          </p>
          <SegmentList segments={result.segments} />
        </>
      )}
    </div>
  );
}
