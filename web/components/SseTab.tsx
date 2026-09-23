"use client";

import { useEffect, useRef, useState } from "react";
import {
  newSessionId,
  streamEvents,
  streamSendAudio,
  streamStart,
  streamStop,
  type PartialResult,
  type TranscriptionResult,
} from "../lib/api";
import { MicCapture, int16ToBase64 } from "../lib/audio";
import SegmentList from "./SegmentList";
import type { TabProps } from "./common";

/** Aba 2: streaming SSE (start → chunks base64 → parciais → stop). */
export default function SseTab({ settings, addLog, onResult }: TabProps) {
  const [running, setRunning] = useState(false);
  const [partials, setPartials] = useState<PartialResult[]>([]);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const capRef = useRef<MicCapture | null>(null);
  const closeEsRef = useRef<(() => void) | null>(null);
  const seqRef = useRef(0);
  const sidRef = useRef("");

  async function start() {
    const sid = newSessionId();
    sidRef.current = sid;
    seqRef.current = 0;
    setPartials([]);
    setResult(null);
    addLog(`POST /stream/${sid} start (diarize=${settings.diarize})`);
    await streamStart(settings.baseUrl, sid, settings.language, settings.diarize,
      settings.minSpeakers);
    closeEsRef.current = streamEvents(
      settings.baseUrl,
      sid,
      (p) => setPartials((old) =>
        old.some((o) => o.tStart === p.tStart && o.text === p.text) ? old : [...old, p]),
      () => addLog("SSE: sessão fechada"),
      (m) => addLog(`SSE: ${m}`),
    );
    const cap = new MicCapture();
    cap.onChunk = (c) => {
      const seq = seqRef.current++;
      streamSendAudio(settings.baseUrl, sid, seq, int16ToBase64(c.pcm16)).catch((e) =>
        addLog(`✗ chunk ${seq}`, String(e)));
    };
    await cap.start();
    capRef.current = cap;
    setRunning(true);
  }

  async function stop() {
    setRunning(false);
    await capRef.current?.stop();
    capRef.current = null;
    closeEsRef.current?.();
    closeEsRef.current = null;
    try {
      addLog(`POST /stream/${sidRef.current} stop`);
      const r = await streamStop(settings.baseUrl, sidRef.current, settings.language);
      setResult(r);
      onResult(r);
      addLog("→ stop", { segments: r.segments.length, participants: r.participants });
    } catch (e) {
      addLog("✗ stop", String(e));
    }
  }

  useEffect(() => () => {
    closeEsRef.current?.();
    capRef.current?.stop();
  }, []);

  return (
    <div className="card">
      <h3>Streaming SSE — /stream/{"{id}"}</h3>
      <div className="row">
        {!running
          ? <button onClick={start}>● Iniciar streaming do mic</button>
          : <button onClick={stop}>■ Parar e transcrever</button>}
      </div>
      {running && <p className="warn">● transmitindo… parciais abaixo</p>}
      {partials.length > 0 && (
        <>
          <h4>Parciais ({partials.length})</h4>
          <SegmentList segments={partials.map((p) => ({
            speaker: p.speaker, text: p.text, tStart: p.tStart, tEnd: p.tEnd,
          }))} />
        </>
      )}
      {result && (
        <>
          <h4>Final — locutores: {result.participants.join(", ")}</h4>
          <SegmentList segments={result.segments} />
        </>
      )}
    </div>
  );
}
