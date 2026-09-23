"use client";

import { useEffect, useRef, useState } from "react";
import { toWsUrl } from "../lib/api";
import { MicCapture } from "../lib/audio";
import type { TabProps } from "./common";

interface Phrase {
  RecognitionStatus: string;
  DisplayText: string;
  Offset: number;
  Duration: number;
  Speaker?: number;
  Locale?: string;
}

/** Aba 3: streaming WebSocket com formato estilo Azure STT. */
export default function WsTab({ settings, addLog }: TabProps) {
  const [connected, setConnected] = useState(false);
  const [hyps, setHyps] = useState<string[]>([]);
  const [phrases, setPhrases] = useState<Phrase[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const capRef = useRef<MicCapture | null>(null);

  async function start() {
    setHyps([]);
    setPhrases([]);
    const ws = new WebSocket(toWsUrl(settings.baseUrl));
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen = async () => {
      ws.send(JSON.stringify({
        type: "config",
        language: settings.language,
        locale: settings.locale,
        diarize: settings.diarize,
        interimIntervalMs: 1000,
        words: true,
      }));
      const cap = new MicCapture();
      cap.onChunk = (c) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(c.pcm16.buffer as ArrayBuffer);
      };
      await cap.start();
      capRef.current = cap;
      setConnected(true);
      addLog("WS conectado + config enviada");
    };
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data as string);
      if (m.path === "turn.start") addLog("WS: turn.start");
      else if (m.path === "speech.hypothesis") setHyps((o) => [...o.slice(-19), m.Text]);
      else if (m.path === "speech.phrase") {
        setPhrases((o) => [...o, m as Phrase]);
        addLog("WS: phrase", { text: m.DisplayText, Speaker: m.Speaker });
      } else if (m.path === "turn.end") addLog("WS: turn.end");
      else if (m.path === "error") addLog("✗ WS", m);
    };
    ws.onclose = () => {
      setConnected(false);
      addLog("WS fechado");
    };
    ws.onerror = () => addLog("✗ WS: erro de conexão");
  }

  async function stop() {
    await capRef.current?.stop();
    capRef.current = null;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end" }));
      addLog("WS: end enviado (aguardando phrases + turn.end)");
    }
  }

  useEffect(() => () => {
    capRef.current?.stop();
    wsRef.current?.close();
  }, []);

  return (
    <div className="card">
      <h3>WebSocket — /speech/stream (formato Azure)</h3>
      <div className="row">
        {!connected
          ? <button onClick={start}>● Conectar e transmitir do mic</button>
          : <button onClick={stop}>■ Enviar end</button>}
      </div>
      {connected && <p className="warn">● transmitindo…</p>}
      {hyps.length > 0 && (
        <>
          <h4>Interim (hypothesis)</h4>
          <p className="hyp">{hyps[hyps.length - 1]}</p>
        </>
      )}
      {phrases.length > 0 && (
        <>
          <h4>Finais ({phrases.length})</h4>
          <ul className="segs">
            {phrases.map((p, i) => (
              <li key={i} className="seg spk-0">
                <span className="seg-head">
                  [{p.RecognitionStatus}
                  {p.Speaker !== undefined ? ` | speaker ${p.Speaker}` : ""}
                  ] ({(p.Offset / 1e7).toFixed(1)}s)
                </span>
                <span className="seg-text">{p.DisplayText}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
