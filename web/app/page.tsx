"use client";

import { useCallback, useEffect, useState } from "react";
import BatchTab from "../components/BatchTab";
import HealthBar from "../components/HealthBar";
import NetLog, { type NetEntry } from "../components/NetLog";
import RefineTab from "../components/RefineTab";
import SseTab from "../components/SseTab";
import WsTab from "../components/WsTab";
import { now, type Settings } from "../components/common";
import { getHealth, type HealthResponse, type TranscriptionResult } from "../lib/api";

type Tab = "batch" | "sse" | "ws" | "refine";

/** Página única com 4 abas (batch, SSE, WebSocket, refine) + health + log. */
export default function Home() {
  const [tab, setTab] = useState<Tab>("batch");
  const [settings, setSettings] = useState<Settings>({
    baseUrl: "http://localhost:4321",
    language: "pt",
    locale: "pt-BR",
    diarize: true,
    minSpeakers: 2,
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);
  const [log, setLog] = useState<NetEntry[]>([]);
  const [lastResult, setLastResult] = useState<TranscriptionResult | null>(null);

  const addLog = useCallback((label: string, data?: unknown) => {
    setLog((o) => [...o.slice(-99), { time: now(), label, data }]);
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await getHealth(settings.baseUrl);
        if (alive) {
          setHealth(h);
          setHealthErr(null);
        }
      } catch (e) {
        if (alive) {
          setHealth(null);
          setHealthErr(`sem conexão com ${settings.baseUrl} (${String(e)})`);
        }
      }
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [settings.baseUrl]);

  const set = (k: keyof Settings) => (v: string | boolean | number | null) =>
    setSettings((o) => ({ ...o, [k]: v }));

  return (
    <main>
      <h1>STT API — teste (com diarização)</h1>

      <div className="settings">
        <label>Servidor
          <input
            type="text"
            value={settings.baseUrl}
            onChange={(e) => set("baseUrl")(e.target.value.replace(/\/$/, ""))}
          />
        </label>
        <label>Idioma
          <select value={settings.language} onChange={(e) => set("language")(e.target.value)}>
            <option value="pt">pt</option>
            <option value="en">en</option>
            <option value="es">es</option>
          </select>
        </label>
        <label>Locale (WS)
          <select value={settings.locale} onChange={(e) => set("locale")(e.target.value)}>
            <option value="pt-BR">pt-BR</option>
            <option value="en-US">en-US</option>
            <option value="es-ES">es-ES</option>
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={settings.diarize}
            onChange={(e) => set("diarize")(e.target.checked)}
          />
          Diarizar
        </label>
        <label title="Piso de locutores (vazio = automático)">Min. falantes
          <input
            type="number"
            min={1}
            max={10}
            placeholder="auto"
            value={settings.minSpeakers ?? ""}
            onChange={(e) => {
              const v = e.target.value === "" ? null : Math.max(1, parseInt(e.target.value, 10) || 1);
              set("minSpeakers")(v);
            }}
          />
        </label>
      </div>

      <HealthBar health={health} error={healthErr} />

      <div className="tabs">
        {(["batch", "sse", "ws", "refine"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t === "batch" ? "Batch" : t === "sse" ? "Streaming SSE" : t === "ws" ? "WebSocket" : "Refine"}
          </button>
        ))}
      </div>

      {tab === "batch" && <BatchTab settings={settings} addLog={addLog} onResult={setLastResult} />}
      {tab === "sse" && <SseTab settings={settings} addLog={addLog} onResult={setLastResult} />}
      {tab === "ws" && <WsTab settings={settings} addLog={addLog} onResult={setLastResult} />}
      {tab === "refine" && <RefineTab settings={settings} lastResult={lastResult} addLog={addLog} />}

      <NetLog entries={log} onClear={() => setLog([])} />
    </main>
  );
}
