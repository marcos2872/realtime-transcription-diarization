"use client";

import { useState } from "react";
import { refineTranscription, type TranscriptionResult } from "../lib/api";
import SegmentList from "./SegmentList";
import type { Settings } from "./common";

/** Aba 4: refinamento via LLM do último resultado obtido em outra aba. */
export default function RefineTab({
  settings,
  lastResult,
  addLog,
}: {
  settings: Settings;
  lastResult: TranscriptionResult | null;
  addLog: (label: string, data?: unknown) => void;
}) {
  const [refined, setRefined] = useState<TranscriptionResult | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!lastResult || busy) return;
    setBusy(true);
    try {
      addLog("POST /refine");
      const r = await refineTranscription(settings.baseUrl, lastResult);
      setRefined(r);
      addLog("→ /refine", { segments: r.segments.length });
    } catch (e) {
      addLog("✗ /refine", String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!lastResult) {
    return (
      <div className="card">
        <h3>Refine — POST /refine</h3>
        <p className="muted">Execute uma transcrição em outra aba primeiro.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3>Refine — POST /refine</h3>
      <p className="muted">Refinando sessão {lastResult.sessionId} ({lastResult.segments.length} segmentos)</p>
      <div className="row">
        <button onClick={run} disabled={busy}>{busy ? "Refinando…" : "Refinar via LLM"}</button>
      </div>
      <div className="cols">
        <div>
          <h4>Original</h4>
          <SegmentList segments={lastResult.segments} />
        </div>
        <div>
          <h4>Refinado</h4>
          {refined ? <SegmentList segments={refined.segments} /> : <p className="muted">—</p>}
        </div>
      </div>
    </div>
  );
}
