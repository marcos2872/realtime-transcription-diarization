"use client";

/** Tipos do contrato da API (espelho de docs/api-contract.md). */

export interface ReadyEvent {
  type: "ready";
  stream_id: string;
  language: string;
  sample_rate: number;
  max_streams: number;
  partials: boolean;
}

export interface Utterance {
  type: "utterance";
  stream_id: string;
  utterance_id: number;
  start: number;
  end: number;
  speaker: string;
  text: string;
}

export interface ErrorEvent {
  type: "error";
  code: string;
  message: string;
  fatal: boolean;
}

export type ServerEvent =
  | ReadyEvent
  | { type: "partial"; stream_id: string; text: string }
  | Utterance
  | ErrorEvent
  | { type: "closed"; stream_id: string }
  | { type: "pong" };

export type ConnectionState = "idle" | "connecting" | "open" | "closing" | "error";

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/** Monta a URL do WebSocket a partir da base HTTP(S) configurada. */
export function toWebSocketUrl(httpBase: string, streamId: string, partials: boolean): string {
  const base = httpBase.replace(/\/$/, "");
  const ws = base.replace(/^http/, "ws");
  return `${ws}/ws/streams/${encodeURIComponent(streamId)}?partials=${partials ? "true" : "false"}`;
}
