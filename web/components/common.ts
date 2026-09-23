import type { TranscriptionResult } from "../lib/api";
import type { NetEntry } from "./NetLog";

export interface Settings {
  baseUrl: string;
  language: string;
  locale: string;
  diarize: boolean;
}

export interface TabProps {
  settings: Settings;
  addLog: (label: string, data?: unknown) => void;
  onResult: (r: TranscriptionResult) => void;
}

export function now(): string {
  return new Date().toLocaleTimeString("pt-BR", { hour12: false });
}

export type { NetEntry };
