"use client";

import { formatTime, type ConnectionState, type Utterance } from "../lib/api";

const STATE_STYLE: Record<ConnectionState, string> = {
  idle: "bg-zinc-700",
  connecting: "bg-amber-500 animate-pulse",
  open: "bg-emerald-500",
  closing: "bg-amber-500 animate-pulse",
  error: "bg-red-500",
};

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: "desconectado",
  connecting: "conectando…",
  open: "transcrevendo",
  closing: "finalizando…",
  error: "erro",
};

export function StatusBar({ connection }: { connection: ConnectionState }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-zinc-800 px-3 py-1 text-sm">
      <span className={`h-2.5 w-2.5 rounded-full ${STATE_STYLE[connection]}`} />
      {STATE_LABEL[connection]}
    </span>
  );
}

const SPEAKER_COLORS = [
  "bg-sky-500",
  "bg-violet-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
];

export function speakerColor(speaker: string): string {
  const match = /SPEAKER_(\d+)/.exec(speaker);
  const index = match ? parseInt(match[1], 10) : speaker.length;
  return SPEAKER_COLORS[index % SPEAKER_COLORS.length];
}

export function UtteranceList({ utterances }: { utterances: Utterance[] }) {
  if (utterances.length === 0) {
    return <p className="text-sm text-zinc-500">Nenhuma frase finalizada ainda.</p>;
  }
  return (
    <ul className="flex flex-col gap-2">
      {utterances.map((u) => (
        <li key={u.utterance_id} className="rounded-lg bg-zinc-900 p-3">
          <div className="mb-1 flex items-center gap-2 text-xs">
            <span className={`rounded px-1.5 py-0.5 font-semibold text-zinc-950 ${speakerColor(u.speaker)}`}>
              {u.speaker}
            </span>
            <span className="text-zinc-500">
              {formatTime(u.start)} → {formatTime(u.end)}
            </span>
          </div>
          <p className="text-[15px] leading-relaxed">{u.text}</p>
        </li>
      ))}
    </ul>
  );
}
