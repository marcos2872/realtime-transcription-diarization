"use client";

import { useRef, useState } from "react";
import { CHUNK_BYTES, TARGET_SAMPLE_RATE, decodeAudioFile, splitIntoChunks, toPCM16Mono16k } from "../lib/audio";
import { startMicCapture } from "../lib/mic";
import { useTranscriptionStream, type StreamOptions } from "../lib/useTranscriptionStream";
import { StatusBar, UtteranceList } from "../components/transcript-ui";

const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function randomStreamId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 8)}`;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm text-zinc-400">
      {label}
      {children}
    </label>
  );
}

const inputClass =
  "rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-sky-500";

export default function Home() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [partials, setPartials] = useState(true);
  const [tab, setTab] = useState<"live" | "file">("live");
  const stream = useTranscriptionStream();
  const micRef = useRef<{ stop: () => void } | null>(null);
  const [recording, setRecording] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const options = (prefix: string): StreamOptions => ({
    httpBase: apiBase,
    streamId: randomStreamId(prefix),
    partials,
  });

  async function toggleRecording() {
    if (recording) {
      micRef.current?.stop();
      micRef.current = null;
      setRecording(false);
      stream.finish(); // flush: emite frases pendentes e fecha o stream
      return;
    }
    try {
      await stream.connect(options("mic"));
      micRef.current = await startMicCapture((pcm) => stream.sendAudio(pcm));
      setRecording(true);
    } catch (error) {
      micRef.current?.stop();
      micRef.current = null;
    }
  }

  async function transcribeFile(file: File) {
    setSending(true);
    setFileName(file.name);
    try {
      // Todo o processamento é local: decodifica mp3 -> PCM16 mono 16 kHz
      // e envia pelo mesmo WebSocket de streaming.
      const { channels, sampleRate } = await decodeAudioFile(file);
      const pcm = toPCM16Mono16k(channels, sampleRate);
      await stream.connect(options("arquivo"));
      for (const chunk of splitIntoChunks(pcm, CHUNK_BYTES)) {
        stream.sendAudio(chunk);
        await new Promise((resolve) => setTimeout(resolve, 5)); // evita rajada no socket
      }
      stream.finish();
    } catch (error) {
      console.error(error);
    } finally {
      setSending(false);
    }
  }

  const busy = stream.connection === "connecting" || stream.connection === "closing";

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Transcrição em tempo real</h1>
          <p className="text-sm text-zinc-500">Nemotron 3.5 ASR + diarização pyannote</p>
        </div>
        <StatusBar connection={recording || sending ? "open" : stream.connection} />
      </header>

      <section className="grid grid-cols-1 gap-3 rounded-xl bg-zinc-900/50 p-4 sm:grid-cols-2">
        <Field label="URL da API">
          <input
            className={inputClass}
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://localhost:8000"
          />
        </Field>
        <label className="flex items-end gap-2 pb-2 text-sm text-zinc-300">
          <input
            type="checkbox"
            checked={partials}
            onChange={(e) => setPartials(e.target.checked)}
            className="h-4 w-4 accent-sky-500"
          />
          Mostrar texto parcial
        </label>
      </section>

      <nav className="flex gap-2">
        {(["live", "file"] as const).map((t) => (
          <button
            key={t}
            onClick={() => {
              stream.reset();
              setTab(t);
            }}
            className={`rounded-lg px-4 py-2 text-sm font-medium ${
              tab === t ? "bg-sky-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
            }`}
          >
            {t === "live" ? "Ao vivo (microfone)" : "Arquivo .mp3"}
          </button>
        ))}
      </nav>

      {tab === "live" ? (
        <section className="flex flex-col gap-3">
          <button
            onClick={toggleRecording}
            disabled={busy}
            className={`rounded-xl px-4 py-3 font-semibold disabled:opacity-50 ${
              recording ? "bg-red-600 hover:bg-red-500" : "bg-emerald-600 hover:bg-emerald-500"
            }`}
          >
            {recording ? "Parar e finalizar" : "Começar a transcrever"}
          </button>
          <p className="text-xs text-zinc-500">
            O microfone exige localhost ou HTTPS. Fale e acompanhe o parcial abaixo; as frases
            saem com o falante (SPEAKER_00, …).
          </p>
        </section>
      ) : (
        <section className="flex flex-col gap-3">
          <label
            className={`cursor-pointer rounded-xl border-2 border-dashed px-4 py-6 text-center text-sm ${
              sending ? "border-zinc-700 text-zinc-500" : "border-zinc-600 text-zinc-300 hover:border-sky-500"
            }`}
          >
            {fileName ?? "Clique para escolher um .mp3"}
            <input
              type="file"
              accept=".mp3,audio/mpeg"
              className="hidden"
              disabled={sending || busy}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void transcribeFile(file);
                e.target.value = "";
              }}
            />
          </label>
          {sending && <p className="text-sm text-zinc-400">Enviando áudio… ({TARGET_SAMPLE_RATE / 1000} kHz mono)</p>}
        </section>
      )}

      {stream.error && (
        <p className="rounded-lg bg-red-950 px-3 py-2 text-sm text-red-200">Erro: {stream.error}</p>
      )}

      {partials && stream.partial && (
        <p className="rounded-lg bg-zinc-900 px-3 py-2 text-sm text-zinc-400">
          <span className="mr-2 text-xs uppercase text-zinc-600">parcial</span>
          {stream.partial}
          <span className="animate-pulse">▍</span>
        </p>
      )}

      <section>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Frases ({stream.utterances.length})
        </h2>
        <UtteranceList utterances={stream.utterances} />
      </section>
    </main>
  );
}
