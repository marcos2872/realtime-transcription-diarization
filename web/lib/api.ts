// Cliente HTTP da STT API (batch, streaming SSE, refine, health).
// O streaming WebSocket usa a API nativa do browser (ver WsTab).

export interface Segment {
  speaker: string;
  text: string;
  tStart: number;
  tEnd: number;
}

export interface TranscriptionResult {
  sessionId: string;
  segments: Segment[];
  participants: string[];
  durationSec: number;
  language: string;
}

export interface HealthResponse {
  status: string;
  gpus: string[];
  whisperLoaded: boolean;
  refineEndpoint: string;
  activeSessions: number;
}

export interface PartialResult {
  channel: string;
  speaker: string;
  text: string;
  tStart: number;
  tEnd: number;
  isFinal: boolean;
}

export function newSessionId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(6)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function check(res: Response): Promise<unknown> {
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getHealth(baseUrl: string): Promise<HealthResponse> {
  const res = await fetch(`${baseUrl}/health`);
  return (await check(res)) as HealthResponse;
}

export async function transcribeBatch(
  baseUrl: string,
  wav: Blob,
  language: string,
  diarize: boolean,
  minSpeakers: number | null = null,
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("audio", wav, "audio.wav");
  form.append("language", language);
  form.append("diarize", String(diarize));
  if (minSpeakers !== null) form.append("minSpeakers", String(minSpeakers));
  const res = await fetch(`${baseUrl}/transcribe`, { method: "POST", body: form });
  return (await check(res)) as TranscriptionResult;
}

export async function streamStart(
  baseUrl: string,
  sessionId: string,
  language: string,
  diarize: boolean,
  minSpeakers: number | null = null,
): Promise<void> {
  const res = await fetch(`${baseUrl}/stream/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId,
      action: "start",
      language,
      channels: ["system"],
      diarize,
      minSpeakers,
    }),
  });
  await check(res);
}

export async function streamSendAudio(
  baseUrl: string,
  sessionId: string,
  seq: number,
  dataB64: string,
): Promise<void> {
  const res = await fetch(`${baseUrl}/stream/${sessionId}/audio`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, channel: "system", seq, data: dataB64 }),
  });
  await check(res);
}

export async function streamStop(
  baseUrl: string,
  sessionId: string,
  language: string,
): Promise<TranscriptionResult> {
  const res = await fetch(`${baseUrl}/stream/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId,
      action: "stop",
      language,
      channels: ["system"],
    }),
  });
  return (await check(res)) as TranscriptionResult;
}

export function streamEvents(
  baseUrl: string,
  sessionId: string,
  onPartial: (p: PartialResult) => void,
  onClosed: () => void,
  onError: (msg: string) => void,
): () => void {
  const es = new EventSource(`${baseUrl}/stream/${sessionId}/events`);
  es.addEventListener("partial", (e) => {
    try {
      onPartial(JSON.parse((e as MessageEvent).data) as PartialResult);
    } catch {
      // ignora payload malformado
    }
  });
  es.addEventListener("heartbeat", (e) => {
    if ((e as MessageEvent).data === "closed") {
      es.close();
      onClosed();
    }
  });
  es.onerror = () => {
    es.close();
    onError("conexão SSE encerrada");
  };
  return () => es.close();
}

export async function refineTranscription(
  baseUrl: string,
  transcription: TranscriptionResult,
): Promise<TranscriptionResult> {
  const res = await fetch(`${baseUrl}/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcription }),
  });
  const data = (await check(res)) as { refined: TranscriptionResult };
  return data.refined;
}

export function toWsUrl(baseUrl: string): string {
  const u = baseUrl.replace(/^http/, "ws").replace(/\/$/, "");
  return `${u}/speech/stream`;
}
