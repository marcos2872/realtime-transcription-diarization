// Captura de áudio: mic → PCM Int16LE 16kHz mono (formato exigido pela API).
// Usa AudioWorklet (public/pcm-worklet.js) + resample linear quando o
// dispositivo não entrega 16kHz nativamente.

export const TARGET_RATE = 16000;

export function floatTo16BitPCM(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function resampleMono(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === TARGET_RATE) return input;
  const ratio = fromRate / TARGET_RATE;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    const a = input[i0] ?? 0;
    const b = input[Math.min(i0 + 1, input.length - 1)] ?? 0;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export function encodeWav(pcm: Int16Array, sampleRate = TARGET_RATE): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const wstr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  wstr(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  wstr(8, "WAVE");
  wstr(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  wstr(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, pcm[i], true);
  return new Blob([buf], { type: "audio/wav" });
}

export function int16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let bin = "";
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH) as unknown as number[]);
  }
  return btoa(bin);
}

/** Converte arquivo de áudio qualquer → WAV 16kHz mono via decode + resample. */
export async function fileToWav16kMono(file: File): Promise<Blob> {
  const Ctx = window.AudioContext;
  const tmp = new Ctx();
  try {
    const buf = await tmp.decodeAudioData(await file.arrayBuffer());
    const n = buf.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i] / buf.numberOfChannels;
    }
    const pcm = floatTo16BitPCM(resampleMono(mono, buf.sampleRate));
    return encodeWav(pcm);
  } finally {
    await tmp.close();
  }
}

export interface PcmChunk {
  pcm16: Int16Array; // sempre 16kHz mono
}

/** Captura contínua do microfone, emitindo Int16 16kHz mono. */
export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private acc = new Float32Array(0);
  private inputRate = TARGET_RATE;
  onChunk: ((c: PcmChunk) => void) | null = null;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: TARGET_RATE, channelCount: 1, echoCancellation: true },
    });
    this.ctx = new AudioContext({ sampleRate: TARGET_RATE });
    await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
    this.inputRate = this.ctx.sampleRate;
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture");
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const mono = e.data;
      const merged = new Float32Array(this.acc.length + mono.length);
      merged.set(this.acc);
      merged.set(mono, this.acc.length);
      this.acc = merged;
      // entrega blocos de ~0.5s (8000 samples a 16kHz)
      const target = Math.floor((TARGET_RATE / this.inputRate) * 8000) || 8000;
      while (this.acc.length >= target) {
        const slice = this.acc.slice(0, target);
        this.acc = this.acc.slice(target);
        this.onChunk?.({ pcm16: floatTo16BitPCM(resampleMono(slice, this.inputRate)) });
      }
    };
    src.connect(this.node);
  }

  /** Para a captura e retorna o PCM acumulado (para gravação batch). */
  async stop(): Promise<Int16Array> {
    const rest = floatTo16BitPCM(resampleMono(this.acc, this.inputRate));
    this.acc = new Float32Array(0);
    this.node?.disconnect();
    this.node = null;
    if (this.ctx) await this.ctx.close();
    this.ctx = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    return rest;
  }
}

export function concatInt16(parts: Int16Array[]): Int16Array {
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Int16Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}
