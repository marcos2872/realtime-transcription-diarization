/** Processamento de áudio no browser (puro, sem I/O).
 *
 * A API só aceita PCM16 mono 16 kHz via WebSocket, então todo áudio
 * (microfone ou arquivo .mp3) passa por este pipeline antes do envio:
 * canal único -> resample 16 kHz -> PCM16 little-endian.
 */

export const TARGET_SAMPLE_RATE = 16_000;
/** Tamanho de envio sugerido: ~1 s de áudio por chunk. */
export const CHUNK_BYTES = TARGET_SAMPLE_RATE * 2;

/** Mistura N canais em mono (média simples). */
export function downmixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const length = channels[0].length;
  const mono = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (const channel of channels) sum += channel[i] ?? 0;
    mono[i] = sum / channels.length;
  }
  return mono;
}

/** Reamostra por interpolação linear para 16 kHz. */
export function resampleTo16k(samples: Float32Array, fromRate: number): Float32Array {
  if (fromRate === TARGET_SAMPLE_RATE) return samples;
  if (fromRate <= 0 || samples.length === 0) return new Float32Array(0);
  const ratio = fromRate / TARGET_SAMPLE_RATE;
  const outLength = Math.floor(samples.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const index = Math.floor(pos);
    const frac = pos - index;
    const a = samples[index] ?? 0;
    const b = samples[index + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

/** Converte float [-1, 1] para PCM16 little-endian (com clipping). */
export function floatToPCM16(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return buffer;
}

/** Pipeline completo: canais Float32 em qualquer sample rate -> PCM16 mono 16 kHz. */
export function toPCM16Mono16k(channels: Float32Array[], fromRate: number): ArrayBuffer {
  return floatToPCM16(resampleTo16k(downmixToMono(channels), fromRate));
}

/** Divide um ArrayBuffer em pedaços de no máximo `size` bytes (sempre pares). */
export function splitIntoChunks(buffer: ArrayBuffer, size: number): ArrayBuffer[] {
  const evenSize = size - (size % 2);
  const chunks: ArrayBuffer[] = [];
  for (let offset = 0; offset < buffer.byteLength; offset += evenSize) {
    chunks.push(buffer.slice(offset, offset + evenSize));
  }
  return chunks;
}

/** Decodifica um arquivo de áudio (mp3/wav/...) para canais Float32. */
export async function decodeAudioFile(file: Blob): Promise<{ channels: Float32Array[]; sampleRate: number }> {
  const Context = window.AudioContext;
  const context = new Context();
  try {
    const raw = await file.arrayBuffer();
    const audio = await context.decodeAudioData(raw);
    const channels: Float32Array[] = [];
    for (let c = 0; c < audio.numberOfChannels; c++) {
      channels.push(audio.getChannelData(c).slice());
    }
    return { channels, sampleRate: audio.sampleRate };
  } finally {
    void context.close();
  }
}
