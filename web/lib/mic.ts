"use client";

/**
 * Captura do microfone via AudioWorklet: recebe Float32 na taxa nativa,
 * acumula ~250 ms, converte para PCM16 mono 16 kHz e entrega via callback.
 * AudioWorklet (em vez de ScriptProcessor) para não travar a thread principal.
 */

import { toPCM16Mono16k } from "./audio";

const WORKLET_NAME = "pcm-capture";
const FRAMES_PER_CALLBACK = 4000; // ~250 ms a 16 kHz

const WORKLET_CODE = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = [];
    this.pendingLength = 0;
    this.batchSize = ${FRAMES_PER_CALLBACK};
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const mono = new Float32Array(input[0].length);
      for (let i = 0; i < mono.length; i++) {
        let sum = 0;
        for (let c = 0; c < input.length; c++) sum += input[c][i];
        mono[i] = sum / input.length;
      }
      this.pending.push(mono);
      this.pendingLength += mono.length;
      if (this.pendingLength >= this.batchSize) {
        const out = new Float32Array(this.pendingLength);
        let offset = 0;
        for (const part of this.pending) {
          out.set(part, offset);
          offset += part.length;
        }
        this.pending = [];
        this.pendingLength = 0;
        this.port.postMessage(out, [out.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('${WORKLET_NAME}', PcmCapture);
`;

export interface MicCapture {
  stop: () => void;
  contextRate: number;
}

export async function startMicCapture(onPcm: (pcm: ArrayBuffer) => void): Promise<MicCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  const context = new AudioContext();
  try {
    const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
    await context.audioWorklet.addModule(URL.createObjectURL(blob));
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, WORKLET_NAME);
    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      onPcm(toPCM16Mono16k([event.data], context.sampleRate));
    };
    source.connect(node);
    // Sem conexão com o destino: o worklet segue processando sem emitir som.
    node.connect(context.destination);

    let stopped = false;
    return {
      contextRate: context.sampleRate,
      stop: () => {
        if (stopped) return;
        stopped = true;
        node.disconnect();
        source.disconnect();
        void context.close();
        for (const track of stream.getTracks()) track.stop();
      },
    };
  } catch (error) {
    void context.close();
    for (const track of stream.getTracks()) track.stop();
    throw error;
  }
}
