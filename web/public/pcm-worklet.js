// Passa Float32 mono do mic para a thread principal em blocos de 128 samples.
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0];
    if (ch && ch.length > 0) {
      const n = ch[0].length;
      const mono = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < ch.length; c++) s += ch[c][i];
        mono[i] = s / ch.length;
      }
      this.port.postMessage(mono);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
