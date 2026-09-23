import type { Segment } from "../lib/api";

const N_COLORS = 8;

function colorIndex(speaker: string): number {
  let h = 0;
  for (let i = 0; i < speaker.length; i++) h = (h * 31 + speaker.charCodeAt(i)) % 997;
  return h % N_COLORS;
}

/** Lista de segmentos com cor por locutor. */
export default function SegmentList({ segments }: { segments: Segment[] }) {
  if (!segments.length) return <p className="muted">Nenhum segmento.</p>;
  return (
    <ul className="segs">
      {segments.map((s, i) => (
        <li key={i} className={`seg spk-${colorIndex(s.speaker)}`}>
          <span className="seg-head">
            [{s.speaker}] ({s.tStart.toFixed(1)}s – {s.tEnd.toFixed(1)}s)
          </span>
          <span className="seg-text">{s.text}</span>
        </li>
      ))}
    </ul>
  );
}
