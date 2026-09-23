export interface NetEntry {
  time: string;
  label: string;
  data?: unknown;
}

/** Log cru de request/response para debug. */
export default function NetLog({ entries, onClear }: { entries: NetEntry[]; onClear: () => void }) {
  return (
    <div className="card">
      <div className="row space-between">
        <h3>Log de rede</h3>
        <button onClick={onClear}>Limpar</button>
      </div>
      <div className="log">
        {entries.length === 0 && <span className="muted">—</span>}
        {entries.map((e, i) => (
          <div key={i} className="log-line">
            <span className="muted">{e.time}</span> <b>{e.label}</b>
            {e.data !== undefined && (
              <pre>{JSON.stringify(e.data, null, 1)?.slice(0, 2000)}</pre>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
