import type { HealthResponse } from "../lib/api";

/** Barra de status do servidor (GET /health). */
export default function HealthBar({
  health,
  error,
}: {
  health: HealthResponse | null;
  error: string | null;
}) {
  if (error) return <div className="health err">✗ {error}</div>;
  if (!health) return <div className="health">… consultando /health</div>;
  return (
    <div className="health">
      <span className={health.whisperLoaded ? "ok" : "warn"}>
        ● {health.whisperLoaded ? "Whisper carregado" : "Whisper NÃO carregado"}
      </span>
      <span>GPUs: {health.gpus.join(", ") || "—"}</span>
      <span>Sessões: {health.activeSessions}</span>
      <span className="muted">{health.refineEndpoint}</span>
    </div>
  );
}
