"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toWebSocketUrl, type ConnectionState, type ServerEvent, type Utterance } from "./api";

export interface StreamOptions {
  httpBase: string;
  streamId: string;
  partials: boolean;
}

interface StreamState {
  connection: ConnectionState;
  partial: string | null;
  utterances: Utterance[];
  error: string | null;
  ready: boolean;
}

/**
 * Gerencia um stream de transcrição: abre o WebSocket, envia áudio PCM16
 * e acumula partial + utterances. O socket é propriedade do hook; quem
 * captura o áudio chama `sendAudio()` / `finish()`.
 */
export function useTranscriptionStream() {
  const [state, setState] = useState<StreamState>({
    connection: "idle",
    partial: null,
    utterances: [],
    error: null,
    ready: false,
  });
  const socketRef = useRef<WebSocket | null>(null);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
  }, []);

  useEffect(() => disconnect, [disconnect]);

  const connect = useCallback(
    (options: StreamOptions): Promise<void> => {
      disconnect();
      setState({ connection: "connecting", partial: null, utterances: [], error: null, ready: false });
      return new Promise((resolve, reject) => {
        const socket = new WebSocket(toWebSocketUrl(options.httpBase, options.streamId, options.partials));
        socketRef.current = socket;
        socket.binaryType = "arraybuffer";

        socket.onopen = () => setState((s) => ({ ...s, connection: "open" }));

        socket.onmessage = (event: MessageEvent) => {
          let parsed: ServerEvent;
          try {
            parsed = JSON.parse(event.data as string) as ServerEvent;
          } catch {
            return; // mensagens binárias do servidor não existem no contrato
          }
          switch (parsed.type) {
            case "ready":
              setState((s) => ({ ...s, ready: true }));
              resolve();
              break;
            case "partial":
              setState((s) => ({ ...s, partial: parsed.text }));
              break;
            case "utterance":
              setState((s) => ({ ...s, utterances: [...s.utterances, parsed], partial: null }));
              break;
            case "error":
              setState((s) => ({
                ...s,
                error: `${parsed.code}: ${parsed.message}`,
                connection: parsed.fatal ? "error" : s.connection,
              }));
              if (parsed.fatal) reject(new Error(`${parsed.code}: ${parsed.message}`));
              break;
            case "closed":
              setState((s) => ({ ...s, connection: "idle" }));
              socket.close();
              break;
            case "pong":
              break;
          }
        };

        socket.onerror = () => {
          setState((s) =>
            s.ready ? s : { ...s, connection: "error", error: "Não foi possível conectar à API." },
          );
          reject(new Error("Falha de conexão com a API."));
        };

        socket.onclose = () => {
          if (socketRef.current === socket) socketRef.current = null;
          setState((s) => (s.connection === "open" ? { ...s, connection: "idle" } : s));
        };
      });
    },
    [disconnect],
  );

  const sendAudio = useCallback((pcm: ArrayBuffer) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(pcm);
  }, []);

  const finish = useCallback(() => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      setState((s) => ({ ...s, connection: "closing" }));
      socket.send(JSON.stringify({ type: "end" }));
    }
  }, []);

  const reset = useCallback(() => {
    disconnect();
    setState({ connection: "idle", partial: null, utterances: [], error: null, ready: false });
  }, [disconnect]);

  return { ...state, connect, sendAudio, finish, reset };
}
