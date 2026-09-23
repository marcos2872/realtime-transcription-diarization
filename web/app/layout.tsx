import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "STT API — Teste",
  description: "Front de teste da STT API (batch, SSE, WebSocket, refine)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
