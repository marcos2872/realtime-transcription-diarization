import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Transcrição em tempo real",
  description: "Cliente de teste para o transcript-service (Nemotron + pyannote)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="bg-zinc-950 text-zinc-100 antialiased">{children}</body>
    </html>
  );
}
