import logging
import time

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class Refiner:
    """Cliente para refinamento via API compatível com OpenAI (llama.cpp, vLLM, etc.)."""

    def __init__(self):
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=settings.refine_base_url,
                api_key=settings.refine_api_key or "sk-no-key-required",
            )
        return self._client

    async def refine(
        self,
        segments: list[dict],
        language: str = "pt",
        model: str | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        """Refina uma transcrição usando LLM.

        Args:
            segments: Lista de segmentos da transcrição bruta.
            language: Idioma da transcrição.
            model: Nome do modelo (default: settings.refine_model).
            prompt: Prompt opcional para customizar o refinamento.

        Returns:
            Lista de segmentos refinados.
        """
        model = model or settings.refine_model
        max_segments = 50  # limite por chamada pra não estourar contexto

        if len(segments) > max_segments:
            # Processa em lotes
            all_refined = []
            for i in range(0, len(segments), max_segments):
                batch = segments[i:i + max_segments]
                refined = await self._refine_batch(batch, language, model, prompt)
                all_refined.extend(refined)
            return all_refined

        return await self._refine_batch(segments, language, model, prompt)

    async def _refine_batch(
        self,
        segments: list[dict],
        language: str,
        model: str,
        custom_prompt: str | None,
    ) -> list[dict]:
        """Refina um lote de segmentos."""
        transcript_text = self._segments_to_text(segments)

        system_prompt = (
            "Você é um especialista em refinar transcrições de reuniões. "
            "Corrija ortografia, pontuação e gramática. "
            "Preserve o conteúdo original, não resuma. "
            "Mantenha a formatação de falas identificadas por '[LOCUTOR]: texto'."
        )

        if custom_prompt:
            system_prompt = custom_prompt

        user_prompt = (
            f"Idioma: {language}\n\n"
            f"Transcrição:\n{transcript_text}\n\n"
            "Refine a transcrição acima, corrigindo apenas ortografia, "
            "pontuação e gramática. Mantenha exatamente o mesmo conteúdo e estrutura."
        )

        try:
            t0 = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            elapsed = time.time() - t0

            refined_text = response.choices[0].message.content or transcript_text
            logger.info(
                "Refinamento concluído em %.1fs (modelo=%s, tokens=%d)",
                elapsed, model,
                response.usage.total_tokens if response.usage else 0,
            )

            # Reconverte o texto refinado de volta para segmentos
            return self._text_to_segments(refined_text, segments)

        except Exception as e:
            logger.warning("Erro no refinamento: %s — retornando original", e)
            return segments

    def _segments_to_text(self, segments: list[dict]) -> str:
        """Converte segmentos em texto plano."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Locutor")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{speaker}]: {text}")
        return "\n\n".join(lines)

    def _text_to_segments(self, text: str, original: list[dict]) -> list[dict]:
        """Converte texto refinado de volta para segmentos, preservando timestamps."""
        import re

        # Tenta parsear o formato [LOCUTOR]: texto
        pattern = re.compile(r'^\[([^\]]+)\]:\s*(.+)$', re.MULTILINE)
        matches = pattern.findall(text)

        if not matches:
            # Fallback: usa o texto inteiro como um segmento único
            return [{
                "speaker": original[0].get("speaker", "Locutor"),
                "text": text.strip(),
                "tStart": original[0].get("tStart", 0),
                "tEnd": original[-1].get("tEnd", 0),
            }]

        # Mapeia segmentos originais por speaker para preservar timestamps
        speaker_times: dict[str, list[dict]] = {}
        for seg in original:
            spk = seg.get("speaker", "Locutor")
            speaker_times.setdefault(spk, []).append(seg)

        refined = []
        speaker_idx: dict[str, int] = {}
        for speaker, line_text in matches:
            speaker_idx.setdefault(speaker, 0)
            orig_segs = speaker_times.get(speaker, original)
            idx = min(speaker_idx[speaker], len(orig_segs) - 1)
            t_start = orig_segs[idx].get("tStart", 0) if orig_segs else 0
            t_end = orig_segs[idx].get("tEnd", 0) if orig_segs else 0
            speaker_idx[speaker] += 1

            refined.append({
                "speaker": speaker,
                "text": line_text.strip(),
                "tStart": t_start,
                "tEnd": t_end,
            })

        return refined
