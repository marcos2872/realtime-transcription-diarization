from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ── STT ──
    whisper_model: str = "large-v3"
    whisper_compute: str = "int8_float16"
    whisper_gpus: str = "cuda:0,cuda:1"  # str separada por vírgula

    @property
    def whisper_gpus_list(self) -> list[str]:
        return [g.strip() for g in self.whisper_gpus.split(",") if g.strip()]

    # ── Refinamento (OpenAI-compatible) ──
    refine_base_url: str = "http://localhost:8080/v1"
    refine_api_key: str = ""
    refine_model: str = "Qwen3-8B"

    # ── Concorrência ──
    max_concurrent_streams: int = 20
    queue_timeout_sec: int = 300

    # ── Hugging Face ──
    hf_token: str = ""

    # ── Gerais ──
    max_file_size_mb: int = 500
    session_timeout_min: int = 60
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
