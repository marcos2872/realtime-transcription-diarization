FROM python:3.12-slim

WORKDIR /app

# FFmpeg do sistema: torchcodec (I/O de áudio do pyannote v4) linka
# contra libavutil do sistema — sem ele, libtorchcodec_core*.so não carrega.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instala uv
RUN pip install uv --quiet

# Copia dependências primeiro (cache de camada)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pré-baixa modelos do pyannote (diarização)
RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
    if [ -n "$HF_TOKEN" ]; then \
        uv run python -c "\
from huggingface_hub import login, snapshot_download;\
login(token='$HF_TOKEN');\
snapshot_download('pyannote/speaker-diarization-3.1');\
snapshot_download('pyannote/segmentation-3.0');\
print('Modelos pyannote baixados');\
" || echo "Aviso: não foi possível baixar modelos pyannote"; \
    else \
        echo "HF_TOKEN não definido, pyannote baixará modelos sob demanda"; \
    fi

# Copia código
COPY src/ src/
COPY .env.example .env

# Porta da API
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
