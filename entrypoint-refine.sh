#!/bin/bash
# Entrypoint para o container llama.cpp
# Baixa o modelo GGUF (sharded) se necessário e inicia o servidor
set -e

MODEL_DIR="/models"
# Qwen3-8B Q4_K_M (arquivo único, ~5GB) — PT-BR bem melhor que 2.5,
# rápido o bastante para o refine em batch.
MODEL_FILES=("Qwen3-8B-Q4_K_M.gguf")
MODEL_URL="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main"
MODEL_LABEL="Qwen3 8B (Q4_K_M)"

echo "=== llama.cpp Server ==="
echo "Modelo: $MODEL_LABEL (${#MODEL_FILES[@]} arquivo(s))"
echo ""

# Verifica se curl está disponível (preferencial) ou wget
DOWNLOAD_CMD=""
if command -v curl &>/dev/null; then
    DOWNLOAD_CMD="curl -L --progress-bar -o"
elif command -v wget &>/dev/null; then
    DOWNLOAD_CMD="wget -O"
else
    echo "Erro: instale curl ou wget no container"
    echo "  apt-get update && apt-get install -y curl"
    exit 1
fi

# Baixa cada arquivo se não existir ou se o tamanho for muito pequeno (< 10MB = corrompido)
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
    MIN_SIZE=$((10 * 1024 * 1024))  # 10MB mínimo

    if [ -f "$MODEL_PATH" ] && [ "$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)" -ge "$MIN_SIZE" ]; then
        echo "  ✓ $MODEL_FILE já existe ($(du -h "$MODEL_PATH" | cut -f1))"
    else
        if [ -f "$MODEL_PATH" ]; then
            echo "  × $MODEL_FILE corrompido ($(du -h "$MODEL_PATH" | cut -f1)), baixando novamente..."
            rm -f "$MODEL_PATH"
        else
            echo "Baixando $MODEL_FILE (aguarde)..."
        fi
        mkdir -p "$MODEL_DIR"

        if [ -n "${HF_TOKEN:-}" ]; then
            $DOWNLOAD_CMD "$MODEL_PATH" "$MODEL_URL/$MODEL_FILE" -H "Authorization: Bearer $HF_TOKEN" 2>&1
        else
            $DOWNLOAD_CMD "$MODEL_PATH" "$MODEL_URL/$MODEL_FILE" 2>&1
        fi
        # Verifica se baixou corretamente
        if [ "$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)" -lt "$MIN_SIZE" ]; then
            echo "  ✗ Erro: download de $MODEL_FILE falhou ($(du -h "$MODEL_PATH" | cut -f1))"
            exit 1
        fi
        echo "  ✓ $MODEL_FILE baixado ($(du -h "$MODEL_PATH" | cut -f1))"
    fi
done

echo ""
echo "Modelo pronto. Iniciando llama-server..."
echo ""

# Caminho do binário no container server-cuda
LLAMA_SERVER="/app/llama-server"
if [ ! -x "$LLAMA_SERVER" ]; then
    # Fallback: tenta encontrar no PATH
    LLAMA_SERVER="llama-server"
fi

# Inicia o servidor com o primeiro arquivo da lista
exec "$LLAMA_SERVER" -m "$MODEL_DIR/${MODEL_FILES[0]}" "$@"
