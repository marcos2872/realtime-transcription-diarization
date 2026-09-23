#!/bin/bash
# Entrypoint para o container llama.cpp
# Baixa o modelo GGUF (sharded) se necessário e inicia o servidor
set -e

MODEL_DIR="/models"
MODEL_PREFIX="qwen2.5-7b-instruct-q4_k_m"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"

# Número de shards (splitado em 2 arquivos)
NUM_SHARDS=2

echo "=== llama.cpp Server ==="
echo "Modelo: Qwen 2.5 7B Instruct (Q4_K_M, $NUM_SHARDS shards)"
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

# Baixa cada shard se não existir ou se o tamanho for muito pequeno (< 10MB = corrompido)
for i in $(seq 1 "$NUM_SHARDS"); do
    SHARD="${MODEL_PREFIX}-$(printf '%05d' $i)-of-$(printf '%05d' $NUM_SHARDS).gguf"
    SHARD_PATH="$MODEL_DIR/$SHARD"
    MIN_SIZE=$((10 * 1024 * 1024))  # 10MB mínimo (cada shard tem ~2GB)

    if [ -f "$SHARD_PATH" ] && [ "$(stat -c%s "$SHARD_PATH" 2>/dev/null || echo 0)" -ge "$MIN_SIZE" ]; then
        echo "  ✓ $SHARD já existe ($(du -h "$SHARD_PATH" | cut -f1))"
    else
        if [ -f "$SHARD_PATH" ]; then
            echo "  × $SHARD corrompido ($(du -h "$SHARD_PATH" | cut -f1)), baixando novamente..."
            rm -f "$SHARD_PATH"
        else
            echo "Baixando $SHARD (~2GB, aguarde)..."
        fi
        mkdir -p "$MODEL_DIR"

        if [ -n "${HF_TOKEN:-}" ]; then
            $DOWNLOAD_CMD "$SHARD_PATH" "$MODEL_URL/$SHARD" -H "Authorization: Bearer $HF_TOKEN" 2>&1
        else
            $DOWNLOAD_CMD "$SHARD_PATH" "$MODEL_URL/$SHARD" 2>&1
        fi
        # Verifica se baixou corretamente
        if [ "$(stat -c%s "$SHARD_PATH" 2>/dev/null || echo 0)" -lt "$MIN_SIZE" ]; then
            echo "  ✗ Erro: download de $SHARD falhou ($(du -h "$SHARD_PATH" | cut -f1))"
            exit 1
        fi
        echo "  ✓ $SHARD baixado ($(du -h "$SHARD_PATH" | cut -f1))"
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

# Passa o primeiro shard, llama.cpp descobre os demais automaticamente
exec "$LLAMA_SERVER" -m "$MODEL_DIR/$MODEL_PREFIX-00001-of-$(printf '%05d' $NUM_SHARDS).gguf" "$@"
