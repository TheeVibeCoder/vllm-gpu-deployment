#!/bin/bash
# Launch vLLM with native LMCache 3-Tier KV Offloading
set -e

CONFIG_PATH=${LMCACHE_CONFIG_FILE:-"configs/lmcache.yaml"}
MODEL_NAME=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
PORT=${PORT:-8000}

echo "Launching vLLM with LMCacheConnectorV1..."
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"

LMCACHE_CONFIG_FILE=$CONFIG_PATH \
vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization 0.75 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
