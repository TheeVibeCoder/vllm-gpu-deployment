#!/bin/bash
# Universal vLLM Startup Command
MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
PORT=${2:-8000}
CONTEXT=${3:-32768}

echo "Launching vLLM for $MODEL on port $PORT with ${CONTEXT} token context..."
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization 0.75 \
  --max-model-len "$CONTEXT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm.log 2>&1 &

echo "vLLM server started in background! Tracking logs..."
sleep 5
tail -n 25 vllm.log
