#!/bin/bash
# Universal Environment Setup (Runs on cloud GPU machine)
set -e

echo "=== 1. Checking NVIDIA GPU Hardware ==="
nvidia-smi

echo "=== 2. Installing vLLM and Dependencies ==="
pip install -q vllm
pip install -q "numpy<2.0.0"

echo "=== 3. Validating CUDA & vLLM ==="
python -c "import torch, vllm; print(f'CUDA Available: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)}), vLLM: {vllm.__version__}')"
echo "Environment is Ready!"
