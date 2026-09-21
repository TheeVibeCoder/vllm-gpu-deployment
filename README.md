# ⚡ Production vLLM Cloud GPU Deployment & KV Cache Pipeline

[![vLLM](https://img.shields.io/badge/vLLM-0.29+-blue.svg)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/Model-Qwen_2.5_0.5B_Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_Tesla_T4_16GB-green.svg)](https://www.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-ready, reproducible guide and automation toolkit for deploying open-weights LLMs with vLLM on cloud GPUs, scaling to 32K context with tool calling, and avoiding costly GPU idle burn.**

---

## 📖 The Story: How Did We End Up Here?

```text
  [ The Problem ]                    [ The Dead Ends ]                       [ The Solution ]
  Mac with No GPU  ──►  Google Colab / Free Tunnels  ──►  Automated Cloud GPU + vLLM
  Can't buy $5k+        • Blocked outbound SSH (port 22)   • Programmatic boot in 30s
  hardware              • Cloudflare "bad handshake"       • 32K context + Hermes tool parsing
                        • Idle timeouts after 15 mins      • 100% OpenAI API compatible
```

### 1. The Starting Point: Zero Local GPU
Running high-performance modern LLM inference engines like **vLLM** requires an NVIDIA GPU with CUDA Compute Capability $\ge 7.0$. On an Apple Silicon Mac, you have unified memory (MLX/Metal), but cannot natively run NVIDIA's optimized CUDA kernels, PagedAttention C++ libraries, or TensorRT-LLM engines. Buying an RTX 4090 or A100 workstation costs thousands of dollars.

### 2. The Roadblocks: Why Free Notebooks Failed
We initially explored free options like Google Colab and SSH tunneling:
* **The Firewall Block**: Google Cloud blocks outbound TCP traffic on ports 22 and 2200, causing standard remote SSH daemons (like `tmate`) to hang indefinitely.
* **The Cloudflare Handshake Crash**: Automated Cloudflare tunnels (`trycloudflare.com`) triggered Cloudflare WAF Bot Fight challenges, dropping SSH proxy streams with `websocket: bad handshake`.
* **The Idle Timeout Trap**: Colab Free Tier automatically terminates the VM and deletes all files after 15–30 minutes if you close your browser tab. It is impossible to run persistent background services for agentic coding.

### 3. The Breakthrough: Cloud Automation via Python SDK
Instead of fighting with web browser sessions and fragile tunnels, we pivoted to **programmatic cloud GPU control** using **Lightning AI**:
* We automated starting and stopping an **NVIDIA Tesla T4 GPU (16 GB VRAM)** on-demand directly from our local Mac terminal.
* We exposed port 8000 through an encrypted public HTTPS endpoint (`*.cloudspaces.litng.ai/v1`).
* **Zero Idle Burn**: When we are done testing, a single local script safely puts the GPU to sleep so free credits are never wasted while idle.

---

## 🛠️ The 4 Real-World Production Bugs Solved

During this deployment, we hit and solved 4 classic errors that every AI systems engineer encounters:

| Error / Symptom | Root Cause | How We Solved It |
|---|---|---|
| **1. NumPy Crash**<br/>`cannot import name 'Inf' from 'numpy'` | NumPy 2.x removed `np.Inf`, breaking older `scipy` and `transformers` builds. | Pinned `numpy<2.0.0` in the environment before importing vLLM. |
| **2. Tool Calling 400**<br/>`"auto" tool choice requires parser` | Coding agents send `tool_choice="auto"`. Neural networks only output flat text; Qwen outputs XML `<tool_call>` tags. vLLM in text mode didn't know which parser to use. | Added `--enable-auto-tool-choice --tool-call-parser hermes` to parse XML tags into clean OpenAI JSON schemas. |
| **3. Context Overflow**<br/>`Context length 8192 exceeded (8193 tokens)` | Agent prompt (6,145 tokens) + max output tokens (2,048 tokens) = 8,193 tokens. Missed default limit by exactly 1 token! | Scaled server to `--max-model-len 32768` (native Qwen 32K context), leaving **25,000+ tokens of free headroom**. |
| **4. 22-Second Latency Spike**<br/>`WARNING: Triton kernel JIT compilation` | Very first prompt had 6,145 tokens (a new tensor shape). PyTorch Triton paused to compile the attention kernel on the fly. | One-time compilation delay. Subsequent requests reused the compiled kernel in milliseconds. |

---

## 📊 Live Measured Benchmarks

Empirical performance captured on the **NVIDIA Tesla T4 (16 GB)** running **`Qwen/Qwen2.5-0.5B-Instruct`**:

| Metric | Measured Value | Practical Impact |
|---|---|---|
| **Decode Throughput (Generation)** | **`76.83 tokens/sec`** | ~10× faster than normal human reading speed (~8 tok/s). |
| **Prefill Throughput (Prompt Reading)**| **`1,893.7 tokens/sec`** | Ingests ~2,000 prompt tokens in a single second. |
| **Model Weights in VRAM** | **`0.93 GiB`** | Extremely lightweight footprint in FP16 precision. |
| **Paged KV Cache Allocated** | **`6.99 GiB`** | Reserved exclusively for dynamic user attention state. |
| **Total KV Cache Pool** | **`590,512 tokens`** | Total concurrent token capacity before swapping. |
| **Max Concurrent Users** | **`576 parallel requests`** | Calculated capacity at 1,024 context tokens per user. |
| **Total Test Session Cost** | **`~$0.25`** | Less than 30 cents for 35 minutes of live GPU time! |

---

## 📁 Repository Structure

```text
vllm-gpu-deployment/
├── README.md                           # This complete guide and story
├── .env.example                        # Safe template for cloud credentials (no keys committed!)
├── .gitignore                          # Protects keys, logs, and environments
│
├── scripts/
│   ├── manage_lightning.py             # Local CLI: Start, stop, or check cloud GPU from Mac
│   ├── setup_gpu.sh                    # 1-Click setup: Installs vLLM & fixes NumPy on ANY cloud GPU
│   ├── launch_vllm.sh                  # Universal command: 32K context + Hermes tool parser
│   └── test_client.py                  # Test client to verify chat & tool completions
│
└── docs/
    ├── vllm_cloud_benchmark_guide.md   # Deep-dive engineering guide
    └── LLM_Inference_and_vLLM_Systems_Guide.pdf # 4-page publication-grade PDF cheat sheet
```

---

## 🚀 Quick Start (Reproduce in 5 Minutes)

### Step 1: Clone & Configure Credentials
```bash
git clone https://github.com/TheeVibeCoder/vllm-gpu-deployment.git
cd vllm-gpu-deployment

# Copy the environment template
cp .env.example .env
```
Open `.env` and enter your credentials (found at [lightning.ai](https://lightning.ai) $\rightarrow$ Settings $\rightarrow$ Keys):
```bash
LIGHTNING_API_KEY="your_api_key_here"
LIGHTNING_USER_ID="your_user_id_here"
LIGHTNING_USERNAME="your_username_here"
```

### Step 2: Wake Up the Cloud GPU & Start vLLM
```bash
# Power on GPU and launch vLLM 32K server
python3 scripts/manage_lightning.py start
```
*Outputs your live public OpenAI Base URL:*
```text
Public OpenAI Base URL: https://8000-xxxx.cloudspaces.litng.ai/v1
```

### Step 3: Test the Endpoint
```bash
# Query the live model from your terminal
python3 scripts/test_client.py https://8000-xxxx.cloudspaces.litng.ai/v1
```

### Step 4: Put the GPU to Sleep (Preserve Credits!)
```bash
# Shut down the GPU VM when you are finished
python3 scripts/manage_lightning.py stop
```

---

## 📄 Complete PDF Documentation
A formatted **4-page technical PDF guide** covering:
1. Complete performance benchmark tables.
2. 16 core LLM inference terminologies mapped to a single real-world prompt.
3. Deep-dive explanations (OpenAI compatibility, PagedAttention, TensorRT-LLM comparison).
4. Three classic AI infrastructure interview questions and mathematical breakdowns.

👉 Located in: [`docs/LLM_Inference_and_vLLM_Systems_Guide.pdf`](docs/LLM_Inference_and_vLLM_Systems_Guide.pdf)

---

## 🛡️ Security & Privacy
This repository contains **ZERO hardcoded API keys, tokens, or private credentials**. All authentication is managed via local environment variables. Never commit your `.env` file!

## 📜 License
MIT License. Open for educational and commercial use.
