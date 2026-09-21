# 🚀 vLLM Cloud GPU Deployment & KV Cache Benchmark Guide

> **A complete, clear reference guide on deploying models with vLLM on cloud GPUs, measured benchmarks, PagedAttention internals, and troubleshooting real-world agentic workloads.**

---

## 📑 Table of Contents
1. [Executive Summary & Measured Benchmarks](#1-executive-summary--measured-benchmarks)
2. [The Deployment Architecture](#2-the-deployment-architecture)
3. [Step-by-Step: How We Deployed the Model](#3-step-by-step-how-we-deployed-the-model)
4. [Deep-Dive: Questions & Concepts Explained](#4-deep-dive-questions--concepts-explained)
   - [Q1: What is an OpenAI-Compatible API vs. a Regular REST API?](#q1-what-is-an-openai-compatible-api-vs-a-regular-rest-api)
   - [Q2: Why Did Tool Calling Fail with Error 400?](#q2-why-did-tool-calling-fail-with-error-400)
   - [Q3: What Does "Breaking Memory into Chunks" Mean? (PagedAttention)](#q3-what-does-breaking-memory-into-chunks-mean-pagedattention)
   - [Q4: What Happens to Empty Page Slots on Short Prompts like "Hey"?](#q4-what-happens-to-empty-page-slots-on-short-prompts-like-hey)
   - [Q5: How Does NVIDIA TensorRT-LLM Compare?](#q5-how-does-nvidia-tensorrt-llm-compare)
   - [Q6: Why Did the First Request Take 22 Seconds?](#q6-why-did-the-first-request-take-22-seconds)
   - [Q7: Why Did Context Overflow and How Did Scaling to 32K Fix It?](#q7-why-did-context-overflow-and-how-did-scaling-to-32k-fix-it)
5. [Reproducible Scripts & Commands](#5-reproducible-scripts--commands)

---

## 1. 📊 Executive Summary & Measured Benchmarks

During our live test deployment on an **NVIDIA Tesla T4 GPU (16 GB VRAM)** using **`Qwen/Qwen2.5-0.5B-Instruct`**, we captured the following real-world operational benchmarks:

### 🏎️ Performance & Throughput Metrics
| Metric | Measured Value | Meaning & Context |
|---|---|---|
| **Decode Throughput (Generation)** | **`76.83 tokens/sec`** | Time to stream tokens. ~10× faster than average human reading speed (~8 tokens/s). |
| **Prefill Throughput (Prompt Reading)** | **`1,893.7 tokens/sec`** | Rate at which the GPU absorbs and computes initial prompt tokens. |
| **Time to First Token (Warm Cache)** | **`< 250 ms`** | Ultra-responsive once CUDA graphs and kernels are warm. |
| **CUDA Graph Warmup Time** | **`4.8 – 6.0 seconds`** | Fast one-time graph compilation at server startup. |

### 🧠 Memory & KV Cache Footprint
| Resource | Allocation | Details |
|---|---|---|
| **Model Weights (In VRAM)** | **`0.93 GiB`** | Qwen 2.5 0.5B in FP16 precision takes less than 1 GB of VRAM. |
| **Paged KV Cache Allocation** | **`6.99 GiB`** | VRAM reserved exclusively for dynamic conversation memory. |
| **Total KV Cache Slots** | **`590,512 tokens`** | Total number of tokens the GPU can actively hold simultaneously. |
| **Max Concurrent Requests** | **`576 parallel users`** | Calculated capacity at 1,024 context tokens per user. |
| **Native Context Window** | **`32,768 tokens (32K)`** | Expanded to allow large agent prompts, file trees, and tools. |

### 💰 Cost & Efficiency Numbers
| Parameter | Value | Notes |
|---|---|---|
| **Cloud GPU Instance** | NVIDIA Tesla T4 (16 GB) | Hosted via Lightning AI Studio. |
| **Instance Rate** | **~$0.40 – $0.55 / hr** | Billed by the second. |
| **Total Test Session Time** | **~35 minutes** | Includes installation, tuning, and benchmark testing. |
| **Total Session Cost** | **~$0.25** | Less than 30 cents total from free starter credits. |
| **Remaining Balance** | **`4.40 credits`** | Gives **~8 to 10 more hours** of free GPU compute. |

---

## 2. 🏗️ The Deployment Architecture

```text
┌────────────────────────────────────────────────────────┐
│             YOUR MAC / LOCAL AGENT (OMP)               │
│  • Holds all tools: bash, file_editor, git, web        │
│  • Configured in ~/.omp/agent/models.yml               │
│  • Context window set to: 32,768 tokens                │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ HTTPS / JSON Request (OpenAI API spec)
                            │ tools: [...] + tool_choice: "auto"
                            ▼
┌────────────────────────────────────────────────────────┐
│         LIGHTNING AI CLOUD (NVIDIA Tesla T4)           │
│  • Studio: <your-studio-name>                          │
│  • Public Port: 8000 (cloudspaces.litng.ai)            │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │              vLLM Serving Engine                 │  │
│  │  • PagedAttention (16 tokens / block)            │  │
│  │  • Hermes Tool Parser (extracts <tool_call>)     │  │
│  │  • Dynamic Continuous Batching                   │  │
│  │  • Model: Qwen/Qwen2.5-0.5B-Instruct             │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 3. 🛠️ Step-by-Step: How We Deployed the Model

### Step 1: Remote Cloud Control via Python SDK
Instead of fragile web browser tunnels, we used the `lightning-sdk` directly from the local Mac:
```python
from lightning_sdk import Studio, Machine, Teamspace

teamspace = Teamspace('general')
studio = Studio('your-studio-name', teamspace=teamspace)
studio.start(machine=Machine.T4) # Powers on NVIDIA T4 GPU
```

### Step 2: Dependency Resolution inside Cloud Linux
We fixed a common NumPy 2.x breaking change (`cannot import name 'Inf' from 'numpy'`):
```bash
pip install vllm
pip install "numpy<2.0.0"
```

### Step 3: Launching the vLLM OpenAI-Compatible Server
We launched the background server with full tool-calling and 32K context:
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > vllm.log 2>&1 &
```

### Step 4: Exposing the Public HTTPS URL
```python
studio.add_ports(8000)
# Returns: https://8000-<cloudspace-id>.cloudspaces.litng.ai/v1
```

---

## 4. 🧠 Deep-Dive: Questions & Concepts Explained

### Q1: What is an OpenAI-Compatible API vs. a Regular REST API?
* **Regular REST API**: Every developer makes up their own URLs and payloads (e.g. `POST /predict`, `POST /chat`, `POST /generate`). No two APIs are the same, meaning you must write custom integration code for every tool.
* **OpenAI-Compatible API**: The AI industry's **universal standard**.
  * Endpoint is always `/v1/chat/completions`.
  * Request is always `{"messages": [{"role": "user", "content": "..."}]}`.
  * Response is always `{"choices": [{"message": {"content": "..."}}]}`.
* **Why it matters**: Every software product in the world (Cursor, Continue, LangChain, Aider, Open WebUI, OMP) can connect to your model simply by pointing `base_url` to your server. Zero custom code required.

---

### Q2: Why Did Tool Calling Fail with Error 400?
* **What Happened**: The client sent `"tool_choice": "auto"` along with tool definitions. vLLM rejected it with:
  `✘ 400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`
* **The Root Cause**: Neural networks **cannot understand JSON or run code**. They only output flat text. When Qwen decides to call a tool, it outputs raw XML text:
  ```text
  <tool_call>
  {"name": "get_weather", "arguments": {"location": "Tokyo"}}
  </tool_call>
  ```
* **The Fix**: The server needs `--tool-call-parser hermes` so it knows how to intercept those XML tags, extract the JSON, and wrap it into a structured OpenAI response object before replying to your client.

---

### Q3: What Does "Breaking Memory into Chunks" Mean? (PagedAttention)
* **The Old Naive Way**: Systems pre-allocated huge, contiguous strips of GPU RAM (e.g., 2,048 slots in a straight line) *in case* a user typed a long question. If you only typed 5 words, the remaining 2,043 slots were locked and wasted. Up to 80% of VRAM was dead space.
* **The vLLM PagedAttention Way**: Inspired by OS Virtual Memory Paging.
  * Memory is broken into tiny fixed pages (**16 tokens per page**).
  * Pages can be scattered anywhere across physical GPU memory (Page 1 at slot #10, Page 2 at slot #800).
  * A **Page Table** maps the sequence. 
  * As you talk, vLLM dynamically grabs any free page anywhere in RAM. Wasted memory drops to **< 3%**.

---

### Q4: What Happens to Empty Page Slots on Short Prompts like "Hey"?
* A page holds **16 tokens**. If you send `"Hey"` (3 tokens), what happens to slots 4–16?
* **They are not wasted!** The model immediately begins generating its reply token-by-token:
  * Token 4: `"Hello"`
  * Token 5: `"!"`
  * Token 6: `"How"` ...
* The incoming generated tokens **fill up the rest of the page**. Only when token 17 is reached does vLLM allocate a second page.
* If generation stops at Token 8, the remaining 8 slots represent "internal fragmentation," but this waste is microscopic (~100 KB), not gigabytes.

---

### Q5: How Does NVIDIA TensorRT-LLM Compare?
* **Did NVIDIA use the same idea?** **Yes!** TensorRT-LLM adopted the exact same concept, calling it **Paged KV Cache**.
* **Differences**:
  * **vLLM**: Python-first, instant setup, runs any Hugging Face model in 1 line of code. Defaults to 16 tokens/page.
  * **TensorRT-LLM**: C++ and CUDA-first. Requires an ahead-of-time build step (`trtllm-build`). Uses larger block sizes (64 or 128 tokens) to optimize for NVIDIA memory bus bursts (128-byte cache lines).
* **Takeaway**: Startups and developers use vLLM for agility; massive hyperscalers use TensorRT-LLM to squeeze the final 10% performance out of H100 clusters.

---

### Q6: Why Did the First Request Take 22 Seconds?
* **6,145 Input Tokens**: Your agent sent its entire system prompt, tool schemas, and file trees along with your short message.
* **Triton JIT Kernel Compilation**: Because 6,145 tokens was a new shape the server had never seen, PyTorch Triton paused to compile a custom CUDA kernel:
  `WARNING: Triton kernel JIT compilation during inference: kernel_unified_attention. This causes a latency spike.`
* **Result**: A one-time 15-second compilation delay. On the second and subsequent turns, the kernel was already warm, dropping latency to milliseconds.

---

### Q7: Why Did Context Overflow and How Did Scaling to 32K Fix It?
* **The Problem**: 
  * Agent prompt = 6,145 tokens.
  * Requested output = 2,048 tokens.
  * Total required = **8,193 tokens**.
  * Because our server had an 8,192 limit, it exceeded it by **1 token** ($8,193 > 8,192$).
* **The Solution**: 
  * Qwen 2.5 0.5B has a native context limit of **32,768 tokens (32K)**.
  * Because the model only weighs 0.93 GB, a 32K KV cache only takes ~400 MB of extra VRAM.
  * We scaled vLLM to `--max-model-len 32768` and set `contextWindow: 32768` in `~/.omp/agent/models.yml`.
  * Your 6,145-token agent prompt now takes only **~18%** of the context, leaving **25,000+ free tokens** of headroom!

---

## 5. 💻 Reproducible Scripts & Commands

### Quick Server Wakeup Script (`start_gpu.py`)
```python
import os
os.environ['LIGHTNING_API_KEY'] = 'your_api_key_here'
os.environ['LIGHTNING_USER_ID'] = 'your_user_id_here'
os.environ['LIGHTNING_USERNAME'] = 'your_username_here'

from lightning_sdk import Studio, Machine, Teamspace

s = Studio('your-studio-name', teamspace=Teamspace('general'))
s.start(machine=Machine.T4)

# Launch vLLM with 32K context and Hermes tool parsing
cmd = "nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.75 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes > /teamspace/studios/this_studio/vllm.log 2>&1 &"
s.run(cmd)
print("vLLM 32K Server is UP!")
```

### Quick Server Shutdown Script (`stop_gpu.py`)
```python
import os
from lightning_sdk import Studio, Teamspace

s = Studio('your-studio-name', teamspace=Teamspace('general'))
s.stop()
print("GPU Stopped. Credits safe!")
```
