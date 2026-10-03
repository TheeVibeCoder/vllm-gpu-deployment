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

## Raw SSD → GPU VRAM transfer benchmark (separate from HTTP TTFT)

`scripts/benchmark_ssd_gpu.cpp` creates a **4 MiB** temporary chunk on the directory you specify, writes and `fsync`s it, then times complete reads into GPU VRAM on the GPU host. By default it uses reproducible, nonzero synthetic KV-sized bytes; to use real KV bytes, supply a pre-existing raw KV chunk with `--input /path/to/kv-chunk.raw` (at least `--bytes` bytes). The first `--bytes` bytes become the fixture for all three paths:

| Mode | Timed operation | Availability |
|---|---|---|
| Buffered | `pread` into pageable host RAM + synchronous `cudaMemcpy` | CUDA GPU |
| Direct I/O | aligned `O_DIRECT` `pread` into CUDA-registered pinned RAM + `cudaMemcpyAsync` + stream synchronization | Linux mount and pinned memory support |
| cuFile | registered `cuFileRead` into GPU memory (synchronous) | cuFile library, compatible driver/mount; **P2P not proven by an API call** |

### Measured on the running Lightning Tesla T4 (2026-09-30)

Input: a **3,145,728-byte LMCache `.pt` KV chunk** already present on the Studio. The benchmark copied its bytes to a temporary file on each tested filesystem, then performed 3 warmups and 20 validated disk-to-VRAM transfers for each **successful** mode. These are **microseconds for the combined file read and GPU copy**, not HTTP response times:

| Fixture filesystem | Buffered mean / p95 | `O_DIRECT` + pinned mean / p95 | cuFile |
|---|---:|---:|---|
| `/tmp` (`overlay`, same filesystem as `/tmp/lmcache_disk`) | 8,894.4 / 9,180.9 µs | **2,107.1 / 2,179.3 µs** | Unavailable: handle registration failed (5027) |
| `/teamspace` (`ext4` on LVM) | 9,392.2 / 10,965.1 µs | **2,504.9 / 3,465.3 µs** | Unavailable: handle registration failed (5027) |

The combined direct-I/O **and pinned-memory** path was ~4.2× faster than the buffered/pageable path on the actual LMCache filesystem. This does **not** isolate the benefit of `O_DIRECT` from pinned memory or prove cold physical NVMe reads. Read-only `dmsetup table vg_data-lv_data` showed `/teamspace`'s LVM is **striped across two devices**: an Amazon EBS NVMe volume (`nvme1n1`) and an instance-local NVMe device (`nvme2n1`); it is **not a local-SSD-only mount**. `/tmp` is an overlay, so its physical backing cannot be identified from the container path alone. cuFile could not register the file handle on either mount; `nvidia-fs` reported `Ops Read=0` before/after, with I/O stats disabled. **No GDS transfer latency or P2P speedup was measured.** Do not reformat the NVMe device: it is part of an active striped volume.

On the **Linux GPU machine** with CUDA toolkit, cuFile headers/library and a writable **local NVMe** directory:

```bash
nvcc -std=c++17 -x cu scripts/benchmark_ssd_gpu.cpp -o benchmark_ssd_gpu -lcufile
# Omit --input for a synthetic 4 MiB KV-sized fixture instead.
./benchmark_ssd_gpu --disk-dir /path/to/local/nvme --bytes 4194304 --iterations 20 --warmups 3
# Or use a real raw KV chunk, if available:
./benchmark_ssd_gpu --disk-dir /path/to/local/nvme --input /path/to/kv-chunk.raw --bytes 4194304 --iterations 20 --warmups 3
```

The benchmark prints mean/p50/p95 elapsed microseconds and decimal MB/s for each completed mode. Fixture setup, memory registration and byte-for-byte GPU readback are outside the timed region; data is checked after **every** transfer. Unsupported modes are explicitly skipped, not silently simulated. The file is temporary and removed after the run. Run `./benchmark_ssd_gpu --help` for options. This is a raw transfer comparison, **not** an LMCache/vLLM integration or TTFT benchmark; it does not measure DirectKV.

For the buffered path, Linux `POSIX_FADV_DONTNEED` is requested between iterations, **outside** timing. This is advisory, not proof of cold NVMe reads; page-cache hits and storage/controller caches can still influence results. `O_DIRECT` bypasses Linux page cache where supported but does not guarantee data came from flash. cuFile can silently use host-memory compatibility mode, so do **not** report its result as direct P2P DMA until the **tested mount/device** is checked with `gdscheck -p` and a before/after comparison of `/proc/driver/nvidia-fs/stats` showing actual direct reads. `libcufile.so` presence alone is not proof.

On the GPU host, capture evidence for the cuFile mode before claiming peer-to-peer DMA:

```bash
if test -x /usr/local/cuda/gds/tools/gdscheck; then /usr/local/cuda/gds/tools/gdscheck -p; fi
cat /proc/driver/nvidia-fs/stats        # Snapshot before the benchmark
./benchmark_ssd_gpu --disk-dir /path/to/local/nvme --iterations 20
cat /proc/driver/nvidia-fs/stats        # Compare direct I/O counters and bytes
```

If `nvidia-fs` is absent or counters do not show direct reads, label that run **cuFile API only**, not proven GDS. Some hosts require root to view these counters.

On a Mac without CUDA, only the fixture/read/check smoke path can be exercised:

```bash
clang++ -std=c++17 -O2 scripts/benchmark_ssd_gpu.cpp -o /tmp/ssd-gpu-bench-smoke
/tmp/ssd-gpu-bench-smoke --cpu-only --disk-dir /tmp --iterations 3
```

The CPU-only output is **not** a GPU benchmark; use the T4 measurements above for GPU results and rerun on your own SSD mount to assess its backing storage.

---

## 📁 Repository Structure

```text
vllm-gpu-deployment/
├── README.md                           # This complete guide and story
├── .env.example                        # Safe template for cloud credentials (no keys committed!)
├── .gitignore                          # Protects keys, logs, and environments
│
├── configs/
│   └── lmcache.yaml                    # LMCache 3-tier offloading config (CPU RAM + NVMe)
│
├── scripts/
│   ├── manage_lightning.py             # Local CLI: Start, stop, or check cloud GPU from Mac
│   ├── setup_gpu.sh                    # 1-Click setup: Installs vLLM & fixes NumPy on ANY cloud GPU
│   ├── launch_vllm.sh                  # Universal command: 32K context + Hermes tool parser
│   ├── launch_vllm_lmcache.sh          # Native LMCacheConnectorV1 multi-tier launcher
│   ├── test_client.py                  # Test client to verify chat & tool completions
│   ├── test_prefix_caching.py          # Native prefix caching benchmark
│   ├── benchmark_ssd_gpu.cpp           # Raw SSD → GPU VRAM transfer benchmark (CUDA/cuFile)
│   └── benchmark_hybrid_cache.py       # Automated Cold vs Warm TTFT evaluation
│
└── docs/
    ├── tier3_local_nvme_ssd_guide.md   # Local SSD KV cache and transfer-path caveats
    ├── kv_cache_offloading_architecture_guide.md # Deep-dive KV offload & PagedAttention guide
    ├── kv_cache_offload_simulator.html # Interactive 3-Tier KV cache simulator
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

## ⚡ Next Level: KV Cache Offloading & Multi-Tier Hierarchy (LMCache + vLLM)

Beyond basic deployment, we explored scaling context retention and reducing Time to First Token (TTFT) by moving beyond single-tier GPU VRAM into **hybrid tiered memory architectures**.

### Architecture: Native Prefix Caching vs. Multi-Tier Offloading

```text
Option A: Native Prefix Caching (vLLM Only)
┌────────────────────────────────────────────────────────┐
│  Tier 1: GPU VRAM (Tesla T4)                           │
│  • PagedAttention (16-token pages, ~195 KB each)       │
│  • Radix Tree token prefix hash                        │
│  ⚠️ Evicted by GPU? DELETED PERMANENTLY.               │
└────────────────────────────────────────────────────────┘

Option B: Hybrid Tiered Offloading (vLLM + LMCache)
┌────────────────────────────────────────────────────────┐
│  Tier 1: GPU VRAM (Tesla T4)                           │
│  • Active tensor execution (~900 GB/s)                 │
└───────────────────────────┬────────────────────────────┘
                            │ PCIe DMA (~16 GB/s, Zero-Copy)
┌───────────────────────────▼────────────────────────────┐
│  Tier 2: Host CPU RAM (LMCache LocalCPUBackend)        │
│  • 5.0 GB pinned memory pool                           │
│  • Onload in ~10 ms (bypasses full prefill math!)      │
└───────────────────────────┬────────────────────────────┘
                            │ NVMe Interface (~3.5 GB/s)
┌───────────────────────────▼────────────────────────────┐
│  Tier 3: Local NVMe SSD (LMCache LocalDiskBackend)     │
│  • 10.0 GB persistent storage                          │
│  • Preserves context across server restarts            │
└────────────────────────────────────────────────────────┘
```

### Empirical Benchmarks Comparison (Tesla T4)

| Metric | Option A: Native Prefix Cache | Option B: vLLM + LMCache Hybrid |
| :--- | :--- | :--- |
| **Cold TTFT (Turn 1)** | 1,007 ms | 1,426 ms |
| **Warm TTFT (Turn 2)** | **456 ms** | **667 ms** |
| **Latency Reduction** | **54.7% faster** | **53.2% faster** |
| **Prefix Cache Hit Rate** | 49.7% | 63.4% |
| **Behavior on GPU Eviction** | **Data is lost forever.** Recompute required. | **Saved to CPU/SSD.** Restored in ~10 ms. |
| **Total Cache Pool** | ~7.0 GB | **~22.0 GB** (7GB VRAM + 5GB RAM + 10GB Disk) |

### Key Architectural Learnings
1. **PagedAttention is the enabler**: Memory is split into discrete 16-token pages (~195 KB). This paging enables offloading individual cold pages over PCIe without having to copy entire monolithic tensors.
2. **Native Upstream Connector**: While `lmcache_vllm` CLI wrapper broke due to vLLM 0.29.0 multimodal refactors, vLLM 0.29.0 natively includes `LMCacheConnectorV1` inside `vllm/distributed/kv_transfer/kv_connector/v1/factory.py`.
3. **PCIe vs. Industry Alternatives**:
   * **Tesla T4 PCIe Gen3 x16**: ~16 GB/s transfer via pinned memory DMA (~0.2 ms per 3MB chunk).
   * **Apple Silicon (MLX)**: Unified Memory Architecture (UMA) with shared physical pool up to 800 GB/s (True zero-copy).
   * **NVIDIA GH200 / NVLink-C2C**: 900 GB/s bidirectional interconnect.
   * **PD Disaggregation**: Prefill and Decode disaggregated across InfiniBand/RoCE (Mooncake).

👉 Full technical breakdown: [`docs/kv_cache_offloading_architecture_guide.md`](docs/kv_cache_offloading_architecture_guide.md)  
👉 Interactive visual simulator: [`docs/kv_cache_offload_simulator.html`](docs/kv_cache_offload_simulator.html)

---

## 🛡️ Security & Privacy
This repository contains **ZERO hardcoded API keys, tokens, or private credentials**. All authentication is managed via local environment variables. Never commit your `.env` file!


