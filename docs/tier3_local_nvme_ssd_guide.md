# 💾 Tier 3: Local NVMe SSD KV Cache Offloading Guide

> **Purpose**: A comprehensive reference guide on using Local NVMe SSDs as a persistent, high-capacity Tier 3 storage layer for LLM Key-Value (KV) caching in vLLM and LMCache.

---

## 1. Executive Overview: The 3-Tier Storage Hierarchy

In modern LLM inference architectures, memory is organized into three distinct tiers based on the trade-off between **speed (bandwidth/latency)** and **cost/capacity**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: GPU VRAM (Tesla T4 - 16 GB)                                    │
│ Bandwidth: ~300–900 GB/s  |  Access Latency: ~0.1 µs                  │
│ Role: Active execution. Fast, expensive, tiny capacity (~7 GB KV pool).│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ PCIe Gen3 x16 (~16 GB/s, DMA)
┌───────────────────────────────────▼────────────────────────────────────┐
│ TIER 2: Host CPU RAM (5–32 GB)                                         │
│ Bandwidth: ~16 GB/s       |  Transfer Latency: ~10 ms (Pipelined)      │
│ Role: Staging buffer for recently evicted or warm sessions.           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ NVMe Interface (~3.5–7.0 GB/s)
┌───────────────────────────────────▼────────────────────────────────────┐
│ TIER 3: Local NVMe SSD (50–500 GB)                                     │
│ Bandwidth: ~3.5 GB/s      |  Transfer Latency: ~30–50 ms (Pipelined)   │
│ Role: Cold, persistent storage. Preserves cache across server restarts!│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Latency Breakdown: How Fast / Slow is Local SSD?

### The Physical Numbers

| Storage Tier | Physical Medium | Access Latency | Bandwidth | Read Time for 1 Chunk (3 MB / 256 tokens) |
| :--- | :--- | :---: | :---: | :---: |
| **Tier 1 (GPU)** | GPU HBM / GDDR6 | **~0.1 $\mu$s** | **300–900 GB/s** | **~0.005 ms** (Instant) |
| **Tier 2 (CPU)** | Host DDR4 / DDR5 RAM | **~80 ns** | **~16 GB/s** (PCIe) | **~0.19 ms** ($190 \ \mu\text{s}$) |
| **Tier 3 (SSD)** | Local NVMe SSD (Gen3/4) | **~50–100 $\mu$s** | **~3.5 GB/s** | **~0.8 to 1.5 ms** |
| **Recomputation**| Full GPU Attention Math | **$O(N^2)$ Matrix Ops** | VRAM Compute | **~10 to 50 ms** per chunk |

### End-to-End Latency for a 4,000-Token Context:
* **Cold Start (Full Prefill on GPU)**: **~1,500 – 2,500 ms** (Compute-heavy matrix multiplication)
* **Tier 2 (Reload from CPU RAM)**: **~10 – 15 ms**
* **Tier 3 (Reload from NVMe SSD)**: **~35 – 50 ms**

> **The Golden Insight**: Even though an NVMe SSD is ~3× slower than CPU RAM, **loading from SSD is still 30× to 50× FASTER than recalculating attention math on the GPU!**

---

## 3. The Golden Rule: Recomputation vs. I/O Crossover Threshold

Research from USENIX OSDI '26 (*DirectKV*) and LMCache benchmarks highlights that SSD caching is not always faster for every prompt size:

$$\text{If } \text{Disk Read Latency} > \text{GPU Recomputation Time} \longrightarrow \mathbf{\text{Do NOT use Disk!}}$$

```text
                       THE BREAK-EVEN POINT
     Latency
        ▲
        │                      / (GPU Recomputation explodes quadratically)
        │                     /
        │                    /
        │    RECOMPUTE      /         SSD LOAD WINS (HUGE SPEEDUP!)
        │      FASTER      /
        │        │        /
        │  ──────┼───────/──────────────── (SSD I/O has fixed overhead ~15ms)
        │        │      /
        │        │     /
        │        ▼    /
        └────────────┴─────────────────────────────► Context Length (Tokens)
                 ~128 tokens
```

* **Short Prompts (< 128 tokens)**: Tensor Cores calculate attention in **~5 ms**. Finding files on disk, reading headers, and transferring across PCIe takes **~10–15 ms**. Here, **recomputing is faster than hitting the SSD**.
* **Long Prompts (1,000 to 32,000 tokens)**: Recomputation takes **1,500 ms to 15,000 ms**. Reading from SSD takes only **~30 to 150 ms**. Here, **SSD offloading provides a massive 95%+ latency reduction**.

---

## 4. State-of-the-Art Approaches on the Internet

### 1. LMCache (`local_disk` with Direct I/O & Prefetching)
* **Asynchronous `put()`**: Disk writes happen in a dedicated background worker thread; the inference loop never freezes.
* **Direct I/O (`use_odirect: true`)**: Bypasses the Linux page cache and buffer copy overhead, streaming directly from NVMe into pinned host memory.
* **Asynchronous Prefetching**: Predicts which KV blocks are needed next and begins reading them from SSD into RAM before the model reaches those attention layers.

### 2. DirectKV (OSDI 2026 Research Paper)
* **Zero-Copy Architecture**: Eliminates the intermediate hop (`SSD -> Host RAM -> GPU VRAM`). Enables the GPU to access memory-mapped host address spaces directly across NVLink-C2C interconnects.

### 3. NVIDIA GPUDirect Storage (GDS)
* **Direct DMA**: Uses PCIe peer-to-peer DMA between the NVMe controller and GPU memory, completely bypassing CPU host memory and CPU cores. Reaches raw wire speeds (up to 14 GB/s on PCIe Gen5 NVMe).

### 4. DeepSpeed-ZeRO-Inference
* Pioneered early NVMe offloading, but suffered from synchronous I/O blocks (causing decode stutter or "freezes" during generation).

---

## 5. Production Configuration Template (`lmcache.yaml`)

```yaml
# LMCache Multi-Tier Configuration with Local NVMe SSD
chunk_size: 256                         # 256 tokens per chunk (~3.1 MB for 0.5B model)
cache_policy: lru                       # Least Recently Used eviction across tiers

# Tier 2: Host CPU RAM
local_cpu: true
max_local_cpu_size: 5.0                 # 5.0 GB in host memory

# Tier 3: Local NVMe SSD
local_disk: /path/to/fast_nvme/lmcache  # Mount path of high-speed local NVMe drive
max_local_disk_size: 50.0               # Max storage pool size (e.g., 50 GB)

# Advanced Performance Flags
use_odirect: true                       # Bypass OS page cache for direct NVMe DMA
save_decode_cache: true                 # Save generation tokens in addition to prefill
min_retrieve_tokens: 64                 # Skip disk I/O if prompt is smaller than 64 tokens
```

---

## 6. Ideal Real-World Use Cases for Tier 3 SSD

1. **Autonomous Agents (e.g., Devin, OMP, AutoGPT)**:
   * Agents execute long tool runs with 10–20 minute gaps between user inputs. Tier 3 preserves their complete 20,000-token scratchpad without occupying GPU VRAM.
2. **Enterprise RAG (Retrieval-Augmented Generation)**:
   * Reusing company policy documents or static codebases across thousands of employee queries.
3. **Multi-User Interactive Chatbots**:
   * Storing returning user conversational history so users who return after lunch don't trigger costly cold prefill recomputations.
