# The Complete Engineering Guide to KV Cache, PagedAttention, and Tiered Offloading (vLLM + LMCache)

> **Context & Purpose**: This document captures the complete architectural journey, theoretical foundations, questions asked, technical roadblocks faced, and empirical benchmark results obtained while deploying and testing **vLLM 0.29.0** and **LMCache 0.5.5** on an **NVIDIA Tesla T4 GPU**.

---

## 1. Executive Summary & Architecture Evolution

During this project, we progressed across two fundamental paradigms of caching attention representations in Large Language Models (LLMs):

```
PHASE 1: Native Prefix Caching (vLLM Only)
┌────────────────────────────────────────────────────────┐
│  Tier 1: GPU VRAM (NVIDIA Tesla T4 - 16GB)             │
│  • PagedAttention (KV Cache blocks: 16 tokens/block)   │
│  • Radix Tree token prefix hashing                     │
│  • Eviction Policy: In-VRAM LRU                        │
│  ⚠️ If GPU fills up: Oldest KV blocks are DESTROYED    │
└────────────────────────────────────────────────────────┘

                         ⬇ Evolution to Multi-Tier

PHASE 2: Hybrid Tiered KV Cache Offloading (vLLM + LMCache)
┌────────────────────────────────────────────────────────┐
│  Tier 1: GPU VRAM (Tesla T4)                           │
│  • PagedAttention active execution (~900 GB/s)         │
└───────────────────────────┬────────────────────────────┘
                            │ PCIe Bus (~16 GB/s, DMA Zero-Copy)
┌───────────────────────────▼────────────────────────────┐
│  Tier 2: Host CPU RAM (LMCache LocalCPUBackend)        │
│  • 5.0 GB pinned memory pool                           │
│  • Onload time: ~10 ms (Prefill math bypassed!)        │
└───────────────────────────┬────────────────────────────┘
                            │ NVMe Storage Interface (~3.5 GB/s)
┌───────────────────────────▼────────────────────────────┐
│  Tier 3: Local NVMe SSD (LMCache LocalDiskBackend)     │
│  • 10.0 GB persistent storage                          │
│  • Restores context even across server restarts        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Fundamental Concepts & Q&A Breakdown

### Question 1: What actually is the KV Cache, and why does Prefill take so long?
* **Prefill Phase (Prompt Evaluation)**: When a prompt arrives (e.g., 500 tokens), the GPU Tensor Cores must compute full multi-head attention across all tokens simultaneously ($O(N^2)$ complexity). Every token computes:
  $$K = X \cdot W_k, \quad V = X \cdot W_v$$
  For a 0.5B model across 24 transformer layers, this matrix multiplication takes **~1,000 to 1,400 ms** (Cold TTFT).
* **Decode Phase (Token-by-Token Generation)**: To generate subsequent tokens, the model doesn't need to recompute past tokens *if* their Key ($K$) and Value ($V$) tensors are saved in memory.
* **The Problem**: KV cache memory grows linearly with context length and batch size ($2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{seq\_len} \times \text{bytes}$). A 16GB GPU runs out of VRAM quickly under heavy load.

---

### Question 2: Are the "blocks" in the simulator actually PagedAttention?
**Yes, absolutely.**
* **Traditional Attention (Pre-vLLM)**: Allocated one huge contiguous chunk of memory for every request. If a request was allocated 2048 tokens but only used 100, the remaining space was wasted (internal fragmentation). Crucially, you could not slice or offload partial tensors without copying the entire memory block.
* **PagedAttention (vLLM)**: Inspired by Virtual Memory Paging in Operating Systems.
  * Memory is partitioned into fixed-size **Pages / Blocks** (default: 16 tokens).
  * Each block stores $K$ and $V$ tensors for those 16 tokens across all layers (~195 KB per block for Qwen2.5-0.5B).
  * Blocks are mapped via a **Block Table** and can reside non-contiguously in memory.
  * **Why this matters for offloading**: Because memory is already divided into discrete 195 KB pages, LMCache can selectively pick cold pages and transfer them across the PCIe bus without disturbing the active pages!

---

### Question 2.1: Is there dedicated memory space per user/question?
**No, there is NO permanently reserved or dedicated space per user.**
* **The Shared Memory Pool**: GPU VRAM is treated like a shared hotel pool with thousands of small 16-token rooms.
* **Dynamic Checkout**: When a request arrives, vLLM dynamically checks out only the exact number of blocks needed for that specific prompt:
  $$\text{Blocks Checked Out} = \lceil \frac{\text{Prompt Length}}{16} \rceil$$
* **Multi-User Sharing**: When two users ask questions simultaneously, their blocks are interleaved anywhere in VRAM. When User A finishes, their blocks are instantly returned to the free pool for User B or User C to consume.

---

### Question 2.2: How do blocks stay connected when a 1,000-token question fills up? (The Block Table)
If you ask a **1,000-token question**, it needs:
$$\lceil \frac{1,000}{16} \rceil = 63 \text{ Blocks}$$
Because GPU memory is heavily fragmented, these 63 blocks are scattered randomly across physical VRAM (e.g., Physical Block #14, #92, #5, #110...).

**How they stay connected:**
The engine maintains a **Block Table** (lookup index) for each active request:

```text
               YOUR 1,000-TOKEN QUERY
┌─────────────────────────────────────────────────────────────┐
│  Logical Sequence:                                          │
│  Page 0 (Tokens 0-15)                                       │
│  Page 1 (Tokens 16-31)                                      │
│  ...                                                        │
│  Page 62 (Tokens 992-1000)                                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Mapped via
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 REQUEST'S BLOCK TABLE                       │
│                                                             │
│   Logical Page #      ──►      Physical GPU Block #         │
│   ─────────────────────────────────────────────────         │
│   Page 0              ──►      Physical Block #14           │
│   Page 1              ──►      Physical Block #92           │
│   Page 2              ──►      Physical Block #5            │
│   ...                          ...                          │
│   Page 62             ──►      Physical Block #8            │
└─────────────────────────────────────────────────────────────┘
```

**Lifecycle as tokens generate (Decode Phase):**
1. **Unfilled Slots**: Page 62 (Physical #8) holds tokens 992–1000 (8 tokens used, 8 empty slots remaining).
2. **Filling Slots**: Generated tokens 1,001 through 1,008 fill those remaining 8 slots inside Physical Block #8. No new memory allocation is needed.
3. **Crossing Block Boundary**: When Token 1,009 is generated, Physical #8 is 100% full. vLLM pulls **Physical Block #44** from the free pool and appends `Page 63 ──► Physical #44` to the Block Table.
4. **Attention Execution**: The PagedAttention CUDA kernel reads the Block Table sequentially like a table of contents, gathering the scattered tensors on the fly without requiring contiguous VRAM.

---

### Question 3: Do the pages contain the prefix cache tensors?
**Yes.**
* Each page contains the computed $K$ and $V$ projection matrices for its 16 tokens:
  $$\text{Page Size} = 16 \text{ tokens} \times 24 \text{ layers} \times 2 (\text{heads}) \times 64 (\text{head dim}) \times 2 (\text{FP16 bytes}) \times 2 (K \text{ and } V) \approx 196,608 \text{ bytes} \ (\sim 192 \text{ KB})$$
* **Prefix Caching** hashes the token sequence in a Radix Tree. When a new prompt starts with the same system instructions or prefix, vLLM reuses these exact tensor pages, completely skipping the Tensor Core matrix multiplications.

---

### Question 4: How does data transfer between tiers? What are PCIe, Pinned Memory, and DMA?
* **PCIe Bus (Peripheral Component Interconnect Express)**: The hardware physical lane connecting the GPU card to the CPU motherboard. On Tesla T4 (PCIe Gen3 x16), bandwidth is capped at **~15.75 GB/s**.
* **Pageable vs Pinned (Page-Locked) Memory**:
  * Normally, CPU memory can be paged out by the OS kernel to swap space, meaning its physical address can change. The GPU cannot safely read this memory directly.
  * **Pinned Memory (`cudaHostAlloc`)**: Tells the OS *never* to move or swap this physical RAM address.
* **DMA (Direct Memory Access)**: With pinned memory, the GPU's DMA controller copies data directly over the PCIe bus without requiring CPU thread cycles.
  $$\text{Transfer Time for 1 KV chunk (3 MB)} = \frac{3 \text{ MB}}{15,750 \text{ MB/s}} \approx 0.19 \text{ ms}$$
  This is why offloading happens in **sub-millisecond** intervals in the background.

---

### Question 5: Does NVIDIA or the industry have alternatives to PCIe?
* **Apple Silicon (M-series / MLX)**: Uses **Unified Memory Architecture (UMA)**. The CPU, GPU, and Neural Engine share one physical pool of high-speed LPDDR5 memory (up to 800 GB/s on M-Max/Ultra). There is **zero PCIe bus bottleneck** because no data copy is needed—it is a true pointer handoff (Zero-Copy).
* **NVIDIA Grace Hopper (GH200 / NVLink-C2C)**: Replaces PCIe with NVLink Chip-to-Chip running at **900 GB/s bidirectional** (7× faster than PCIe Gen5). CPU RAM acts as an extension of GPU VRAM with cache coherence.
* **AMD MI300A**: Accelerated Processing Unit (APU) combining Zen 4 CPU cores and CDNA 3 GPU cores sharing unified HBM3 memory (5.3 TB/s).
* **CXL (Compute Express Link)**: Open standard over PCIe Gen5/Gen6 allowing shared memory pools across heterogeneous compute nodes.
* **Prefill/Decode (PD) Disaggregation (Mooncake / Splitwise)**: Separates prefill GPUs (compute-heavy) from decode GPUs (memory bandwidth-heavy), streaming KV blocks over 400Gbps InfiniBand/RoCE RDMA networks.

---

### Question 6: Eviction Policies — Size Limit vs Time-Based Aging
* **Size-Based LRU (Least Recently Used)**: Evicts oldest blocks only when memory reaches 90-95% capacity.
  * *Downside*: Stale user sessions occupy expensive VRAM indefinitely while traffic is low.
* **Time-Based Aging (TTL - Time to Live)**: If a user hasn't queried the model for $T$ seconds (e.g., 30s):
  1. Move KV blocks from Tier 1 (GPU) $\rightarrow$ Tier 2 (Host RAM).
  2. If inactive for 5 minutes $\rightarrow$ Tier 3 (NVMe SSD).
  3. If inactive for 1 hour $\rightarrow$ Discard / Flush.
* **Hybrid Approach (Industry Standard - S3-FIFO / Two-Q)**: Uses both frequency and recency queues combined with a decay timer so one-off exploratory prompts don't flush long-term cached knowledge.

---

## 3. Roadblocks Faced & How We Solved Them

### Roadblock 1: NumPy 2.0 Compatibility Crash
* **Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`.
* **Root Cause**: Installing modern helper packages upgraded NumPy to 2.x, breaking pre-compiled C-extensions in scikit-learn and PyTorch.
* **Fix**: Pinned environment to `numpy<2.0.0`.

### Roadblock 2: The `lmcache_vllm` Wrapper Import Error
* **The Confusion**: We initially assumed `lmcache_vllm` was required to use LMCache with vLLM.
* **Error**:
  ```python
  ImportError: cannot import name 'MultiModalInputs' from 'vllm.multimodal'
  ```
* **Root Cause**: `lmcache_vllm` (v0.6.2.3) is an **outdated CLI launcher** designed for older vLLM releases. In vLLM 0.29.0, multimodal interfaces were refactored, breaking the external wrapper.
* **The Discovery from Official Docs**: vLLM 0.29.0 already includes a **natively integrated connector** (`LMCacheConnectorV1`) inside `vllm/distributed/kv_transfer/kv_connector/v1/factory.py`.
* **The Resolution**: We eliminated `lmcache_vllm` completely and launched standard `vllm serve` using native flags:
  ```bash
  LMCACHE_CONFIG_FILE=/teamspace/studios/this_studio/lmcache.yaml \
  vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  ```

---

## 4. Empirical Benchmark Results (Tesla T4 Cloud GPU)

Both options were benchmarked using a 101-token prompt with identical model parameters on our live cloud instance:

### Option A: vLLM Native Prefix Caching (GPU VRAM Only)
```
Turn 1 (Cold - Full Prefill Math):    1,007 ms  |  Hit Rate: 0.0%
Turn 2 (Warm - GPU Prefix Hit):         456 ms  |  Hit Rate: 49.7%
------------------------------------------------------------------
Result: 54.7% TTFT Reduction | Cache Capacity: ~7 GB (VRAM only)
Limitation: If evicted by other users, KV is PERMANENTLY DESTROYED.
```

### Option B: vLLM + LMCache Hybrid (3-Tier Offloading)
```
Turn 1 (Cold - Full Prefill Math):    1,426 ms  |  LMCache Store triggered
Turn 2 (Warm - Prefix + Engine Hit):    667 ms  |  Hit Rate: 63.4%
Turn 3 (Hot - Confirmed Cached):        692 ms  |  Hit Rate: 63.4%
------------------------------------------------------------------
Result: 53.2% TTFT Reduction | Cache Capacity: ~22 GB (GPU + RAM + SSD)
Advantage: Evicted GPU blocks survive in CPU RAM / SSD and reload in ~10 ms.
```

---

## 5. Architectural Comparison Matrix

| Architectural Dimension | Option A: Native Prefix Cache | Option B: LMCache Hybrid Offload |
| :--- | :--- | :--- |
| **Backing Software** | vLLM Core | vLLM + LMCache Core Engine |
| **Available Tiers** | Tier 1 (GPU VRAM) only | Tier 1 (GPU) + Tier 2 (CPU RAM) + Tier 3 (NVMe) |
| **Total Cache Pool** | ~7.0 GB | ~22.0 GB (7GB VRAM + 5GB RAM + 10GB Disk) |
| **Behavior on GPU Eviction** | **Data is deleted.** Future query must re-run prefill. | **Data is offloaded.** Future query loads in ~10 ms. |
| **Transfer Mechanism** | In-VRAM pointer remapping | Pinned Memory PCIe DMA & NVMe serialization |
| **Inter-request Sharing** | Yes (if still resident in VRAM) | Yes (cross-request, cross-session, persistent) |
| **Best Used For** | High concurrency with short, continuous dialogues | Long multi-turn conversations, idle users, agentic loops |

---

## 6. How to Run and Reproduce

### Server Configuration File (`lmcache.yaml`)
```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
local_disk: /teamspace/studios/this_studio/lmcache_disk
max_local_disk_size: 10.0
cache_policy: lru
save_decode_cache: true
```

### Server Startup Command
```bash
LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Verification in Server Logs
Confirm the following log entries appear during startup:
* `Creating v1 connector with name: LMCacheConnectorV1`
* `Created backend: LocalCPUBackend`
* `Created backend: LocalDiskBackend`
* `Prefix cache hit rate: >0.0%`
