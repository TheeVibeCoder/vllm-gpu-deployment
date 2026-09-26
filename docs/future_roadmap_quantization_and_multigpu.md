# 🚀 Next-Phase Engineering Roadmap: Quantization, Disaggregated Serving & Multi-GPU Scale

> **Purpose**: This roadmap outlines the strategic next learning objectives and implementation phases for advancing our LLM inference infrastructure from a single small model to production-scale serving systems.

---

## 🗺️ The 3 Future Engineering Horizons

```text
PHASE 1: Weight Quantization & Scaling (Single GPU)
┌─────────────────────────────────────────────────────────────┐
│  • Scale from 0.5B ──► 7B / 8B / 14B                        │
│  • AWQ (Activation-aware Weight Quantization) 4-bit         │
│  • GPTQ vs FP8 vs INT4 trade-offs (Perplexity vs VRAM)      │
│  • Stress-test LMCache under true VRAM pressure             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
PHASE 2: Prefill vs. Decode Optimization (1-GPU & 2-GPU)
┌─────────────────────────────────────────────────────────────┐
│  • Single-GPU: Chunked Prefill (Piggybacking generation)    │
│  • Single-GPU: Inter-Token Latency (ITL) jitter elimination │
│  • 2-GPU / Multi-GPU: Physical PD Disaggregation            │
│  • KV cache streaming across high-speed interconnects       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
PHASE 3: Large-Scale Distributed Multi-GPU Serving
┌─────────────────────────────────────────────────────────────┐
│  • Scale to 32B / 70B models (DeepSeek, Llama-3-70B)        │
│  • Tensor Parallelism (TP) across multiple GPUs (Megatron)  │
│  • Pipeline Parallelism (PP) across nodes                   │
│  • Ray cluster management with vLLM distributed engine      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📌 Topic 1: Scaling to Bigger Models & Quantization Deep Dive (AWQ, GPTQ, FP8)

### What We Will Learn:
1. **The VRAM Formula**:
   $$\text{Weight VRAM} = \text{Parameters (Billions)} \times \text{Bytes per Parameter}$$
   * FP16 (16-bit): $7\text{B} \times 2\text{ bytes} = \mathbf{14 \text{ GB}}$ (Leaves no room for KV cache on a 16GB GPU!)
   * INT4 (4-bit / AWQ): $7\text{B} \times 0.5\text{ bytes} = \mathbf{3.5 \text{ GB}}$ (**Leaves >10 GB for KV cache!**)
2. **AWQ (Activation-aware Weight Quantization)**:
   * Why naive INT4 quantization makes models stupid (destroys perplexity).
   * How AWQ protects the top 1% "salient weights" (based on activation magnitudes) in FP16 while compressing the other 99% into 4-bit integers.
3. **Quantization Landscape Comparison**:
   * **AWQ**: Best for low-batch, latency-sensitive inference (preserves reasoning capabilities).
   * **GPTQ**: Layer-by-layer second-order error minimization (very fast on older GPUs).
   * **FP8 (Floating Point 8)**: Native support on Ada Lovelace (L4/RTX 4090) and Hopper (H100). Same range as FP16 with 2× memory reduction and hardware tensor acceleration.
   * **BitsAndBytes (NF4 / QLoRA)**: Best for fine-tuning, but slower for production high-concurrency inference.

### Action Item for Phase 1:
* Deploy **`Qwen/Qwen2.5-7B-Instruct-AWQ`** or **`meta-llama/Llama-3.1-8B-Instruct-AWQ`** on our Tesla T4.
* Measure inference throughput, TTFT, and observe LMCache Tier 2 (CPU RAM) offload trigger under real memory pressure.

---

## 📌 Topic 2: Prefill and Decode (1-GPU vs. 2-GPU Architecture)

### What We Will Learn:
1. **The Fundamental Conflict**:
   * **Prefill is Compute-Bound**: Matrix multiplication on prompt tokens saturates Tensor Cores ($O(N^2)$ attention). High GPU utilization, low memory bandwidth sensitivity.
   * **Decode is Memory-Bandwidth Bound**: Generating 1 token at a time requires reading the entire model weights and KV cache from VRAM to SRAM ($O(1)$ arithmetic intensity). Tensor Cores sit 90% idle waiting for VRAM memory buses.
2. **Single-GPU Solution: Chunked Prefill**:
   * Without Chunking: A 4,000-token prompt blocks all existing active generation streams for 2 seconds (causes massive Inter-Token Latency spikes / jitter).
   * With Chunking (`--enable-chunked-prefill`): Slices prompt into 512-token chunks and batches prefill slices together with decode steps.
3. **2-GPU / Multi-GPU Solution: Physical PD Disaggregation**:
   * GPU 1 (Prefill Worker): Specialized in prompt processing (e.g., NVIDIA H100).
   * GPU 2 (Decode Worker): Specialized in memory bandwidth (e.g., L40S or multiple smaller GPUs).
   * KV Transfer: Streaming generated KV blocks from GPU 1 $\rightarrow$ GPU 2 over NVLink or RDMA networks using LMCache / Mooncake connectors.

### Action Item for Phase 2:
* **Single GPU**: Benchmark TTFT vs ITL jitter under 5 concurrent users with Chunked Prefill enabled vs disabled.
* **Dual GPU (Future Expansion)**: Spin up a 2-GPU instance to physically disaggregate prefill and decode instances.

---

## 📌 Topic 3: Distributed Multi-GPU Serving (70B+ Scale)

### What We Will Learn:
1. **Tensor Parallelism (TP)**:
   * Slicing attention heads and MLP weight matrices across multiple GPUs (e.g., 2×, 4×, or 8× GPUs).
   * Requires ultra-fast interconnects (NVLink $\ge 600 \text{ GB/s}$) due to high-frequency `All-Reduce` communication at every transformer layer.
2. **Pipeline Parallelism (PP)**:
   * Splitting model layers sequentially across GPUs (e.g., Layers 1–40 on GPU 1, Layers 41–80 on GPU 2).
   * Tolerate slower interconnects (PCIe / Ethernet), but introduces pipeline bubbles.
3. **Expert Parallelism (EP) for MoE**:
   * Routing Mixture-of-Experts tokens (e.g., Mixtral 8x7B or DeepSeek-V3) to distinct GPU workers.

### Action Item for Phase 3:
* Test multi-GPU vLLM launch commands using `--tensor-parallel-size 2` and evaluate inter-GPU scaling efficiency.

---

## 🗓️ Next Steps Checklist

- [ ] **Step 1**: Study AWQ activation-aware quantization mechanics and compare against GPTQ/FP8.
- [ ] **Step 2**: Test a 7B/8B AWQ model on our single Tesla T4.
- [ ] **Step 3**: Benchmark Chunked Prefill on 1 GPU under concurrent load.
- [ ] **Step 4**: Design and execute a 2-GPU Prefill-Decode disaggregation proof-of-concept.
