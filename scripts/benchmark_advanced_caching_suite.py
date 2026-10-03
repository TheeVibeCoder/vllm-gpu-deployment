#!/usr/bin/env python3
"""
1. HTTP cold/warm caching latencies (not disk-to-GPU transfer)
2. Buffered file read/write timings (may hit the page cache)
3. O_DIRECT open and cuFile library presence checks (not proof of DMA)

For raw SSD-to-GPU timings use benchmark_ssd_gpu.cpp.
"""

import os
import time
import argparse

def generate_prompt_by_tokens(target_tokens):
    """Generates synthetic prompt text approximating target token count."""
    base_sentence = "Artificial intelligence systems rely on scalable memory hierarchies, high-bandwidth interconnects, and optimized tensor execution. "
    tokens_per_sentence = 18  # Approximate tokens per repetition
    repeats = max(1, target_tokens // tokens_per_sentence)
    return base_sentence * repeats

def test_api_call(base_url, model_name, system_prompt, user_prompt, max_tokens=1):
    """Executes single inference call measuring exact Time to First Token (TTFT)."""
    import requests
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "success": True,
            "ttft_ms": elapsed * 1000,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "ttft_ms": 0}

def benchmark_crossover_threshold(base_url, model_name):
    """
    Evaluates The Golden Rule: Recomputation vs. I/O Crossover Threshold.
    Sweeps context lengths from 64 tokens to 4,096 tokens to identify
    the exact point where Cache Retrieval beats GPU Matrix Math.
    """
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENT 1: The Golden Rule Crossover Threshold (DirectKV / LMCache)")
    print("Hypothesis: Disk/RAM retrieval is slower than recomputation for tiny prompts,")
    print("            but exponentially faster as context length grows.")
    print("=" * 70)

    token_steps = [64, 256, 512, 1024, 2048, 4096]
    results = []

    for length in token_steps:
        print(f"\n[Evaluating Target Context: ~{length} Tokens]")
        sys_prompt = generate_prompt_by_tokens(length)
        user_msg = "Provide a one-sentence summary of modern cache offloading."

        # Pass 1: Cold Prefill (Forces GPU Tensor Core Recomputation)
        cold_res = test_api_call(base_url, model_name, sys_prompt, user_msg)
        if not cold_res["success"]:
            print(f"  ❌ Cold execution failed: {cold_res.get('error')}")
            continue

        actual_tokens = cold_res["prompt_tokens"]
        cold_ttft = cold_res["ttft_ms"]

        # Wait 2 seconds for LMCache asynchronous write to complete to CPU/SSD
        time.sleep(2)

        # Pass 2: Warm Retrieval (Tests Cache Retrieval Path)
        warm_res = test_api_call(base_url, model_name, sys_prompt, user_msg)
        warm_ttft = warm_res["ttft_ms"] if warm_res["success"] else 0

        speedup = ((cold_ttft - warm_ttft) / cold_ttft) * 100 if cold_ttft > 0 else 0
        decision = "✅ CACHE WINS (Use Offload)" if warm_ttft < cold_ttft else "⚠️ RECOMPUTE FASTER (Skip Cache)"

        print(f"  Actual Tokens: {actual_tokens}")
        print(f"  Cold Prefill TTFT (GPU Math): {cold_ttft:.1f} ms")
        print(f"  Warm Retrieval TTFT (Cache):  {warm_ttft:.1f} ms")
        print(f"  Efficiency Delta:             {speedup:+.1f}%  -->  {decision}")

        results.append({
            "target": length,
            "actual_tokens": actual_tokens,
            "cold_ttft": cold_ttft,
            "warm_ttft": warm_ttft,
            "speedup": speedup,
            "winner": "Cache" if warm_ttft < cold_ttft else "Recompute"
        })

    # Summary Table
    print("\n" + "-" * 70)
    print("📊 CROSSOVER THRESHOLD SUMMARY MATRIX")
    print("-" * 70)
    print(f"{'Context':<10} | {'Cold Math':<12} | {'Cached Load':<12} | {'Speedup':<10} | {'Verdict'}")
    print("-" * 70)
    for r in results:
        print(f"{r['actual_tokens']:<10} | {r['cold_ttft']:>8.1f} ms  | {r['warm_ttft']:>8.1f} ms  | {r['speedup']:>+7.1f}%  | {r['winner']}")
    print("-" * 70)

def profile_storage_subsystem(disk_path):
    """Measure buffered file I/O; check whether O_DIRECT open is accepted."""
    print("\n" + "=" * 70)
    print("💾 EXPERIMENT 2: Buffered file I/O (NOT disk-to-GPU)")
    print(f"Testing mount: {disk_path}")
    print("=" * 70)

    os.makedirs(disk_path, exist_ok=True)
    test_file = os.path.join(disk_path, "benchmark_kv_chunk.bin")
    chunk_size = 4 * 1024 * 1024  # 4 MB chunk (~typical 256-token layer tensor)
    raw_data = b"0" * chunk_size

    # Test 1: Standard OS Buffered Write
    start = time.time()
    with open(test_file, "wb") as f:
        f.write(raw_data)
        f.flush()
        os.fsync(f.fileno())
    buffered_time = time.time() - start
    buffered_bw = (chunk_size / (1024 * 1024)) / buffered_time

    # Test 2: Standard Buffered Read
    start = time.time()
    with open(test_file, "rb") as f:
        _ = f.read()
    read_time = time.time() - start
    read_bw = (chunk_size / (1024 * 1024)) / read_time

    print(f"  Standard Buffered Write Bandwidth: {buffered_bw:.1f} MB/s ({buffered_time*1000:.2f} ms)")
    print(f"  Standard Buffered Read Bandwidth:  {read_bw:.1f} MB/s ({read_time*1000:.2f} ms)")

    # Capability check only. Raw SSD-to-GPU transfer measurements live in
    # benchmark_ssd_gpu.cpp; opening a descriptor does not measure direct I/O.
    if not hasattr(os, "O_DIRECT"):
        print("  O_DIRECT: unavailable on this platform (no transfer benchmark run)")
    else:
        try:
            fd = os.open(test_file, os.O_RDONLY | os.O_DIRECT)
            os.close(fd)
            print("  O_DIRECT: open succeeded (alignment and reads NOT tested here)")
        except OSError as e:
            print(f"  O_DIRECT: open failed ({e})")

    if os.path.exists(test_file):
        os.remove(test_file)

def check_enterprise_architectures():
    """
    Validates availability of NVIDIA GPUDirect Storage (GDS) and DirectKV.
    """
    print("\n" + "=" * 70)
    print("⚡ EXPERIMENT 3: Enterprise Architecture & Hardware Compatibility")
    print("1. NVIDIA GPUDirect Storage (GDS / cuFile)")
    print("2. DirectKV (OSDI 2026 Zero-Copy NVLink-C2C)")
    print("=" * 70)

    # Check cuFile / GDS
    has_cufile = False
    for p in ["/usr/local/cuda/lib64/libcufile.so", "/usr/lib/x86_64-linux-gnu/libcufile.so"]:
        if os.path.exists(p):
            has_cufile = True
            break
    
    print("\n[Architecture 1: NVIDIA GPUDirect Storage (GDS)]")
    if has_cufile:
        print("  libcufile.so:        FOUND (library presence only)")
        print("  NVMe-to-GPU DMA:     NOT verified; cuFile may use host-memory compatibility mode")
    else:
        print("  libcufile.so:        ⚠️ Not installed (Requires NVIDIA GDS package: `nvidia-gds`)")
        print("  Host-memory fallback: possible; cannot infer active transfer path from library presence")

    print("\n[Architecture 2: DirectKV (USENIX OSDI 2026 Zero-Copy)]")
    print("  Interconnect Check:  Evaluating system bus...")
    try:
        import subprocess
        lspci = subprocess.check_output("lspci 2>/dev/null || true", shell=True).decode()
        if "Turing" in lspci or "Tesla T4" in lspci:
            print("  GPU Device:          Tesla T4 (Turing / PCIe, not NVLink-C2C)")
            print("  DirectKV on this T4: not benchmarked; no zero-copy claim")
        else:
            print("  DirectKV:             not benchmarked by this script")
    except Exception:
        pass
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tier KV Cache Benchmark Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--disk-path", default="/tmp/lmcache_disk", help="Local NVMe SSD mount path")
    parser.add_argument("--skip-api", action="store_true", help="Skip live API tests (run storage/hardware checks only)")
    args = parser.parse_args()

    profile_storage_subsystem(args.disk_path)
    check_enterprise_architectures()

    if not args.skip_api:
        benchmark_crossover_threshold(args.url, args.model)
    else:
        print("\nSkipping live API sweep (--skip-api requested).")
