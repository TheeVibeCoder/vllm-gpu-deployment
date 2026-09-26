#!/usr/bin/env python3
"""
Benchmark script for evaluating vLLM + LMCache hybrid KV Cache offloading.
Tests Cold TTFT vs Warm Cache TTFT.
"""
import os
import time
import requests
import argparse

def run_benchmark(base_url, model_name):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    
    system_prompt = (
        "You are an expert AI assistant specialized in distributed systems, "
        "cloud architecture, and machine learning infrastructure. You provide "
        "detailed, accurate, and comprehensive answers about GPU computing, "
        "model serving, cache optimization, memory hierarchies, and performance engineering."
    )
    user_prompt = "Explain how KV cache offloading works in LLM serving, including PCIe, pinned memory, and tiered storage."

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "stream": False
    }

    def call_api(label):
        start = time.time()
        resp = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("usage", {})
        print(f"[{label}]")
        print(f"  Response Time: {elapsed*1000:.1f} ms")
        print(f"  Prompt Tokens: {tokens.get('prompt_tokens', 0)}")
        print(f"  Completion Tokens: {tokens.get('completion_tokens', 0)}")
        return elapsed

    print("=" * 60)
    print(f"Running KV Cache Benchmark on: {url}")
    print("=" * 60)

    t1 = call_api("Turn 1 (Cold Start)")
    time.sleep(3)
    t2 = call_api("Turn 2 (Warm Cache Hit)")
    time.sleep(1)
    t3 = call_api("Turn 3 (Hot Cache Hit)")

    print("=" * 60)
    print("Benchmark Summary:")
    print(f"  Cold TTFT: {t1*1000:.1f} ms")
    print(f"  Warm TTFT: {t2*1000:.1f} ms")
    print(f"  Speedup:   {((t1 - t2) / t1) * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    args = parser.parse_args()

    run_benchmark(args.url, args.model)
