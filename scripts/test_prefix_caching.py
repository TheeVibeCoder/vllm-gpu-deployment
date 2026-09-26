#!/usr/bin/env python3
"""
vLLM Prefix Caching Benchmark
Demonstrates TTFT reduction by reusing KV cache blocks for identical prompt prefixes.
"""
import os
import sys
import time
import json
import requests

def benchmark_turn(endpoint_url, prefix, question, turn_label):
    print(f"\n--- Running {turn_label} ---")
    full_prompt = f"{prefix}\n\nUser Question: {question}\nPlease provide a concise answer."
    
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": True
    }
    
    start_time = time.perf_counter()
    ttft = None
    output_tokens = []
    
    try:
        response = requests.post(
            f"{endpoint_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_content = line_str[6:].strip()
                if data_content == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_content)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token_text = delta.get("content", "")
                    if token_text:
                        if ttft is None:
                            ttft = time.perf_counter() - start_time
                        output_tokens.append(token_text)
                except json.JSONDecodeError:
                    pass
                    
        total_time = time.perf_counter() - start_time
        num_tokens = len(output_tokens)
        gen_time = (total_time - ttft) if ttft else total_time
        tok_per_sec = (num_tokens / gen_time) if gen_time > 0 else 0
        answer_text = "".join(output_tokens).strip()
        
        print(f"Response: {answer_text[:120]}...")
        print(f"› TTFT (Time to First Token): {ttft*1000:.2f} ms")
        print(f"› Total Time: {total_time*1000:.2f} ms")
        print(f"› Generation Speed: {tok_per_sec:.2f} tok/s ({num_tokens} tokens)")
        
        return {
            "ttft_ms": ttft * 1000 if ttft else 0,
            "total_ms": total_time * 1000,
            "tokens": num_tokens,
            "tok_per_sec": tok_per_sec
        }
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        endpoint = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    else:
        endpoint = sys.argv[1].rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
            
    print(f"Connecting to vLLM endpoint: {endpoint}")
    
    # Create a realistic long prefix (~2,000 tokens)
    base_doc = (
        "Project Architecture Document: KV Cache Management System.\n"
        "In modern transformer architectures, attention computation is a fundamental performance driver. "
        "Each layer computes Query, Key, and Value representations across hidden dimensions. "
        "During prefill, the model computes cross-attention across all positions in parallel. "
        "During decode, the model relies on the KV Cache stored in GPU VRAM to maintain conversational history. "
        "PagedAttention eliminates memory fragmentation by allocating 16-token non-contiguous virtual memory blocks. "
    )
    long_prefix = base_doc * 35 # ~2,100 tokens of structured context
    
    print(f"Prepared Shared Context Prefix: ~{len(long_prefix.split())} words (~2,200 tokens)")
    
    # Turn 1: Cold start / Cache miss
    turn1_res = benchmark_turn(
        endpoint,
        long_prefix,
        "What does PagedAttention eliminate in the memory system?",
        "Turn 1 (Cold Start - Prefill Math Required)"
    )
    
    # Brief pause between turns
    time.sleep(1)
    
    # Turn 2: Exact same prefix + new question (Cache Hit!)
    turn2_res = benchmark_turn(
        endpoint,
        long_prefix,
        "What representations are computed across hidden dimensions?",
        "Turn 2 (Prefix Cached - Expecting Cache Hit)"
    )
    
    if turn1_res and turn2_res:
        t1_ttft = turn1_res["ttft_ms"]
        t2_ttft = turn2_res["ttft_ms"]
        speedup = ((t1_ttft - t2_ttft) / t1_ttft) * 100 if t1_ttft > 0 else 0
        
        print("\n" + "="*55)
        print("          PREFIX CACHING BENCHMARK RESULTS")
        print("="*55)
        print(f"Turn 1 TTFT (Cache Miss):   {t1_ttft:8.2f} ms")
        print(f"Turn 2 TTFT (Cache Hit!):   {t2_ttft:8.2f} ms")
        print(f"TTFT Latency Reduction:     {speedup:8.1f} %")
        print("="*55)
        if speedup > 20:
            print("🚀 SUCCESS: Prefix Caching successfully bypassed prefill math!")
        else:
            print("Note: Fast initial responses or warm caches observed.")
        print("="*55)

if __name__ == "__main__":
    main()
