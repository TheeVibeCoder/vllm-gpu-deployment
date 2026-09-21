#!/usr/bin/env python3
"""
Test client to verify any active vLLM OpenAI-compatible endpoint.
"""
import sys
import json
import urllib.request

if len(sys.argv) < 2:
    print("Usage: python test_client.py <OPENAI_BASE_URL>")
    print("Example: python test_client.py https://your-server.litng.ai/v1")
    sys.exit(1)

base_url = sys.argv[1].rstrip('/')
if not base_url.endswith('/chat/completions'):
    url = f"{base_url}/chat/completions"
else:
    url = base_url

print(f"Sending test chat request to: {url}")

payload = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a concise expert assistant."},
        {"role": "user", "content": "Say hello in 5 words."}
    ],
    "max_tokens": 50
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
        print("\nResponse from Model:")
        print(result["choices"][0]["message"]["content"])
        print("\nTokens Used:")
        print(f"Prompt: {result['usage']['prompt_tokens']}, Output: {result['usage']['completion_tokens']}")
except Exception as e:
    print(f"Connection Failed: {e}")
