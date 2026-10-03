import subprocess
import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"

app = modal.App("vllm-7b-openai-server")

# Official CUDA 12 development image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "transformers",
        "accelerate",
        "torch",
    )
    .env({
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    })
)

@app.function(
    image=image,
    gpu=["A10G", "L4", "T4"],  # Multi-GPU fallback: instant scheduling!
    timeout=3600,
    scaledown_window=300,  # Keep alive for 5 minutes after last request
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    print(f"🚀 Starting vLLM 8-bit quantized server: {' '.join(cmd)}")
    subprocess.Popen(cmd)
