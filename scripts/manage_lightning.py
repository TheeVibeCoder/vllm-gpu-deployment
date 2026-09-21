#!/usr/bin/env python3
"""
Lightning AI GPU Studio Manager
Starts, stops, and queries the remote GPU instance without hardcoding any keys.
Set your credentials via environment variables or a .env file.
"""
import os
import sys

# Verify credentials exist in environment
api_key = os.environ.get('LIGHTNING_API_KEY')
user_id = os.environ.get('LIGHTNING_USER_ID')
username = os.environ.get('LIGHTNING_USERNAME')
teamspace_name = os.environ.get('LIGHTNING_TEAMSPACE', 'general')
studio_name = os.environ.get('LIGHTNING_STUDIO', 'my-gpu-studio')

if not api_key:
    print("ERROR: LIGHTNING_API_KEY environment variable is not set.")
    print("Please run: export LIGHTNING_API_KEY='your_key'")
    sys.exit(1)

try:
    from lightning_sdk import Studio, Machine, Teamspace
except ImportError:
    print("ERROR: lightning-sdk is not installed. Run: pip install lightning-sdk")
    sys.exit(1)

teamspace = Teamspace(teamspace_name)
studio = Studio(studio_name, teamspace=teamspace)

action = sys.argv[1].lower() if len(sys.argv) > 1 else 'status'

if action == 'start':
    print(f"Starting Studio '{studio_name}' on NVIDIA T4 GPU...")
    studio.start(machine=Machine.T4)
    endpoints = studio.add_ports(8000)
    print(f"Studio Status: {studio.status}")
    for ep in endpoints:
        print(f"Public OpenAI Base URL: {ep.urls[0]}/v1")
    
    # Launch vLLM in background with 32K context and Hermes tool parsing
    cmd = (
        "nohup python -m vllm.entrypoints.openai.api_server "
        "--model Qwen/Qwen2.5-0.5B-Instruct "
        "--host 0.0.0.0 --port 8000 "
        "--gpu-memory-utilization 0.75 "
        "--max-model-len 32768 "
        "--enable-auto-tool-choice "
        "--tool-call-parser hermes > /teamspace/studios/this_studio/vllm.log 2>&1 &"
    )
    studio.run(cmd)
    print("vLLM 32K Server initiated successfully!")

elif action == 'stop':
    print(f"Stopping Studio '{studio_name}' to preserve credits...")
    studio.stop()
    print(f"Studio Status: {studio.status}")

elif action == 'status':
    print(f"Studio Name: {studio.name}")
    print(f"Studio Status: {studio.status}")
    print(f"Current Machine: {studio.machine}")
    ports = studio.list_ports()
    if ports:
        for ep in ports:
            print(f"Active Port {ep.ports}: {ep.urls[0]}/v1")
else:
    print("Usage: python manage_lightning.py [start|stop|status]")
