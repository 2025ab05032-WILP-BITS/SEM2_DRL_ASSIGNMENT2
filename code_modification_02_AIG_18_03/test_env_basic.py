#!/usr/bin/env python3
"""Test basic environment creation."""
import sys
import os

print("Starting test...", flush=True)
sys.stdout.flush()

try:
    print("Importing gymnasium...", flush=True)
    import gymnasium as gym
    print("✓ Gymnasium imported", flush=True)
    
    print("Creating LunarLander-v3 environment...", flush=True)
    env = gym.make('LunarLander-v3', render_mode=None)
    print("✓ LunarLander-v3 created", flush=True)
    
    print("Running a quick test step...", flush=True)
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful. Obs shape: {obs.shape}", flush=True)
    
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"✓ Step successful. Reward: {reward}", flush=True)
    
    env.close()
    print("✓ Environment closed", flush=True)
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Test complete!", flush=True)
