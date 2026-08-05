#!/usr/bin/env python3
"""Minimal test of the wrapper fix - just run verification."""
import sys
import os
import gymnasium as gym
import numpy as np
from RL_DQN_DDQN_Analysis import StochasticFailureLunarLanderWrapper

print("Creating environment...", flush=True)
sys.stdout.flush()

# Create base environment
base_env = gym.make('LunarLander-v3', render_mode=None)

# Seed for reproducibility
seed = 42
np.random.seed(seed)
rng = np.random.Generator(np.random.PCG64(seed))

# Create wrapper with debug mode
print("Creating wrapper with debug_mode=True...", flush=True)
sys.stdout.flush()

wrapped_env = StochasticFailureLunarLanderWrapper(
    base_env,
    debug_mode=True,
    rng=rng
)

print("Running 50 random episodes...", flush=True)
sys.stdout.flush()

# Run episodes with random policy
num_episodes = 50
for ep in range(num_episodes):
    obs, info = wrapped_env.reset(seed=seed + ep)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
    if (ep + 1) % 10 == 0:
        print(f"  Completed {ep + 1}/{num_episodes} episodes", flush=True)
        sys.stdout.flush()

print("\nRetrieving debug statistics...", flush=True)
sys.stdout.flush()

debug_stats = wrapped_env.get_debug_stats()

print("\n" + "=" * 80, flush=True)
print("VERIFICATION RESULTS", flush=True)
print("=" * 80, flush=True)

thruster_attempts = debug_stats['thruster_attempt_count']
thruster_failures = debug_stats['thruster_failure_count']
failure_rate = debug_stats['computed_failure_rate']

print(f"\nTotal thruster actions attempted: {thruster_attempts}", flush=True)
print(f"Total thruster actions that FAILED: {thruster_failures}", flush=True)
print(f"Observed failure rate: {failure_rate:.3f} ({failure_rate*100:.1f}%)", flush=True)
print(f"Expected failure rate: 0.150 (15.0%)", flush=True)

if 0.10 <= failure_rate <= 0.20:
    print(f"Status: ✓ PASS (within expected 10-20% range)", flush=True)
else:
    print(f"Status: ✗ FAIL (outside expected 10-20% range)", flush=True)

print("\n✓ Test complete!", flush=True)
sys.stdout.flush()
