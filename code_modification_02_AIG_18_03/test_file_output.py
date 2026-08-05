#!/usr/bin/env python3
"""Test wrapper fix - write output to file."""
import sys
import os
import gymnasium as gym
import numpy as np
from RL_DQN_DDQN_Analysis import StochasticFailureLunarLanderWrapper

output_file = "verification_results.txt"
with open(output_file, 'w') as f:
    f.write("Starting wrapper verification test...\n")
    f.flush()

try:
    # Create base environment
    with open(output_file, 'a') as f:
        f.write("Creating environment...\n")
        f.flush()

    base_env = gym.make('LunarLander-v3', render_mode=None)

    # Seed for reproducibility
    seed = 42
    np.random.seed(seed)
    rng = np.random.Generator(np.random.PCG64(seed))

    # Create wrapper with debug mode
    with open(output_file, 'a') as f:
        f.write("Creating wrapper with debug_mode=True...\n")
        f.flush()

    wrapped_env = StochasticFailureLunarLanderWrapper(
        base_env,
        debug_mode=True,
        rng=rng
    )

    with open(output_file, 'a') as f:
        f.write("Running 50 random episodes...\n")
        f.flush()

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
            with open(output_file, 'a') as f:
                f.write(f"  Completed {ep + 1}/{num_episodes} episodes\n")
                f.flush()

    with open(output_file, 'a') as f:
        f.write("\nRetrieving debug statistics...\n")
        f.flush()

    debug_stats = wrapped_env.get_debug_stats()

    with open(output_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("VERIFICATION RESULTS\n")
        f.write("=" * 80 + "\n")

        thruster_attempts = debug_stats['thruster_attempt_count']
        thruster_failures = debug_stats['thruster_failure_count']
        failure_rate = debug_stats['computed_failure_rate']

        f.write(f"\nTotal thruster actions attempted: {thruster_attempts}\n")
        f.write(f"Total thruster actions that FAILED: {thruster_failures}\n")
        f.write(f"Observed failure rate: {failure_rate:.3f} ({failure_rate*100:.1f}%)\n")
        f.write(f"Expected failure rate: 0.150 (15.0%)\n")

        if 0.10 <= failure_rate <= 0.20:
            f.write(f"Status: ✓ PASS (within expected 10-20% range)\n")
        else:
            f.write(f"Status: ✗ FAIL (outside expected 10-20% range)\n")

        f.write("\n✓ Test complete!\n")
        f.flush()

    # Also print to console
    print(f"✓ Test completed! Results written to {output_file}")
    with open(output_file, 'r') as f:
        print(f.read())
        
except Exception as e:
    with open(output_file, 'a') as f:
        f.write(f"\n✗ Error: {type(e).__name__}: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.flush()
    raise
