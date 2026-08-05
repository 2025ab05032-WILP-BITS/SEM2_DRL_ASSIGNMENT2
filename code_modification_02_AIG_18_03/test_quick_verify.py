#!/usr/bin/env python3
"""Test the wrapper fix - extract only verification results."""

from RL_DQN_DDQN_Analysis import verify_wrapper_correctness

print("Running wrapper verification with 150 episodes (debug output enabled)...\n")
verify_wrapper_correctness(num_episodes=150)

# Only print verification section (this will print after the 150 episodes complete)
