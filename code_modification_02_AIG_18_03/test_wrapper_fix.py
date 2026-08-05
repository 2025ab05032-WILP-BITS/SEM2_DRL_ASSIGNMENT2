#!/usr/bin/env python3
"""Test the wrapper fix by running verification with 150 episodes."""

from RL_DQN_DDQN_Analysis import verify_wrapper_correctness

print("Running wrapper verification with 150 episodes...\n")
verify_wrapper_correctness(num_episodes=150)
