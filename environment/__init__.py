"""
Environment module for Reinforcement Learning with stochastic actuator failures.

This module provides a custom Gymnasium wrapper around LunarLander-v3 that introduces
realistic stochastic engine failures and modified reward functions, along with
verification and testing utilities.

Key Components:
    - StochasticFailureLunarLanderWrapper: Custom wrapper applying engine failures
    - run_verification(): Verification script that statistically validates wrapper behavior
    - Comprehensive unit tests for wrapper correctness

Example:
    >>> from environment import StochasticFailureLunarLanderWrapper
    >>> import gymnasium as gym
    >>> base_env = gym.make("LunarLander-v3")
    >>> env = StochasticFailureLunarLanderWrapper(base_env)
    >>> obs, _ = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

from environment.lunar_lander_wrapper import StochasticFailureLunarLanderWrapper
from environment.verify_wrapper import run_verification

__all__ = [
    "StochasticFailureLunarLanderWrapper",
    "run_verification",
]

__version__ = "1.0.0"
__author__ = "Senior AI Researcher"
__description__ = (
    "Custom Gymnasium wrapper introducing stochastic engine failures to LunarLander-v3 "
    "for realistic reinforcement learning simulation"
)
