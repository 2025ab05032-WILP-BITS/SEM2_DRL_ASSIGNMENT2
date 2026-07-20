"""
Quick test script to validate wrapper without Box2D dependency issues.
This script tests the core wrapper logic by mocking the environment.
"""

import sys
import numpy as np
from gymnasium import Env, spaces

# Add the current directory to path
sys.path.insert(0, '.')

# Import wrapper components
from environment.lunar_lander_wrapper import (
    StochasticFailureLunarLanderWrapper,
    OBS_VEL_X,
    OBS_VEL_Y,
    OBS_ANGLE,
    OBS_LEG_LEFT,
    OBS_LEG_RIGHT,
    FUEL_PENALTY,
    SAFE_LANDING_BONUS,
)

# Create a mock environment for testing
class MockLunarLander(Env):
    """Mock LunarLander environment for testing without Box2D dependency."""
    
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        
    def step(self, action):
        # Return a dummy observation and reward
        obs = np.zeros(8, dtype=np.float32)
        obs[OBS_VEL_X] = 0.05
        obs[OBS_VEL_Y] = 0.05
        obs[OBS_ANGLE] = 0.05
        obs[OBS_LEG_LEFT] = 1.0
        obs[OBS_LEG_RIGHT] = 1.0
        return obs, 100.0, True, False, {}
    
    def reset(self, seed=None, options=None):
        obs = np.zeros(8, dtype=np.float32)
        return obs, {}
    
    def close(self):
        pass


# Test wrapper with mock environment
print("=" * 80)
print("WRAPPER IMPLEMENTATION TEST (Mock Environment)")
print("=" * 80)

try:
    # Create mock environment
    mock_env = MockLunarLander()
    
    # For testing purposes, we'll directly instantiate wrapper without strict type checking
    # by using a minimal wrapper object that has the required methods
    wrapper = object.__new__(StochasticFailureLunarLanderWrapper)
    wrapper.env = mock_env
    wrapper.observation_space = mock_env.observation_space
    wrapper.action_space = mock_env.action_space
    
    print("✓ Wrapper successfully initialized for testing\n")
    
    # Test observation space
    assert wrapper.observation_space.shape == (8,), "Observation space should be 8-dimensional"
    print("✓ Observation space: 8-dimensional (correct)")
    
    # Test action space
    assert wrapper.action_space.n == 4, "Action space should have 4 discrete actions"
    print("✓ Action space: 4 discrete actions (correct)\n")
    
    # Test fuel penalty calculation
    print("Testing fuel penalty calculation:")
    penalty_action_0 = wrapper._calculate_fuel_penalty(0)
    assert penalty_action_0 == 0.0, f"Action 0 penalty should be 0.0, got {penalty_action_0}"
    print(f"  - Action 0 (Do Nothing): {penalty_action_0} ✓")
    
    for action in [1, 2, 3]:
        penalty = wrapper._calculate_fuel_penalty(action)
        assert penalty == FUEL_PENALTY, f"Action {action} penalty should be {FUEL_PENALTY}, got {penalty}"
        print(f"  - Action {action} (Thruster): {penalty} ✓")
    
    # Test safe landing bonus
    print("\nTesting safe landing bonus calculation:")
    
    # Perfect landing
    obs_perfect = np.array([0, 0, 0.05, 0.05, 0.05, 0, 1.0, 1.0], dtype=np.float32)
    bonus_perfect = wrapper._calculate_safe_landing_bonus(obs_perfect, terminated=True, truncated=False)
    assert bonus_perfect == SAFE_LANDING_BONUS, f"Perfect landing should give {SAFE_LANDING_BONUS} bonus"
    print(f"  - Perfect landing (all criteria met): {bonus_perfect} ✓")
    
    # Not terminated
    obs_not_term = obs_perfect.copy()
    bonus_not_term = wrapper._calculate_safe_landing_bonus(obs_not_term, terminated=False, truncated=False)
    assert bonus_not_term == 0.0, "Should not give bonus if not terminated"
    print(f"  - Not terminated: {bonus_not_term} ✓")
    
    # Truncated
    bonus_truncated = wrapper._calculate_safe_landing_bonus(obs_not_term, terminated=True, truncated=True)
    assert bonus_truncated == 0.0, "Should not give bonus if truncated"
    print(f"  - Truncated: {bonus_truncated} ✓")
    
    # Left leg not touching
    obs_no_left_leg = obs_perfect.copy()
    obs_no_left_leg[OBS_LEG_LEFT] = 0.0
    bonus_no_left = wrapper._calculate_safe_landing_bonus(obs_no_left_leg, terminated=True, truncated=False)
    assert bonus_no_left == 0.0, "Should not give bonus if left leg not touching"
    print(f"  - Left leg not touching: {bonus_no_left} ✓")
    
    # Velocity too high
    obs_high_vel = obs_perfect.copy()
    obs_high_vel[OBS_VEL_X] = 0.15
    bonus_high_vel = wrapper._calculate_safe_landing_bonus(obs_high_vel, terminated=True, truncated=False)
    assert bonus_high_vel == 0.0, "Should not give bonus if velocity too high"
    print(f"  - Velocity too high (0.15 > 0.10 threshold): {bonus_high_vel} ✓")
    
    # Test stochastic failure rate
    print("\nTesting stochastic failure mechanism:")
    np.random.seed(42)
    num_trials = 1000
    failures = 0
    
    for _ in range(num_trials):
        action = 1  # Test with action 1 (thruster)
        executed = wrapper._apply_stochastic_failure(action)
        if executed == 0:
            failures += 1
    
    failure_rate = failures / num_trials
    print(f"  - Stochastic failure rate: {failure_rate:.3f} (expected ~0.15)")
    assert 0.10 < failure_rate < 0.20, f"Failure rate {failure_rate} should be near 0.15"
    print(f"  ✓ Failure rate is within expected range\n")
    
    # Test step function with mock environment
    print("Testing step function:")
    wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(1)  # Thruster action
    
    # Expected reward: base (100.0) - fuel penalty (0.3) + bonus (50.0) = 149.7
    expected_reward = 100.0 - 0.3 + 50.0
    assert abs(reward - expected_reward) < 0.01, f"Expected reward ~{expected_reward}, got {reward}"
    print(f"  - Step executed successfully")
    print(f"  - Reward calculation: {reward:.2f} (base:100.0 - fuel:0.3 + bonus:50.0 = {expected_reward})")
    print(f"  ✓ Reward correctly modified\n")
    
    # Verify info dict has no leakage
    assert 'failure' not in info, "Info dict should not contain 'failure'"
    assert 'stochastic_failure' not in info, "Info dict should not contain 'stochastic_failure'"
    assert 'executed_action' not in info, "Info dict should not contain 'executed_action'"
    print("  ✓ No failure information leaked to info dict\n")
    
    print("=" * 80)
    print("✓ ALL TESTS PASSED - Wrapper implementation is correct!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
