#!/usr/bin/env python3
"""
Simplified test runner - verify core wrapper mechanics
"""

import sys
import os

# Test imports first
print("\n" + "="*70)
print("DEPENDENCY CHECK")
print("="*70)

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

try:
    import gymnasium as gym
    print(f"✓ gymnasium {gym.__version__}")
except ImportError as e:
    print(f"✗ gymnasium: {e}")
    sys.exit(1)

try:
    from RL_DQN_DDQN_Analysis import StochasticFailureLunarLanderWrapper, FUEL_PENALTY, SAFE_LANDING_BONUS
    print(f"✓ StochasticFailureLunarLanderWrapper loaded")
    print(f"  - FUEL_PENALTY = {FUEL_PENALTY}")
    print(f"  - SAFE_LANDING_BONUS = {SAFE_LANDING_BONUS}")
except ImportError as e:
    print(f"✗ Failed to import wrapper: {e}")
    sys.exit(1)

# Quick verification tests
print("\n" + "="*70)
print("QUICK VERIFICATION TESTS")
print("="*70)

# Test 1: Check fuel penalty calculation via direct helper call
print("\n[Test 1] Fuel Penalty Calculation")
print("-" * 70)

try:
    env = StochasticFailureLunarLanderWrapper(gym.make("LunarLander-v3"))
    
    # Test attempted action = 0 → no fuel penalty
    penalty_no_op = env._calculate_fuel_penalty(0)
    assert penalty_no_op == 0.0, f"Expected 0.0 for no-op, got {penalty_no_op}"
    print(f"✓ No-op action (0) → penalty = {penalty_no_op}")
    
    # Test attempted action = 1 → fuel penalty
    penalty_thruster = env._calculate_fuel_penalty(1)
    assert penalty_thruster == FUEL_PENALTY, f"Expected {FUEL_PENALTY}, got {penalty_thruster}"
    print(f"✓ Thruster action (1) → penalty = {penalty_thruster}")
    
    # Test all thruster actions
    for action in [1, 2, 3]:
        penalty = env._calculate_fuel_penalty(action)
        assert penalty == FUEL_PENALTY, f"Action {action}: expected {FUEL_PENALTY}, got {penalty}"
    print(f"✓ All thruster actions (1,2,3) → penalty = {FUEL_PENALTY}")
    
    env.close()
    print("\n✓ Fuel Penalty Test PASSED")
    
except Exception as e:
    print(f"\n✗ Fuel Penalty Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check landing bonus with all criteria met
print("\n[Test 2] Landing Bonus - All Criteria Met")
print("-" * 70)

try:
    env = StochasticFailureLunarLanderWrapper(gym.make("LunarLander-v3"))
    
    # Synthetic observation with all 7 criteria satisfied
    obs = np.array([
        0.0,      # x position (within range)
        0.0,      # y position (within range)
        0.087,    # h_vel: |0.087| < 0.10 ✓
        0.045,    # v_vel: |0.045| < 0.10 ✓
        0.032,    # angle: |0.032| < 0.10 ✓
        0.0,      # angular velocity
        1.0,      # left_leg (landed) ✓
        1.0       # right_leg (landed) ✓
    ])
    
    # Test with terminated=True, truncated=False
    bonus = env._calculate_safe_landing_bonus(obs, terminated=True, truncated=False)
    assert bonus == SAFE_LANDING_BONUS, f"Expected {SAFE_LANDING_BONUS}, got {bonus}"
    print(f"✓ All 7 criteria met → bonus = {bonus}")
    
    # Detailed criteria check
    print("\n  Criteria verification:")
    print(f"    1. terminated = True ✓")
    print(f"    2. truncated = False ✓")
    print(f"    3. left_leg = {obs[6]} ✓")
    print(f"    4. right_leg = {obs[7]} ✓")
    print(f"    5. |h_vel| = {abs(obs[2]):.3f} < 0.10 ✓")
    print(f"    6. |v_vel| = {abs(obs[3]):.3f} < 0.10 ✓")
    print(f"    7. |angle| = {abs(obs[4]):.3f} < 0.10 ✓")
    
    env.close()
    print("\n✓ Landing Bonus (All Criteria Met) Test PASSED")
    
except Exception as e:
    print(f"\n✗ Landing Bonus (All Criteria Met) Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check landing bonus with individual criterion violations
print("\n[Test 3] Landing Bonus - Individual Criterion Violations")
print("-" * 70)

try:
    env = StochasticFailureLunarLanderWrapper(gym.make("LunarLander-v3"))
    
    violations = [
        ("terminated=False", {"obs_override": None, "terminated": False, "truncated": False}),
        ("truncated=True", {"obs_override": None, "terminated": True, "truncated": True}),
        ("|h_vel| > 0.10", {"obs_override": np.array([0, 0, 0.15, 0.045, 0.032, 0, 1, 1]), "terminated": True, "truncated": False}),
        ("|v_vel| > 0.10", {"obs_override": np.array([0, 0, 0.087, 0.11, 0.032, 0, 1, 1]), "terminated": True, "truncated": False}),
        ("|angle| > 0.10", {"obs_override": np.array([0, 0, 0.087, 0.045, 0.15, 0, 1, 1]), "terminated": True, "truncated": False}),
        ("left_leg=0", {"obs_override": np.array([0, 0, 0.087, 0.045, 0.032, 0, 0, 1]), "terminated": True, "truncated": False}),
        ("right_leg=0", {"obs_override": np.array([0, 0, 0.087, 0.045, 0.032, 0, 1, 0]), "terminated": True, "truncated": False}),
    ]
    
    # Base passing observation
    base_obs = np.array([0, 0, 0.087, 0.045, 0.032, 0, 1, 1])
    
    for violation_name, test_params in violations:
        obs = test_params.get("obs_override", base_obs)
        terminated = test_params.get("terminated", True)
        truncated = test_params.get("truncated", False)
        
        bonus = env._calculate_safe_landing_bonus(obs, terminated=terminated, truncated=truncated)
        assert bonus == 0.0, f"Violation '{violation_name}': expected 0.0, got {bonus}"
        print(f"✓ {violation_name:20} → bonus = 0.0")
    
    env.close()
    print("\n✓ Landing Bonus (Criterion Violations) Test PASSED")
    
except Exception as e:
    print(f"\n✗ Landing Bonus (Criterion Violations) Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ All verification tests PASSED")
print("")
print("Wrapper Mechanics Verified:")
print("  1. Fuel penalty: -0.3 for each attempted thruster (actions 1,2,3)")
print("  2. Landing bonus: +50.0 when all 7 criteria simultaneously met")
print("  3. AND logic: Any single criterion violation → bonus = 0")
print("")
print("="*70)
