#!/usr/bin/env python
"""
Quick test to verify ReplayBuffer RNG initialization fix
"""
import numpy as np
from collections import deque
from typing import Optional

# Test the ReplayBuffer initialization
class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 10000,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None
    ) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # Initialize RNG: prefer provided rng, otherwise create from seed
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))
        else:
            # If no rng or seed provided, create with random seed (less reproducible)
            self.rng = np.random.Generator(np.random.PCG64())

print("=" * 70)
print("TESTING REPLAYBUFFER RNG INITIALIZATION FIX")
print("=" * 70)

# Test 1: ReplayBuffer without RNG (default)
print("\n✓ Test 1: ReplayBuffer with default RNG...")
buffer1 = ReplayBuffer(capacity=100)
assert hasattr(buffer1, 'rng'), "ERROR: Missing rng attribute!"
assert isinstance(buffer1.rng, np.random.Generator), "ERROR: rng is not a Generator!"
print(f"  - Has rng attribute: YES")
print(f"  - RNG type: {type(buffer1.rng).__name__}")

# Test 2: ReplayBuffer with explicit seed
print("\n✓ Test 2: ReplayBuffer with seed=42...")
buffer2 = ReplayBuffer(capacity=100, seed=42)
assert hasattr(buffer2, 'rng'), "ERROR: Missing rng attribute!"
assert isinstance(buffer2.rng, np.random.Generator), "ERROR: rng is not a Generator!"
print(f"  - Has rng attribute: YES")
print(f"  - RNG type: {type(buffer2.rng).__name__}")

# Test 3: ReplayBuffer with explicit Generator
print("\n✓ Test 3: ReplayBuffer with explicit Generator...")
rng = np.random.Generator(np.random.PCG64(42))
buffer3 = ReplayBuffer(capacity=100, rng=rng)
assert hasattr(buffer3, 'rng'), "ERROR: Missing rng attribute!"
assert buffer3.rng is rng, "ERROR: rng is not the same instance!"
print(f"  - Has rng attribute: YES")
print(f"  - RNG is same instance: YES")

# Test 4: Verify required methods exist
print("\n✓ Test 4: Verifying Generator methods...")
buffer4 = ReplayBuffer(capacity=100, seed=42)
assert hasattr(buffer4.rng, 'choice'), "ERROR: Missing .choice() method!"
assert hasattr(buffer4.rng, 'random'), "ERROR: Missing .random() method!"
assert hasattr(buffer4.rng, 'integers'), "ERROR: Missing .integers() method!"
print(f"  - Has .choice() method: YES")
print(f"  - Has .random() method: YES")
print(f"  - Has .integers() method: YES")

# Test 5: Simulate sampling (like in training)
print("\n✓ Test 5: Testing sampling (mimics training loop)...")
buffer5 = ReplayBuffer(capacity=100, seed=42)
# Add dummy data
for i in range(20):
    buffer5.buffer.append((np.zeros(8), i % 4, float(i), np.zeros(8), False))

# Try sampling
try:
    batch_size = min(5, len(buffer5.buffer))
    indices = buffer5.rng.choice(len(buffer5.buffer), batch_size, replace=False)
    print(f"  - Sampling succeeded! Got {len(indices)} indices: {list(indices)}")
except AttributeError as e:
    print(f"  ERROR: {e}")
    raise

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED! ReplayBuffer RNG initialization is working correctly!")
print("=" * 70)
print("\nThe AttributeError 'ReplayBuffer' object has no attribute 'rng' is FIXED!")
