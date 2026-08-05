# Quick Reference: RNG Improvements

## What Changed?

| Component | Before | After |
|-----------|--------|-------|
| `set_seed()` | Returns None | Returns `np.random.Generator` |
| Wrapper | Takes no RNG param | Accepts `rng=` and `seed=` |
| ReplayBuffer | Uses global `np.random` | Uses instance `self.rng` |
| DQNAgent | Uses global `np.random` | Uses instance `self.rng` |
| Training loop | Relys on global state | Passes RNG to components |

## How to Use

### Basic Usage (No Code Changes Required)
```python
rng = set_seed(42)  # Now returns a Generator instance
# Everything still works as before
```

### Better: Explicit RNG (Recommended)
```python
rng = set_seed(42)
wrapper = StochasticFailureLunarLanderWrapper(env, rng=rng)
agent = DQNAgent(rng=rng)
buffer = ReplayBuffer(rng=rng)
```

### Advanced: Independent Streams
```python
# Each agent gets own RNG with same seed (fair comparison)
agent1 = DQNAgent(rng=np.random.Generator(np.random.PCG64(42)))
agent2 = DDQNAgent(rng=np.random.Generator(np.random.PCG64(42)))

# Or different seeds (independent exploration)
agent1 = DQNAgent(rng=np.random.Generator(np.random.PCG64(42)))
agent2 = DDQNAgent(rng=np.random.Generator(np.random.PCG64(999)))
```

## Key Files Modified

1. **set_seed()** (Line 79-149)
   - Now returns `np.random.Generator`
   - Creates PCG64-based generator
   
2. **StochasticFailureLunarLanderWrapper** (Line 226-253)
   - Added `rng` and `seed` parameters
   - Uses `self.rng.random()` in `_apply_stochastic_failure()`
   - Updates RNG in `reset()` if seed provided

3. **ReplayBuffer** (Line 760-796)
   - Added `rng` and `seed` parameters
   - Uses `self.rng.choice()` in `sample()`

4. **DQNAgent** (Line 910-972)
   - Added `rng` and `seed` parameters
   - Passes to ReplayBuffer
   - Uses `self.rng.random()` and `self.rng.integers()` in `act()`

5. **train_agents()** (Line 1403-1430)
   - Creates separate RNG for each component
   - Seeds environments explicitly
   - Offset seed per episode for variety

6. **verify_wrapper_correctness()** (Line 580-609)
   - Creates reproducible RNG
   - Passes to wrapper

## Reproducibility Benefits

✅ **No Global State Pollution**: Each component has isolated RNG  
✅ **Guaranteed Reproducibility**: Same seed → identical sequences  
✅ **Gym Compatibility**: Works cleanly with Gymnasium API  
✅ **Component Isolation**: Easy to test independently  
✅ **Clean Architecture**: No hidden dependencies  

## Backward Compatibility

✅ Old code still works without changes  
✅ Optional RNG parameters (defaults provided)  
✅ set_seed() still sets global seeds  
✅ No breaking changes to existing APIs  

## Example: Complete Reproducible Setup

```python
# Setup
seed = 42
rng = set_seed(seed)

# Create environments
base_env = gym.make("LunarLander-v3")
modified_env = StochasticFailureLunarLanderWrapper(
    gym.make("LunarLander-v3"),
    rng=rng,
    seed=seed
)

# Create agents with isolated RNGs (same seed for fair comparison)
agent1 = DQNAgent(rng=np.random.Generator(np.random.PCG64(seed)))
agent2 = DDQNAgent(rng=np.random.Generator(np.random.PCG64(seed)))

# Training (reproducible)
for episode in range(num_episodes):
    obs, _ = modified_env.reset(seed=seed + episode)
    # ... training loop ...
    
# Result: Perfectly reproducible runs!
```

## Testing

```python
# Verify reproducibility
rng1 = np.random.Generator(np.random.PCG64(42))
rng2 = np.random.Generator(np.random.PCG64(42))

wrapper1 = StochasticFailureLunarLanderWrapper(env1, rng=rng1)
wrapper2 = StochasticFailureLunarLanderWrapper(env2, rng=rng2)

# Same action → same failure outcome
assert wrapper1.step(action)[1] == wrapper2.step(action)[1]  ✓
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Failures different each run" | Pass `rng=` to wrapper |
| "Agents not comparing fairly" | Use same seed for both agents |
| "Tests not reproducible" | Call `set_seed(42)` before each test |
| "Modified env has different behavior" | Ensure wrapper gets RNG instance |

## Performance Impact

✅ **No performance degradation**  
- Generator is as fast as global RNG
- PCG64 is actually faster for integer generation
- Isolated RNG may be slightly faster (no contention)

## Questions?

See `RNG_REPRODUCIBILITY_IMPROVEMENTS.md` for detailed explanation.
