# LunarLander — Stochastic Actuator Failures (Assignment 2 — DRL)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-pytest-orange)]()

An academically focused implementation and verification suite that augments the classic LunarLander environment with realistic stochastic actuator (thruster) failures, fuel-penalty on attempted thruster commands, and a strict safe-landing bonus. This repository is the companion code for the DRL assignment "Lunar Lander with stochastic actuator failures".

Why this repo exists
- To provide a clear, testable wrapper that simulates actuator unreliability for reinforcement learning experiments.
- To provide deterministic-ish verification, unit tests, and a mock test harness so students and researchers can validate behavior without heavy deps (Box2D).
- To be reproducible, auditable, and easy to extend for experiment design.

Contents
- environment/
  - lunar_lander_wrapper.py — core wrapper implementing stochastic failures, fuel penalty, safe-landing bonus
  - verify_wrapper.py — large-scale verification run and JSON output
  - test_wrapper.py — pytest test suite with unit, statistical and integration checks
  - __init__.py — package exports
- RL_DQN_DDQN_Analysis.ipynb — notebook (training/analysis) included
- test_wrapper_mock.py — quick mock-based validator (no Box2D)
- Assignment 2-v2.docx — rubric / assignment (binary)
- LICENSE — project license

Table of contents
- Quick start
- Installation
- Usage examples
- Running tests & verification
- API reference (wrapper)
- Design and rationale
- Reproducibility notes
- Recommended improvements (short)
- Troubleshooting
- Contributing & license

Quick start (3 commands)
1. Create virtual env and install (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt   # see "Installation" if requirements.txt missing
   ```
2. Quick mock test (no Box2D required):
   ```
   python test_wrapper_mock.py
   ```
3. Run unit tests:
   ```
   pytest environment/test_wrapper.py -q
   ```

Installation (recommended)
- Minimal dependencies (example). Create a file `requirements.txt` with:
  ```
  numpy
  scipy
  gymnasium
  box2d-py
  pytest
  ```
- If you have trouble with Box2D on some platforms, use the mock test (test_wrapper_mock.py) or run tests that avoid Box2D.

Usage examples

- Wrap a real LunarLander environment
  ```python
  import gymnasium as gym
  from environment.lunar_lander_wrapper import StochasticFailureLunarLanderWrapper

  base_env = gym.make("LunarLander-v3")       # or "LunarLander-v2" depending on your Gym version
  env = StochasticFailureLunarLanderWrapper(base_env)

  obs, info = env.reset()
  action = env.action_space.sample()
  obs, reward, terminated, truncated, info = env.step(action)
  ```

- Quick verification run (small)
  ```python
  from environment.verify_wrapper import run_verification
  run_verification(num_episodes=50, seed=42, output_dir='.', verbose=True)
  # A full-run (num_episodes=1000) will produce verification_results.json in the cwd
  ```

- Run the mock harness (no Box2D)
  ```
  python test_wrapper_mock.py
  ```

Running tests
- Unit + integration suite:
  ```
  pytest environment/test_wrapper.py -q
  ```
- Notes:
  - The test suite includes statistical checks (failure rates). For CI, consider running a short mode or marking slow tests via pytest markers.
  - Some tests use global numpy seeding (np.random.seed). For reproducible runs, set the seed explicitly before tests or use the existing fixtures.

What the wrapper does (concise)
- Stochastic thruster failure:
  - If agent attempts actions 1, 2, or 3 (thrusters), there is a 15% chance the command is replaced with action 0 (Do Nothing).
- Fuel penalty:
  - Attempted thruster actions (regardless of whether they executed) incur a penalty of -0.3 applied to the reward.
- Safe-landing bonus:
  - A strict +50 bonus awarded only when:
    - Episode terminated (not truncated)
    - Both legs are in contact (obs[6] == 1 and obs[7] == 1)
    - |horizontal velocity| < 0.10, |vertical velocity| < 0.10
    - |angle| < 0.10 (radians)

API reference — StochasticFailureLunarLanderWrapper
- Class: StochasticFailureLunarLanderWrapper(env)
  - env: base Gym/Gymnasium LunarLander env
  - Methods:
    - reset(seed=None, options=None) -> (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info)
  - Behavior notes:
    - Observation and action spaces are passed through from the base env.
    - The wrapper does not add any failure-related keys to the info dict in default mode (no leakage).
    - Constants defined in code: STOCHASTIC_FAILURE_RATE (0.15), FUEL_PENALTY (0.3), SAFE_LANDING_BONUS (50.0), and thresholds for landing safety.

Design and rationale (short)
- Fuel penalty applied to the attempted action gives the agent feedback about its own control choices even when the actuator fails, forcing the policy to trade off fuel use vs safety.
- Failure is hidden from the agent in observations/info to force learning under partial observability (agent cannot observe whether the action executed).
- The safe-landing bonus is intentionally strict to make “perfect” landings rare and meaningful for learning.

Verification & evaluation
- environment/verify_wrapper.py runs many episodes under a random policy and collects:
  - Total attempted thruster actions
  - Total fuel penalties
  - Episodes receiving the safe-landing bonus
  - Per-episode JSON output (verification_results.json)
- Run the full verification:
  ```
  python -m environment.verify_wrapper   # or run run_verification() from Python
  ```
- The verification script is useful for sanity checks and to produce quantifiable evidence for grading.

Reproducibility notes (important)
- The wrapper uses numpy's RNG calls (np.random.random()) for stochastic failures.
  - To improve reproducibility, call:
    ```python
    import numpy as np
    np.random.seed(42)
    env.reset(seed=42)
    ```
  - Note: Gym/Gymnasium may manage its own RNG. For truly robust reproducibility we recommend injecting an RNG (np.random.Generator) into the wrapper; see "Recommended improvements" below.
- The test suite uses explicit seeding in fixtures to reduce flakiness, but full determinism across different Gym and Box2D versions is not guaranteed without further changes.

Recommended improvements (how to take this from great → 11/10)
- Add RNG injection to the wrapper (accept a np.random.Generator) for deterministic sampling and testability.
- Make failure_rate, fuel_penalty, safe_bonus constructor params (currently defined as module constants).
- Add a debug mode that records executed_action and failure booleans to a non-agent-facing log or to info only when debug=True, to enable direct verification without compromising default non-leakage behavior.
- Provide requirements.txt / pyproject.toml and a CI workflow (GitHub Actions) that runs fast tests.
- Add README mapping to Assignment 2-v2.docx rubric and a test checklist for graders (I can produce this mapping automatically — say "map docx" and I'll parse it).
- Mark long statistical tests with pytest.mark.slow and configure CI to run a fast subset.

Troubleshooting & FAQ
- "ImportError: Box2D missing" — either install box2d-py or run the mock test (test_wrapper_mock.py) which avoids Box2D.
- "Different results on different machines" — check Python, numpy, gym/gymnasium and Box2D versions; seed numpy explicitly.
- "I want to see executed actions for debugging" — currently wrapper intentionally does not leak failure data; enable debug instrumentation by modifying wrapper (recommended patch available).

Notebook (RL_DQN_DDQN_Analysis.ipynb)
- The included notebook contains training/analysis code comparing DQN/DDQN runs. Open it in Jupyter, ensure the same env & deps are installed, and follow the notebook's setup cells. For reproducibility, set seeds in the notebook and note hyperparameters and env versions.

Grading rubric mapping
- If you'd like an itemized mapping of the implementation to the Assignment 2 rubric (Assignment 2-v2.docx), I can parse the docx and produce a line-by-line map showing which requirements are met, partially met, or missing and provide recommended code or doc changes to achieve full marks. Reply "map docx" to proceed.

Contributing
- Contributions are welcome. Please:
  - Fork, create a feature branch, run tests, open a PR.
  - Keep changes small and focused. Add tests for new behavior.
  - Follow Python typing and formatting (black/flake8 recommended).

License
- See LICENSE in the repository root.

Contact & citation
- Repo owner: 2025ab05032-WILP-BITS/SEM2_DRL_ASSIGNMENT2
- If you reuse code for publications, cite the repository and note modifications.

Appendix — Commands summary
```
# Quick checks
python test_wrapper_mock.py

# Unit tests
pytest environment/test_wrapper.py -q

# Verification (small)
python -c "from environment.verify_wrapper import run_verification; run_verification(num_episodes=50, seed=42)"

# Full verification (may take long)
python -c "from environment.verify_wrapper import run_verification; run_verification(num_episodes=1000, seed=42)"
```

---

If you want, I can:
- (A) Add this README.md to the repository (create a branch and push a PR with the file and a requirements.txt + basic GitHub Actions CI).
- (B) Parse Assignment 2-v2.docx and produce a rubric mapping report and change list tied to grading criteria.
- (C) Create a minimal patch to (1) inject RNG and (2) add a debug flag (I can open a PR or show the exact diff here).

Which would you like me to do next?
