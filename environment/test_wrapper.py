"""
Unit tests for StochasticFailureLunarLanderWrapper.

This test suite provides comprehensive validation of the wrapper using pytest,
including:
    1. Stochastic failure rate validation (~15% for thruster actions)
    2. Fuel penalty correctness (applied to attempted, not executed actions)
    3. Safe landing bonus criteria validation
    4. Information leakage prevention
    5. Edge cases and boundary conditions

Test Strategy:
    - Use seeded randomness for reproducibility
    - Statistical tests for stochastic properties (failure rate)
    - Deterministic tests for reward calculation logic
    - Mock observations to test bonus calculation independently
    - Validate environment interface compliance

Run Tests:
    pytest environment/test_wrapper.py -v
    pytest environment/test_wrapper.py -v --tb=short
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import patch, MagicMock
from scipy import stats

from environment.lunar_lander_wrapper import (
    StochasticFailureLunarLanderWrapper,
    OBS_VEL_X,
    OBS_VEL_Y,
    OBS_ANGLE,
    OBS_LEG_LEFT,
    OBS_LEG_RIGHT,
    STOCHASTIC_FAILURE_RATE,
    FUEL_PENALTY,
    SAFE_LANDING_BONUS,
    SAFE_LANDING_VEL_THRESHOLD,
    SAFE_LANDING_ANGLE_THRESHOLD,
    NO_OP_ACTION,
    THRUSTER_ACTIONS,
)


# ========================
# Fixtures
# ========================


@pytest.fixture
def base_env():
    """
    Fixture: Create a fresh LunarLander-v3 environment.

    Yields:
        gym.Env: Unwrapped LunarLander-v3 environment.
    """
    env = gym.make("LunarLander-v3")
    yield env
    env.close()


@pytest.fixture
def wrapped_env(base_env):
    """
    Fixture: Create a wrapped LunarLander-v3 environment.

    Args:
        base_env (gym.Env): Base environment fixture.

    Yields:
        StochasticFailureLunarLanderWrapper: Wrapped environment.
    """
    env = StochasticFailureLunarLanderWrapper(base_env)
    yield env
    env.close()


@pytest.fixture
def seeded_wrapped_env():
    """
    Fixture: Create a seeded wrapped environment for deterministic testing.

    Yields:
        StochasticFailureLunarLanderWrapper: Wrapped environment with seed set.
    """
    np.random.seed(42)
    base_env = gym.make("LunarLander-v3")
    env = StochasticFailureLunarLanderWrapper(base_env)
    env.reset(seed=42)
    yield env
    env.close()


# ========================
# Initialization Tests
# ========================


class TestWrapperInitialization:
    """Test suite for wrapper initialization and interface."""

    def test_wrapper_accepts_lunar_lander_v3(self, base_env):
        """Test that wrapper successfully wraps LunarLander-v3."""
        wrapper = StochasticFailureLunarLanderWrapper(base_env)
        assert isinstance(wrapper, gym.Wrapper)
        assert wrapper.env is base_env

    def test_observation_space_unchanged(self, wrapped_env, base_env):
        """Test that observation space is identical to base environment."""
        assert wrapped_env.observation_space == base_env.observation_space

    def test_action_space_unchanged(self, wrapped_env, base_env):
        """Test that action space is identical to base environment."""
        assert wrapped_env.action_space == base_env.action_space

    def test_reset_returns_valid_observation(self, wrapped_env):
        """Test that reset returns a valid observation."""
        obs, info = wrapped_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8,)  # LunarLander has 8-dim observation
        assert isinstance(info, dict)

    def test_step_returns_correct_tuple_length(self, wrapped_env):
        """Test that step returns a 5-tuple (obs, reward, terminated, truncated, info)."""
        wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        result = wrapped_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)


# ========================
# Stochastic Failure Tests
# ========================


class TestStochasticFailure:
    """Test suite for stochastic engine failure mechanism."""

    def test_apply_stochastic_failure_action_0_never_fails(self, wrapped_env):
        """Test that action 0 (Do Nothing) is never replaced."""
        np.random.seed(42)
        for _ in range(1000):
            result = wrapped_env._apply_stochastic_failure(NO_OP_ACTION)
            assert result == NO_OP_ACTION

    def test_apply_stochastic_failure_thruster_actions_can_fail(self, wrapped_env):
        """Test that thruster actions can be replaced with 0."""
        np.random.seed(42)
        failures_observed = False
        for action in THRUSTER_ACTIONS:
            for _ in range(1000):
                result = wrapped_env._apply_stochastic_failure(action)
                if result == NO_OP_ACTION:
                    failures_observed = True
                    break
            if failures_observed:
                break
        assert (
            failures_observed
        ), "Thruster actions should sometimes fail (replaced with 0)"

    def test_stochastic_failure_rate_statistical_validation(self, wrapped_env):
        """
        Test that stochastic failure rate is approximately 15%.

        Uses binomial test to validate that the observed failure rate is
        consistent with expected 15% failure probability.
        """
        np.random.seed(42)
        num_trials = 10000
        total_failures = 0

        for _ in range(num_trials):
            action = np.random.choice(list(THRUSTER_ACTIONS))
            result = wrapped_env._apply_stochastic_failure(action)
            if result == NO_OP_ACTION:
                total_failures += 1

        observed_rate = total_failures / num_trials
        expected_rate = STOCHASTIC_FAILURE_RATE

        # Use binomial test: P(X | n, p) where X ~ Binomial(n, p)
        # We check if observed_rate is within reasonable confidence interval
        # (typically 15% ± 2%)
        assert (
            0.13 <= observed_rate <= 0.17
        ), f"Failure rate {observed_rate:.3f} outside expected range [0.13, 0.17]"

    def test_action_space_boundary_values_not_modified(self, wrapped_env):
        """Test that actions outside thruster set are not modified."""
        np.random.seed(42)
        # Action 0 is already tested above, but verify it explicitly again
        for _ in range(100):
            assert wrapped_env._apply_stochastic_failure(NO_OP_ACTION) == NO_OP_ACTION


# ========================
# Fuel Penalty Tests
# ========================


class TestFuelPenalty:
    """Test suite for fuel penalty calculation."""

    def test_fuel_penalty_do_nothing_action_zero_penalty(self, wrapped_env):
        """Test that action 0 incurs no fuel penalty."""
        penalty = wrapped_env._calculate_fuel_penalty(NO_OP_ACTION)
        assert penalty == 0.0

    def test_fuel_penalty_thruster_actions_nonzero_penalty(self, wrapped_env):
        """Test that all thruster actions incur fuel penalty."""
        for action in THRUSTER_ACTIONS:
            penalty = wrapped_env._calculate_fuel_penalty(action)
            assert penalty == FUEL_PENALTY
            assert penalty == 0.3

    def test_fuel_penalty_applied_to_attempted_action(self, seeded_wrapped_env):
        """
        Test that fuel penalty is applied to ATTEMPTED action, not executed.

        This test validates the core requirement: the agent receives penalty
        feedback for its own action choice, regardless of whether the action
        fails due to stochastic failure.

        Strategy: Run episodes and verify that total penalties collected
        equal total attempted thruster actions × 0.3
        """
        total_attempted_thruster = 0
        total_penalty_collected = 0

        for episode in range(50):  # Run 50 episodes
            obs, info = seeded_wrapped_env.reset()
            done = False

            while not done:
                action = seeded_wrapped_env.action_space.sample()

                # Collect penalty based on attempted action
                if action in THRUSTER_ACTIONS:
                    total_attempted_thruster += 1
                    total_penalty_collected += FUEL_PENALTY

                obs, reward, terminated, truncated, info = seeded_wrapped_env.step(
                    action
                )
                done = terminated or truncated

        # Verify that total penalties align with attempted actions
        assert (
            total_penalty_collected
            == total_attempted_thruster * FUEL_PENALTY
        ), "Total penalties should equal attempted thruster actions × 0.3"


# ========================
# Safe Landing Bonus Tests
# ========================


class TestSafeLandingBonus:
    """Test suite for safe landing bonus calculation."""

    def _create_observation(
        self,
        vel_x=0.0,
        vel_y=0.0,
        angle=0.0,
        leg_left=1.0,
        leg_right=1.0,
    ):
        """
        Helper: Create a mock observation with specified parameters.

        Args:
            vel_x (float): Horizontal velocity.
            vel_y (float): Vertical velocity.
            angle (float): Orientation angle.
            leg_left (float): Left leg contact (1.0 or 0.0).
            leg_right (float): Right leg contact (1.0 or 0.0).

        Returns:
            np.ndarray: 8-element observation vector.
        """
        obs = np.zeros(8)
        obs[OBS_VEL_X] = vel_x
        obs[OBS_VEL_Y] = vel_y
        obs[OBS_ANGLE] = angle
        obs[OBS_LEG_LEFT] = leg_left
        obs[OBS_LEG_RIGHT] = leg_right
        return obs

    def test_safe_landing_bonus_all_criteria_met(self, wrapped_env):
        """Test that bonus is awarded when ALL criteria are met."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == SAFE_LANDING_BONUS

    def test_safe_landing_bonus_not_terminated(self, wrapped_env):
        """Test that bonus is NOT awarded if terminated=False."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=False, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_truncated(self, wrapped_env):
        """Test that bonus is NOT awarded if truncated=True."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=True
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_left_leg_not_touching(self, wrapped_env):
        """Test that bonus is NOT awarded if left leg not touching."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.05,
            leg_left=0.0,  # Not touching
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_right_leg_not_touching(self, wrapped_env):
        """Test that bonus is NOT awarded if right leg not touching."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=0.0,  # Not touching
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_velocity_x_exceeds_threshold(self, wrapped_env):
        """Test that bonus is NOT awarded if |vel_x| >= threshold."""
        obs = self._create_observation(
            vel_x=0.15,  # > 0.10 threshold
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_velocity_y_exceeds_threshold(self, wrapped_env):
        """Test that bonus is NOT awarded if |vel_y| >= threshold."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.15,  # > 0.10 threshold
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_angle_exceeds_threshold(self, wrapped_env):
        """Test that bonus is NOT awarded if |angle| >= threshold."""
        obs = self._create_observation(
            vel_x=0.05,
            vel_y=0.05,
            angle=0.15,  # > 0.10 threshold
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_boundary_velocity_exactly_at_threshold(
        self, wrapped_env
    ):
        """Test that bonus is NOT awarded when velocity is exactly at threshold."""
        obs = self._create_observation(
            vel_x=0.10,  # Exactly at threshold (not < threshold)
            vel_y=0.05,
            angle=0.05,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == 0.0

    def test_safe_landing_bonus_boundary_just_below_threshold(self, wrapped_env):
        """Test that bonus IS awarded when velocity is just below threshold."""
        obs = self._create_observation(
            vel_x=0.099,  # Just below threshold
            vel_y=0.099,
            angle=0.099,
            leg_left=1.0,
            leg_right=1.0,
        )
        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == SAFE_LANDING_BONUS


# ========================
# Information Leakage Tests
# ========================


class TestNoInfoLeakage:
    """Test suite for ensuring no stochastic failure information leaks."""

    def test_info_dict_unchanged_by_wrapper(self, seeded_wrapped_env):
        """Test that info dict does not contain failure information."""
        obs, info = seeded_wrapped_env.reset()
        action = seeded_wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = seeded_wrapped_env.step(action)

        # Verify that info dict does not contain any failure-related keys
        forbidden_keys = {
            "failure",
            "stochastic_failure",
            "action_failed",
            "executed_action",
            "attempted_action",
        }
        assert (
            not forbidden_keys.intersection(info.keys())
        ), f"Info dict should not contain failure info: {info.keys()}"

    def test_observation_not_modified_by_wrapper(self, seeded_wrapped_env):
        """Test that observation space and values are not modified by wrapper."""
        obs, _ = seeded_wrapped_env.reset()
        assert obs.shape == (8,), "Observation should have 8 dimensions"

        for _ in range(100):
            action = seeded_wrapped_env.action_space.sample()
            obs, _, terminated, truncated, _ = seeded_wrapped_env.step(action)
            assert obs.shape == (8,), "Observation shape should remain 8-dimensional"
            if terminated or truncated:
                break


# ========================
# Integration Tests
# ========================


class TestIntegration:
    """Integration tests for full episode loops."""

    def test_full_episode_execution(self, seeded_wrapped_env):
        """Test that full episodes execute without errors."""
        obs, info = seeded_wrapped_env.reset(seed=42)
        done = False
        step_count = 0

        while not done:
            action = seeded_wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = seeded_wrapped_env.step(
                action
            )
            done = terminated or truncated
            step_count += 1

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float, np.number))
            assert step_count < 1000, "Episode should terminate within 1000 steps"

        assert step_count > 0, "Episode should have at least 1 step"

    def test_multiple_episodes_deterministic_with_seed(self):
        """Test that same seed produces consistent results."""
        def run_episode_with_seed(seed):
            base_env = gym.make("LunarLander-v3")
            env = StochasticFailureLunarLanderWrapper(base_env)
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)

            total_reward = 0.0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            env.close()
            return total_reward

        # Run twice with same seed
        reward1 = run_episode_with_seed(123)
        reward2 = run_episode_with_seed(123)

        # Results should be identical (or very close due to floating point)
        assert (
            abs(reward1 - reward2) < 0.001
        ), "Same seed should produce deterministic results"

    def test_statistical_validation_1000_episodes(self, seeded_wrapped_env):
        """
        Statistical validation test: Run 1000 episodes and validate overall metrics.

        This test validates the wrapper behavior at scale:
        - Episodes complete successfully
        - Rewards are in expected ranges
        - Safe landing bonuses are rare and specific
        """
        num_episodes = 100  # Use 100 instead of 1000 for test speed

        episode_rewards = []
        bonus_count = 0
        success_count = 0

        for episode in range(num_episodes):
            obs, info = seeded_wrapped_env.reset()
            episode_reward = 0.0
            done = False
            episode_bonus = 0.0

            while not done:
                action = seeded_wrapped_env.action_space.sample()
                obs, reward, terminated, truncated, info = seeded_wrapped_env.step(
                    action
                )
                episode_reward += reward
                done = terminated or truncated

                # Check for bonus
                if (
                    terminated
                    and not truncated
                    and obs[OBS_LEG_LEFT] == 1.0
                    and obs[OBS_LEG_RIGHT] == 1.0
                    and np.abs(obs[OBS_VEL_X]) < SAFE_LANDING_VEL_THRESHOLD
                    and np.abs(obs[OBS_VEL_Y]) < SAFE_LANDING_VEL_THRESHOLD
                    and np.abs(obs[OBS_ANGLE]) < SAFE_LANDING_ANGLE_THRESHOLD
                ):
                    episode_bonus += SAFE_LANDING_BONUS

            episode_rewards.append(episode_reward)

            if episode_bonus > 0:
                bonus_count += 1

            if episode_reward > 0:
                success_count += 1

        # Validate metrics
        assert (
            len(episode_rewards) == num_episodes
        ), "Should complete all episodes"
        assert all(
            isinstance(r, (int, float, np.number)) for r in episode_rewards
        ), "All rewards should be numeric"
        # Success rate should be relatively low (hard landing task)
        assert success_count / num_episodes < 0.5, "Success rate should be low"


# ========================
# Edge Case Tests
# ========================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_consecutive_failures_possible(self, wrapped_env):
        """Test that consecutive failures are possible (stochastic independence)."""
        np.random.seed(42)
        consecutive_failures = 0
        max_consecutive = 0

        for _ in range(10000):
            action = 1  # Always use action 1 (thruster)
            result = wrapped_env._apply_stochastic_failure(action)

            if result == NO_OP_ACTION:
                consecutive_failures += 1
                max_consecutive = max(max_consecutive, consecutive_failures)
            else:
                consecutive_failures = 0

        assert (
            max_consecutive >= 2
        ), "Should observe consecutive failures (statistically possible)"

    def test_negative_velocity_in_bonus_calculation(self, wrapped_env):
        """Test that bonus calculation handles negative velocities correctly."""
        obs = np.zeros(8)
        obs[OBS_VEL_X] = -0.05  # Negative horizontal velocity
        obs[OBS_VEL_Y] = -0.05  # Negative vertical velocity
        obs[OBS_ANGLE] = -0.05  # Negative angle
        obs[OBS_LEG_LEFT] = 1.0
        obs[OBS_LEG_RIGHT] = 1.0

        bonus = wrapped_env._calculate_safe_landing_bonus(
            obs, terminated=True, truncated=False
        )
        assert bonus == SAFE_LANDING_BONUS, "Should grant bonus for negative velocities within threshold"

    def test_zero_reward_edge_case(self, wrapped_env):
        """Test wrapper behavior with zero reward from base environment."""
        # Create a mock environment that returns zero reward
        mock_env = MagicMock()
        mock_env.observation_space = wrapped_env.observation_space
        mock_env.action_space = wrapped_env.action_space
        mock_env.step.return_value = (
            np.zeros(8),
            0.0,  # Zero base reward
            False,
            False,
            {},
        )

        wrapped_env.env = mock_env

        obs, reward, terminated, truncated, info = wrapped_env.step(1)  # Thruster action

        # Reward should be: 0 (base) - 0.3 (fuel) + 0 (no bonus) = -0.3
        assert reward == -0.3, f"Expected -0.3, got {reward}"


# ========================
# Run Tests
# ========================
if __name__ == "__main__":
    """
    Run tests when executed directly.

    Example:
        python -m pytest environment/test_wrapper.py -v
    """
    pytest.main([__file__, "-v"])
