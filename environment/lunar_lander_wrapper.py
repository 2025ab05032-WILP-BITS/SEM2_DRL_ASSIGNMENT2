"""
Custom Gymnasium wrapper for LunarLander-v3 with stochastic engine failures.

This module implements StochasticFailureLunarLanderWrapper, which wraps the official
LunarLander-v3 environment and introduces:
    1. 15% probability that thruster commands (actions 1, 2, 3) are replaced with "Do Nothing" (action 0)
    2. Modified reward function with fuel penalty for attempted (not executed) thruster actions
    3. +50 bonus for perfectly safe landings meeting strict criteria
    4. No information leakage: the agent never knows when actions fail

The wrapper preserves the observation and action spaces while modifying only the
reward signal and the executed action.

Mathematical Formulation:
    - Stochastic Failure: If a ∈ {1, 2, 3}, replace with 0 with probability 0.15
    - Fuel Penalty: -0.3 applied to the *attempted* action, not the executed action
    - Safe Landing Bonus: B = 50 if all conditions met, else B = 0
    - Final Reward: R = R_base - 0.3 × 𝟙(a ≠ 0) + B

Observation Space (indices):
    obs[0]: x-position
    obs[1]: y-position
    obs[2]: horizontal velocity (v_x)
    obs[3]: vertical velocity (v_y)
    obs[4]: orientation angle
    obs[5]: angular velocity
    obs[6]: left leg contact (1 if touching, 0 otherwise)
    obs[7]: right leg contact (1 if touching, 0 otherwise)

Action Space:
    0: Do Nothing
    1: Left Thruster
    2: Main Thruster (down)
    3: Right Thruster
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

# ========================
# Observation Space Constants
# ========================
# These constants define indices into the observation vector for clarity and maintainability
OBS_X_POS = 0                   # x-position of the lander
OBS_Y_POS = 1                   # y-position of the lander
OBS_VEL_X = 2                   # horizontal velocity
OBS_VEL_Y = 3                   # vertical velocity
OBS_ANGLE = 4                   # orientation angle (radians)
OBS_ANGULAR_VEL = 5             # angular velocity
OBS_LEG_LEFT = 6                # left leg contact flag (1 if touching ground)
OBS_LEG_RIGHT = 7               # right leg contact flag (1 if touching ground)

# ========================
# Wrapper Configuration Constants
# ========================
STOCHASTIC_FAILURE_RATE = 0.15  # 15% probability of thruster failure
FUEL_PENALTY = 0.3              # Penalty per attempted thruster action
SAFE_LANDING_BONUS = 50.0       # Bonus for perfect safe landings

# Thresholds for safe landing criteria
SAFE_LANDING_VEL_THRESHOLD = 0.10   # Max absolute velocity (m/s)
SAFE_LANDING_ANGLE_THRESHOLD = 0.10 # Max absolute angle (radians)

# Thruster action indices (the actions subject to stochastic failure)
THRUSTER_ACTIONS = {1, 2, 3}    # Actions that can fail
NO_OP_ACTION = 0                # "Do Nothing" action (never fails)


class StochasticFailureLunarLanderWrapper(gym.Wrapper):
    """
    Custom Gymnasium wrapper that adds stochastic engine failures to LunarLander-v3.

    This wrapper simulates realistic robotic systems where actuator commands may fail.
    It introduces:
        1. A 15% chance that thruster commands are replaced with "Do Nothing"
        2. A fuel penalty applied to attempted (not executed) thruster actions
        3. A +50 bonus for perfect safe landings meeting all strict criteria
        4. No information leakage to the agent about failed commands

    The wrapper preserves the original observation and action spaces while modifying
    only the reward signal and the action executed by the base environment.

    Args:
        env (gym.Env): The base LunarLander-v3 environment to wrap.
                       Must be a Gymnasium environment with the standard LunarLander interface.

    Attributes:
        env (gym.Env): The wrapped base environment.
        observation_space (gym.spaces.Box): Unchanged from base environment.
        action_space (gym.spaces.Discrete): Unchanged from base environment.

    Example:
        >>> import gymnasium as gym
        >>> from environment import StochasticFailureLunarLanderWrapper
        >>> base_env = gym.make("LunarLander-v3")
        >>> wrapped_env = StochasticFailureLunarLanderWrapper(base_env)
        >>> obs, info = wrapped_env.reset()
        >>> action = wrapped_env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = wrapped_env.step(action)
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Initialize the stochastic failure wrapper.

        Args:
            env (gym.Env): The base LunarLander-v3 environment.

        Raises:
            AssertionError: If the environment is not LunarLander-v3.
        """
        super().__init__(env)

        # Validate that the wrapped environment is LunarLander-v3
        if not isinstance(env.unwrapped, gym.envs.box2d.lunar_lander.LunarLander):
            raise ValueError(
                "StochasticFailureLunarLanderWrapper must wrap LunarLander-v3. "
                f"Received: {type(env.unwrapped)}"
            )

        # Copy spaces from base environment (unchanged)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step with stochastic engine failure and reward modification.

        This method implements the core wrapper logic:
            1. Store the attempted action `a`
            2. Apply stochastic failure: if a ∈ {1, 2, 3}, replace with 0 with 15% probability
            3. Pass the executed action to the base environment
            4. Calculate modified reward:
               - Fuel penalty: -0.3 if attempted action != 0 (applied to attempted, not executed action)
               - Safe landing bonus: +50 if all landing criteria met
            5. Return the observation, modified reward, and termination flags

        Mathematical Formulation:
            - Executed action: a_exec = 0 with prob 0.15 if a ∈ {1, 2, 3}, else a_exec = a
            - Fuel penalty: P = 0.3 if a != 0 else 0  (applied to attempted action a)
            - Safe landing bonus: B = 50 if (terminated ∧ ¬truncated ∧ obs[6] ∧ obs[7]
                                            ∧ |obs[2]| < 0.10 ∧ |obs[3]| < 0.10 ∧ |obs[4]| < 0.10)
                                  else B = 0
            - Final reward: R = R_base - P + B

        Args:
            action (int): The action chosen by the RL agent (not modified by wrapper).
                         Must be in action_space {0, 1, 2, 3}.

        Returns:
            obs (np.ndarray): Observation from the environment (unchanged).
            reward (float): Modified reward including fuel penalty and landing bonus.
            terminated (bool): Whether the episode reached a terminal state.
            truncated (bool): Whether the episode was truncated (timeout).
            info (dict): Info dict from the base environment (no failure info added).

        Notes:
            - The agent does NOT know whether its action was replaced. The observation
              reflects the actual environment state resulting from the executed action,
              but no information about the failure is added to the info dict.
            - The fuel penalty is applied to the *attempted* action, not the executed
              action, so the agent receives consistent feedback about its own policy.
            - The safe landing bonus is only granted if ALL criteria are met: both legs
              touching, low velocities, low angle, and no truncation.
        """
        # ========================================
        # Step 1: Store the attempted action
        # ========================================
        attempted_action = action

        # ========================================
        # Step 2: Apply stochastic engine failure
        # ========================================
        # If the attempted action is a thruster command (1, 2, or 3),
        # there is a 15% chance it is replaced with "Do Nothing" (0).
        executed_action = self._apply_stochastic_failure(attempted_action)

        # ========================================
        # Step 3: Execute the action in base environment
        # ========================================
        # Pass the executed action (which may differ from attempted action) to the base env
        obs, reward_base, terminated, truncated, info = self.env.step(executed_action)

        # ========================================
        # Step 4: Calculate modified reward
        # ========================================
        # Calculate fuel penalty based on ATTEMPTED action (not executed)
        fuel_penalty = self._calculate_fuel_penalty(attempted_action)

        # Calculate safe landing bonus based on final state
        safe_landing_bonus = self._calculate_safe_landing_bonus(
            obs, terminated, truncated
        )

        # Compute final reward
        modified_reward = reward_base - fuel_penalty + safe_landing_bonus

        # ========================================
        # Step 5: Return environment transition
        # ========================================
        # Return the observation, modified reward, and flags.
        # Crucially, we do NOT add failure information to the info dict.
        return obs, modified_reward, terminated, truncated, info

    def _apply_stochastic_failure(self, action: int) -> int:
        """
        Apply stochastic engine failure to the attempted action.

        If the action is a thruster command (1, 2, or 3), there is a 15% probability
        that it is replaced with "Do Nothing" (0). Otherwise, the action is unchanged.

        Args:
            action (int): The attempted action chosen by the RL agent.

        Returns:
            int: The executed action (either the same as input or replaced with 0).

        Notes:
            - Action 0 (Do Nothing) is never modified.
            - Actions outside {0, 1, 2, 3} are not modified (passed through unchanged).
            - This function uses np.random to sample failure with the specified probability.
        """
        # If action is not a thruster command, return it unchanged
        if action not in THRUSTER_ACTIONS:
            return action

        # Sample random failure: if random < 0.15, replace action with 0
        if np.random.random() < STOCHASTIC_FAILURE_RATE:
            return NO_OP_ACTION
        else:
            return action

    def _calculate_fuel_penalty(self, attempted_action: int) -> float:
        """
        Calculate the fuel penalty based on the ATTEMPTED action.

        The fuel penalty is applied to the action the agent *tried* to execute,
        not the action that was actually executed. This ensures the agent receives
        consistent feedback about its policy choices, regardless of stochastic failures.

        Args:
            attempted_action (int): The action chosen by the RL agent (before stochastic failure).

        Returns:
            float: The fuel penalty (0.0 if action 0, 0.3 if any thruster action).

        Rationale:
            - If attempted_action != 0 (any thruster command), penalty = 0.3
            - If attempted_action == 0 (Do Nothing), penalty = 0.0
            - This penalty is independent of whether the action actually executed
              (which may have failed due to stochastic failure).
        """
        if attempted_action != NO_OP_ACTION:
            return FUEL_PENALTY
        else:
            return 0.0

    def _calculate_safe_landing_bonus(
        self, obs: np.ndarray, terminated: bool, truncated: bool
    ) -> float:
        """
        Calculate the safe landing bonus based on strict landing criteria.

        A bonus of +50 is awarded ONLY if ALL of the following conditions are met:
            1. terminated == True (episode reached a terminal state, i.e., lander touched ground)
            2. truncated == False (episode was not truncated; we didn't run out of time)
            3. obs[6] == 1 (left leg is in contact with ground)
            4. obs[7] == 1 (right leg is in contact with ground)
            5. |obs[2]| < 0.10 (horizontal velocity is very low)
            6. |obs[3]| < 0.10 (vertical velocity is very low)
            7. |obs[4]| < 0.10 (orientation angle is nearly vertical, very small)

        If any condition is not met, the bonus is 0.

        Args:
            obs (np.ndarray): The observation vector from the environment.
                             Length-8 array with elements defined by OBS_* constants.
            terminated (bool): Whether the episode reached a terminal state.
            truncated (bool): Whether the episode was truncated (timed out).

        Returns:
            float: +50.0 if all conditions are met, else 0.0.

        Notes:
            - All conditions must be True simultaneously for the bonus to apply.
            - This bonus is stricter than the default "successful landing" reward in
              LunarLander-v3, ensuring only "perfect" landings receive this extra reward.
            - The velocity and angle thresholds ensure the lander is nearly stationary
              and vertical, simulating a smooth, controlled landing.
        """
        # Check all landing criteria
        is_terminated = terminated
        is_not_truncated = not truncated
        left_leg_touching = obs[OBS_LEG_LEFT] == 1.0
        right_leg_touching = obs[OBS_LEG_RIGHT] == 1.0
        horizontal_velocity_safe = np.abs(obs[OBS_VEL_X]) < SAFE_LANDING_VEL_THRESHOLD
        vertical_velocity_safe = np.abs(obs[OBS_VEL_Y]) < SAFE_LANDING_VEL_THRESHOLD
        angle_safe = np.abs(obs[OBS_ANGLE]) < SAFE_LANDING_ANGLE_THRESHOLD

        # Award bonus only if ALL criteria are met
        if (
            is_terminated
            and is_not_truncated
            and left_leg_touching
            and right_leg_touching
            and horizontal_velocity_safe
            and vertical_velocity_safe
            and angle_safe
        ):
            return SAFE_LANDING_BONUS
        else:
            return 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return initial observation.

        This method delegates to the base environment's reset method, preserving
        the standard Gymnasium interface.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for the base environment.

        Returns:
            obs (np.ndarray): Initial observation from the environment.
            info (dict): Info dict from the base environment reset.
        """
        return self.env.reset(seed=seed, options=options)
