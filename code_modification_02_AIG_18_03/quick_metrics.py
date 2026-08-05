#!/usr/bin/env python3
"""
Quick script to extract metrics from trained agents for Question 4 validation.
"""
import sys
sys.path.insert(0, '.')

# Import only necessary components
import numpy as np
import gymnasium as gym
import torch
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import from main script
from RL_DQN_DDQN_Analysis import (
    set_seed, StochasticFailureLunarLanderWrapper, 
    DQNAgent, DDQNAgent, ReplayBuffer,
    THRUSTER_ACTIONS, OBS_LEG_LEFT, OBS_LEG_RIGHT, 
    OBS_VEL_X, OBS_VEL_Y, OBS_ANGLE,
    SAFE_LANDING_VEL_THRESHOLD, SAFE_LANDING_ANGLE_THRESHOLD
)

def train_agents_quick(num_episodes=300, batch_size=64, update_frequency=4, seed=42):
    """Quick training function - minimal output, metrics only"""
    
    # Set seed
    rng_main = set_seed(seed)
    
    # Create environments
    original_env = gym.make("LunarLander-v3")
    original_env.reset(seed=seed)
    
    rng_wrapper = np.random.Generator(np.random.PCG64(seed))
    modified_env = StochasticFailureLunarLanderWrapper(
        gym.make("LunarLander-v3"),
        rng=rng_wrapper,
        seed=seed
    )
    modified_env.reset(seed=seed)
    
    # Sample validation set
    validation_states = []
    for _ in range(1000):
        obs, _ = original_env.reset()
        validation_states.append(obs)
    validation_states = np.array(validation_states)
    
    # Create agents
    agents = {}
    seed_dqn = seed
    seed_ddqn = seed + 1000
    
    seed_mapping = {
        'DQN-Original': seed_dqn,
        'DQN-Modified': seed_dqn,
        'DDQN-Original': seed_ddqn,
        'DDQN-Modified': seed_ddqn
    }
    
    for agent_name in ['DQN-Original', 'DDQN-Original', 'DQN-Modified', 'DDQN-Modified']:
        AgentClass = DQNAgent if 'DQN' in agent_name and 'DDQN' not in agent_name else DDQNAgent
        rng_agent = np.random.Generator(np.random.PCG64(seed_mapping[agent_name]))
        agents[agent_name] = AgentClass(rng=rng_agent, seed=seed_mapping[agent_name])
    
    # Initialize metrics
    metrics = {
        'DQN-Original': {'rewards': [], 'q_values': [], 'landings': [], 'thrusters': []},
        'DDQN-Original': {'rewards': [], 'q_values': [], 'landings': [], 'thrusters': []},
        'DQN-Modified': {'rewards': [], 'q_values': [], 'landings': [], 'thrusters': []},
        'DDQN-Modified': {'rewards': [], 'q_values': [], 'landings': [], 'thrusters': []}
    }
    
    # Training configs
    training_configs = [
        ('DQN-Original', agents['DQN-Original'], original_env, False),
        ('DDQN-Original', agents['DDQN-Original'], original_env, False),
        ('DQN-Modified', agents['DQN-Modified'], modified_env, True),
        ('DDQN-Modified', agents['DDQN-Modified'], modified_env, True)
    ]
    
    print(f"Training {len(training_configs)} agents for {num_episodes} episodes...\n")
    
    for config_name, agent, env, is_modified in training_configs:
        print(f"Training {config_name}...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            episode_thrusters = 0
            achieved_landing_bonus = False
            done = False
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                if action in THRUSTER_ACTIONS:
                    episode_thrusters += 1
                
                if (terminated and not truncated and 
                    next_obs[OBS_LEG_LEFT] == 1.0 and next_obs[OBS_LEG_RIGHT] == 1.0 and
                    abs(next_obs[OBS_VEL_X]) < SAFE_LANDING_VEL_THRESHOLD and
                    abs(next_obs[OBS_VEL_Y]) < SAFE_LANDING_VEL_THRESHOLD and
                    abs(next_obs[OBS_ANGLE]) < SAFE_LANDING_ANGLE_THRESHOLD):
                    achieved_landing_bonus = True
                
                agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                
                if step_count % update_frequency == 0 and agent.replay_buffer.is_ready(batch_size):
                    agent.learn(batch_size)
                
                obs = next_obs
                done = terminated or truncated
                step_count += 1
            
            agent.decay_epsilon()
            
            # Metrics
            q_values = agent.get_q_values(validation_states)
            max_q_per_state = np.max(q_values, axis=1)
            avg_q_value = np.mean(max_q_per_state)
            
            metrics[config_name]['rewards'].append(episode_reward)
            metrics[config_name]['q_values'].append(avg_q_value)
            metrics[config_name]['landings'].append(achieved_landing_bonus)
            metrics[config_name]['thrusters'].append(episode_thrusters)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(metrics[config_name]['rewards'][-50:])
                landing_rate = np.mean(metrics[config_name]['landings'][-50:])
                print(f"  Episode {episode + 1:3d}/{num_episodes}: Reward: {avg_reward:7.2f}, "
                      f"Landing Rate: {landing_rate:.2%}, Epsilon: {agent.epsilon:.4f}")
        
        print(f"✓ {config_name} complete\n")
    
    original_env.close()
    modified_env.close()
    
    return metrics

if __name__ == "__main__":
    print("="*80)
    print("QUICK TRAINING AND METRICS EXTRACTION")
    print("="*80)
    print()
    
    all_metrics = train_agents_quick(num_episodes=300, batch_size=64, update_frequency=4)
    
    print("\n" + "=" * 80)
    print("FINAL 50 EPISODES ANALYSIS - QUESTION 4 VALIDATION")
    print("=" * 80)
    
    for agent_name in ['DQN-Modified', 'DDQN-Modified', 'DQN-Original', 'DDQN-Original']:
        landings = all_metrics[agent_name]['landings']
        final_50_landings = landings[-50:]
        success_rate = np.mean(final_50_landings) * 100
        
        print(f"\n{agent_name}:")
        print(f"  Final 50 episodes landing success rate: {success_rate:.1f}%")
        print(f"  Landings in final 50: {sum(final_50_landings)}/50")
        print(f"  Overall success rate (300 episodes): {np.mean(landings)*100:.1f}%")
    
    print("\n" + "=" * 80)
