#!/usr/bin/env python3
"""
Standalone training script for metrics extraction.
No imports from main file, just core training logic.
"""
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Dict, List, Any, Optional
import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Constants
OBS_LEG_LEFT = 6
OBS_LEG_RIGHT = 7
OBS_VEL_X = 2
OBS_VEL_Y = 3
OBS_ANGLE = 4
THRUSTER_ACTIONS = {1, 2, 3}
SAFE_LANDING_VEL_THRESHOLD = 0.10
SAFE_LANDING_ANGLE_THRESHOLD = 0.10

def set_seed(seed: int = 42) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 4) -> None:
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> None:
        self.buffer = deque(maxlen=capacity)
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self.rng = np.random.Generator(np.random.PCG64())
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if done else 0)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

class DQNAgent:
    def __init__(self, state_dim: int = 8, action_dim: int = 4, learning_rate: float = 1e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, tau: float = 1e-3, device: str = None,
                 rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self.rng = np.random.Generator(np.random.PCG64())
        
        self.q_network_local = QNetwork(state_dim, action_dim).to(self.device)
        self.q_network_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(rng=self.rng)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network_local(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def learn(self, batch_size: int = 64) -> Optional[float]:
        if not self.replay_buffer.is_ready(batch_size):
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        with torch.no_grad():
            target_q_values = self.q_network_target(next_states)
            max_target_q_values = target_q_values.max(dim=1)[0]
            y = rewards + self.gamma * max_target_q_values * (1 - dones)
        
        q_values = self.q_network_local(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = nn.MSELoss()(q_values, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._soft_update_target_network()
        return loss.item()
    
    def _soft_update_target_network(self) -> None:
        for local_param, target_param in zip(self.q_network_local.parameters(), self.q_network_target.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        states_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_network_local(states_tensor).cpu().numpy()
        return q_values

class DDQNAgent(DQNAgent):
    def learn(self, batch_size: int = 64) -> Optional[float]:
        if not self.replay_buffer.is_ready(batch_size):
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        with torch.no_grad():
            best_actions = self.q_network_local(next_states).argmax(dim=1)
            target_q_values = self.q_network_target(next_states)
            max_target_q_values = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            y = rewards + self.gamma * max_target_q_values * (1 - dones)
        
        q_values = self.q_network_local(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = nn.MSELoss()(q_values, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._soft_update_target_network()
        return loss.item()

def train_quick(num_episodes=300):
    """Quick training without verbose output"""
    set_seed(42)
    
    # Create environments
    original_env = gym.make("LunarLander-v3")
    original_env.reset(seed=42)
    modified_env = gym.make("LunarLander-v3")
    modified_env.reset(seed=42)
    
    # Validation set
    validation_states = []
    for _ in range(1000):
        obs, _ = original_env.reset()
        validation_states.append(obs)
    validation_states = np.array(validation_states)
    
    # Create agents
    agents = {
        'DQN-Original': DQNAgent(rng=np.random.Generator(np.random.PCG64(42))),
        'DDQN-Original': DDQNAgent(rng=np.random.Generator(np.random.PCG64(1042))),
        'DQN-Modified': DQNAgent(rng=np.random.Generator(np.random.PCG64(42))),
        'DDQN-Modified': DDQNAgent(rng=np.random.Generator(np.random.PCG64(1042)))
    }
    
    metrics = {name: {'rewards': [], 'landings': [], 'q_values': [], 'thrusters': []} for name in agents}
    
    # Training
    configs = [
        ('DQN-Original', agents['DQN-Original'], original_env),
        ('DDQN-Original', agents['DDQN-Original'], original_env),
        ('DQN-Modified', agents['DQN-Modified'], modified_env),
        ('DDQN-Modified', agents['DDQN-Modified'], modified_env)
    ]
    
    for name, agent, env in configs:
        print(f"\nTraining {name}...")
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=42 + episode)
            episode_reward = 0.0
            episode_landing = False
            episode_thrusters = 0
            done = False
            step_count = 0
            
            while not done and step_count < 1000:
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                if action in THRUSTER_ACTIONS:
                    episode_thrusters += 1
                
                # Simple landing check
                if (terminated and not truncated and 
                    next_obs[OBS_LEG_LEFT] == 1.0 and next_obs[OBS_LEG_RIGHT] == 1.0 and
                    abs(next_obs[OBS_VEL_X]) < SAFE_LANDING_VEL_THRESHOLD and
                    abs(next_obs[OBS_VEL_Y]) < SAFE_LANDING_VEL_THRESHOLD and
                    abs(next_obs[OBS_ANGLE]) < SAFE_LANDING_ANGLE_THRESHOLD):
                    episode_landing = True
                
                agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                
                if step_count % 4 == 0 and agent.replay_buffer.is_ready(64):
                    agent.learn(64)
                
                obs = next_obs
                done = terminated or truncated
                step_count += 1
            
            agent.decay_epsilon()
            
            # Metrics
            q_vals = agent.get_q_values(validation_states)
            avg_q = np.mean(np.max(q_vals, axis=1))
            
            metrics[name]['rewards'].append(episode_reward)
            metrics[name]['landings'].append(episode_landing)
            metrics[name]['q_values'].append(avg_q)
            metrics[name]['thrusters'].append(episode_thrusters)
            
            if (episode + 1) % 50 == 0:
                avg_r = np.mean(metrics[name]['rewards'][-50:])
                land_rate = np.mean(metrics[name]['landings'][-50:])
                print(f"  Ep {episode + 1:3d}: Avg Reward={avg_r:7.2f}, Landing={land_rate:.2%}")
    
    original_env.close()
    modified_env.close()
    return metrics

if __name__ == "__main__":
    print("=" * 80)
    print("QUICK TRAINING AND METRICS FOR QUESTION 4")
    print("=" * 80)
    
    all_metrics = train_quick(num_episodes=300)
    
    print("\n" + "=" * 80)
    print("FINAL 50 EPISODES ANALYSIS - QUESTION 4")
    print("=" * 80)
    
    for name in ['DQN-Modified', 'DDQN-Modified', 'DQN-Original', 'DDQN-Original']:
        landings = all_metrics[name]['landings']
        final_50 = landings[-50:]
        rate = np.mean(final_50) * 100
        count = sum(final_50)
        overall = np.mean(landings) * 100
        print(f"\n{name}:")
        print(f"  Final 50 episodes success rate: {rate:.1f}%")
        print(f"  Successful landings in final 50: {count}/50")
        print(f"  Overall success rate (300 ep): {overall:.1f}%")
    
    print("\n" + "=" * 80)
