import os
import sys
import csv
import json
import math
import time
import random
from collections import deque, namedtuple
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Improve matmul precision for better numerics/speed on modern GPUs
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# Add the parent directory to sys.path to import our environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbf_base.gridworld25_env import GridWorld25v0

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Use non-interactive backend for plotting progress figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))
JointTransition = namedtuple("JointTransition", ("state", "reward_total", "next_state", "done"))
# Strategic transition for centralized DQN (multi-head)
StrategicTransition = namedtuple("StrategicTransition", ("obs_env", "actions", "reward", "next_obs_env", "done"))


class PrioritizedReplayBuffer:
    """Enhanced replay buffer with optional prioritization"""
    def __init__(self, capacity: int, alpha: float = 0.0):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.alpha = alpha  # 0 = uniform sampling, 1 = full prioritization
        self.max_priority = 1.0
        
    def push(self, *args):
        self.buffer.append(Transition(*args))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int, beta: float = 0.0):
        """Sample batch with optional importance sampling weights"""
        if self.alpha == 0.0:
            # Uniform sampling
            batch = random.sample(self.buffer, batch_size)
            return batch, None, None
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class JointReplayBuffer:
    """Replay buffer for joint state transitions"""
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
    
    def push(self, *args):
        self.buffer.append(JointTransition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class StrategicReplayBuffer:
    """Replay buffer for centralized strategic DQN"""
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
    
    def push(self, *args):
        self.buffer.append(StrategicTransition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RewardNormalizer:
    """Reward normalization using running statistics"""
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        self.eps = 1e-8
    
    def update(self, rewards: np.ndarray):
        """Update running statistics"""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        # Update running statistics
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / (self.count + batch_count)
        
        delta2 = batch_var - self.running_var
        self.running_var += delta2 * batch_count / (self.count + batch_count)
        
        self.count += batch_count
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics"""
        if self.count < 2:
            return rewards
        
        normalized = (rewards - self.running_mean) / (np.sqrt(self.running_var) + self.eps)
        return normalized


class DQNNetwork(nn.Module):
    """DQN Network with optional dueling architecture"""
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128], 
                 dropout: float = 0.0, use_dueling: bool = True):
        super().__init__()
        self.use_dueling = use_dueling
        
        # Build shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], num_actions)
            )
        else:
            # Standard DQN
            self.q_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], num_actions)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        
        if self.use_dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            # Combine value and advantage (using mean advantage for stability)
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            q_values = self.q_head(x)
        
        return q_values


class CentralValueNet(nn.Module):
    """Centralized value network for joint state evaluation"""
    def __init__(self, total_obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = total_obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CentralStrategicQNet(nn.Module):
    """Q-critic: estimates scalar value Q(s, a) where a is continuous positions (n*2)."""
    def __init__(self, obs_env_dim: int, num_agents: int):
        super().__init__()
        self.input_dim = obs_env_dim + num_agents * 2
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, obs_env: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # obs_env: [B, D], actions: [B, n*2]
        x = torch.cat([obs_env, actions], dim=-1)
        return self.net(x)


class CentralStrategicPolicy(nn.Module):
    """Actor: outputs continuous positions (n*2), scaled to valid [4,20] range."""
    def __init__(self, obs_env_dim: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(obs_env_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_agents * 2)
        )
    
    def forward(self, obs_env: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs_env)  # unconstrained
        # Map to [4,20] using tanh -> [-1,1] then scale
        scaled = torch.tanh(raw)
        # scale [-1,1] to [4,20]: ((x+1)/2)*16 + 4 = 8x + 12
        return 8.0 * scaled + 12.0


class CentralStrategicAgent:
    def __init__(self, obs_env_dim: int, num_agents: int, device: str = "cpu", lr_actor: float = 5e-4, lr_critic: float = 1e-3, gamma: float = 0.97):
        self.device = device
        self.gamma = gamma
        self.num_agents = num_agents
        self.actor = CentralStrategicPolicy(obs_env_dim, num_agents).to(device)
        self.actor_target = CentralStrategicPolicy(obs_env_dim, num_agents).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.critic = CentralStrategicQNet(obs_env_dim, num_agents).to(device)
        self.critic_target = CentralStrategicQNet(obs_env_dim, num_agents).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.opt_actor = optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.opt_critic = optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)
        self.critic_loss_fn = nn.SmoothL1Loss()
        self.grad_clip = 10.0
        self.avg_q_value = 0.0
        # AMP scalers (new API)
        self.actor_scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
        self.critic_scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    def act(self, obs_env_np: np.ndarray) -> np.ndarray:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            x = torch.tensor(obs_env_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            pos = self.actor(x)[0]  # [n*2]
            pos = pos.clamp(4.0, 20.0)
            # Enforce integer grid centers
            pos = torch.round(pos)
        return pos.detach().cpu().numpy()
    
    def learn(self, batch: List[StrategicTransition]) -> Tuple[float, float]:
        # Prepare tensors
        obs = torch.stack([torch.tensor(t.obs_env, dtype=torch.float32) for t in batch]).to(self.device)
        next_obs = torch.stack([torch.tensor(t.next_obs_env, dtype=torch.float32) for t in batch]).to(self.device)
        actions = torch.stack([torch.tensor(t.actions, dtype=torch.float32) for t in batch]).to(self.device)  # [B,n*2], integer valued
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        
        # Critic update
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            next_actions = self.actor_target(next_obs).clamp(4.0, 20.0)
            next_actions = torch.round(next_actions)
            next_q = self.critic_target(next_obs, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            current_q = self.critic(obs, actions)
            critic_loss = self.critic_loss_fn(current_q, target_q)
        self.opt_critic.zero_grad(set_to_none=True)
        self.critic_scaler.scale(critic_loss).backward()
        self.critic_scaler.unscale_(self.opt_critic)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_scaler.step(self.opt_critic)
        self.critic_scaler.update()
        
        # Actor update
        with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            pred_actions = self.actor(obs)
            actor_q = self.critic(obs, pred_actions)
            actor_loss = -actor_q.mean()
        self.opt_actor.zero_grad(set_to_none=True)
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.unscale_(self.opt_actor)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_scaler.step(self.opt_actor)
        self.actor_scaler.update()
        
        self.avg_q_value = float(actor_q.mean().detach().cpu().item())
        return float(actor_loss.item()), float(critic_loss.item())
    
    def soft_update(self, tau: float):
        with torch.no_grad():
            for t_param, p_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_param.data.copy_(tau * p_param.data + (1 - tau) * t_param.data)
            for t_param, p_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_param.data.copy_(tau * p_param.data + (1 - tau) * t_param.data)


class DQNAgent:
    """DQN Agent with target network and experience replay"""
    def __init__(self, obs_dim: int, num_actions: int, lr: float = 0.0005, 
                 gamma: float = 0.97, device: str = "cpu", hidden_dims: List[int] = [128, 128],
                 use_dueling: bool = True, use_double_dqn: bool = True):
        self.device = device
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn
        
        # Networks
        self.policy_net = DQNNetwork(obs_dim, num_actions, hidden_dims, use_dueling=use_dueling).to(device)
        self.target_net = DQNNetwork(obs_dim, num_actions, hidden_dims, use_dueling=use_dueling).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        
        # AMP scaler (new API)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
        
        # Training stats
        self.training_steps = 0
        self.last_loss = 0.0
        self.avg_q_value = 0.0
        self.grad_clip = 10.0
    
    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, 5)  # 6 actions: 0-5
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            q_values = self.policy_net(state)
            action = q_values.argmax().item()
            self.avg_q_value = float(q_values.mean().item())
        
        return action
    
    def learn(self, batch: List[Transition], weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Learn from a batch of transitions"""
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        
        with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
            # Current Q values
            q_values = self.policy_net(states).gather(1, actions)
            
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: select actions with policy net, evaluate with target net
                    next_q_policy = self.policy_net(next_states)
                    next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = self.target_net(next_states).gather(1, next_actions)
                else:
                    # Standard DQN
                    next_q_target = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                
                target_q = rewards + (1.0 - dones) * self.gamma * next_q_target
            
            # TD errors for prioritization
            td_errors = torch.abs(q_values - target_q).squeeze().detach().cpu().numpy()
            
            # Compute loss
            losses = self.criterion(q_values, target_q).squeeze()
            
            # Apply importance sampling weights if provided
            if weights is not None:
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
                loss = (losses * weights_tensor).mean()
            else:
                loss = losses.mean()
        
        # Optimize with AMP
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.training_steps += 1
        self.last_loss = float(loss.item())
        
        return self.last_loss, td_errors
    
    def soft_update(self, tau: float):
        """Soft update of target network"""
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def hard_update(self):
        """Hard update of target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy array to torch tensor"""
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_hierarchy(
    total_episodes: int = 2000,
    max_steps_per_ep: int = 200,
    buffer_capacity: int = 100000,
    batch_size: int = 128,
    gamma: float = 0.97,
    lr: float = 0.0005,
    lr_decay_factor: float = 0.995,
    soft_update_tau: float = 0.005,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 300,
    min_buffer_before_training: int = 1000,
    update_every: int = 3,
    num_updates_per_step: int = 1,
    use_prioritized_replay: bool = True,
    prioritized_alpha: float = 0.6,
    prioritized_beta_start: float = 0.4,
    prioritized_beta_end: float = 1.0,
    normalize_rewards: bool = True,
    use_dueling: bool = True,
    use_double_dqn: bool = True,
    hidden_dims: List[int] = [128, 128],
    grad_clip: float = 10.0,
    seed: int = 1,
    device: str = "auto",
    k_update: int = 5,
    gamma_str: float = 0.95
):
    """Train Hierarchy agents on the GridWorld25 environment"""
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determinism and performance toggles
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Allow TF32 where available for speed without much accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld25v0(mode="mode_2", seed=seed)
    num_agents = 4
    obs_dim = 20  # Individual agent observation dimension
    num_actions = 6  # Action space size
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/hierarchy_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # Best model directory
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    # Create tactical progress directory at same level as checkpoints
    tactical_dir = os.path.join(output_dir, "tactical_progress")
    os.makedirs(tactical_dir, exist_ok=True)
    
    # Strategic model directory
    strategic_dir = os.path.join(output_dir, "strategic_model")
    os.makedirs(strategic_dir, exist_ok=True)
    # Strategic progress plots directory
    strategic_prog_dir = os.path.join(output_dir, "strategic_progress")
    os.makedirs(strategic_prog_dir, exist_ok=True)
    
    # Logging files
    logs_csv = os.path.join(output_dir, "training_logs.csv")
    config_json = os.path.join(output_dir, "config.json")
    strategic_logs_csv = os.path.join(output_dir, "strategic_training_logs.csv")
    
    # Save configuration
    config = {
        "total_episodes": total_episodes,
        "max_steps_per_ep": max_steps_per_ep,
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "lr_decay_factor": lr_decay_factor,
        "soft_update_tau": soft_update_tau,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_episodes": epsilon_decay_episodes,
        "min_buffer_before_training": min_buffer_before_training,
        "update_every": update_every,
        "num_updates_per_step": num_updates_per_step,
        "use_prioritized_replay": use_prioritized_replay,
        "prioritized_alpha": prioritized_alpha,
        "prioritized_beta_start": prioritized_beta_start,
        "prioritized_beta_end": prioritized_beta_end,
        "normalize_rewards": normalize_rewards,
        "use_dueling": use_dueling,
        "use_double_dqn": use_double_dqn,
        "hidden_dims": hidden_dims,
        "grad_clip": grad_clip,
        "seed": seed,
        "device": device,
        "num_agents": num_agents,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "k_update": k_update,
        "gamma_str": gamma_str
    }
    
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    
    # Create agents and buffers
    agents = []
    buffers = []
    
    for i in range(num_agents):
        agent = DQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            lr=lr,
            gamma=gamma,
            device=device,
            hidden_dims=hidden_dims,
            use_dueling=use_dueling,
            use_double_dqn=use_double_dqn
        )
        agents.append(agent)
        
        if use_prioritized_replay:
            buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=prioritized_alpha)
        else:
            buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=0.0)
        buffers.append(buffer)
    
    # Centralized critic
    critic = CentralValueNet(total_obs_dim=num_agents * obs_dim).to(device)
    critic_target = CentralValueNet(total_obs_dim=num_agents * obs_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()
    
    critic_opt = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-5)
    critic_loss_fn = nn.SmoothL1Loss()
    joint_buffer = JointReplayBuffer(buffer_capacity)
    # AMP scaler for critic (new API)
    critic_scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    # Reward normalizer
    reward_normalizer = RewardNormalizer(gamma) if normalize_rewards else None
    
    # Learning rate schedulers
    agent_schedulers = [
        optim.lr_scheduler.ExponentialLR(agent.optimizer, gamma=lr_decay_factor)
        for agent in agents
    ]
    critic_scheduler = optim.lr_scheduler.ExponentialLR(critic_opt, gamma=lr_decay_factor)
    
    # Centralized strategic DQN settings
    # k_update provided by caller
    obs_env_dim = 637
    strategic_agent = CentralStrategicAgent(obs_env_dim=obs_env_dim, num_agents=num_agents, device=device, lr_actor=lr, lr_critic=lr*2, gamma=gamma)
    strategic_buffer = StrategicReplayBuffer(buffer_capacity)
    strategic_actor_losses: List[float] = []
    strategic_critic_losses: List[float] = []
    strategic_true_episode = 0  # counts only when a strategic interval closes
    
    # Setup CSV logging
    with open(logs_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "episode", "steps", "epsilon", "return_total", "return_mean", "return_std"
        ] + [f"return_agent_{i}" for i in range(num_agents)] + [
            "policy_loss_mean", "critic_loss", "avg_q_value", "buffer_size",
            "learning_rate"
        ]
        writer.writerow(header)
    # Setup strategic CSV logging
    with open(strategic_logs_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        s_header = [
            "episode", "strategic_return", "num_intervals", "actor_loss_mean", "critic_loss_mean", "avg_q_value"
        ]
        writer.writerow(s_header)
    
    # Training variables
    global_step = 0
    training_started = False
    returns_history = []
    best_return = -float('inf')
    
    # Tactical progress histories
    episodes_history: List[int] = []
    total_return_history: List[float] = []
    per_agent_return_history: List[List[float]] = [[] for _ in range(num_agents)]
    epsilon_history: List[float] = []
    policy_loss_history: List[float] = []
    critic_loss_history: List[float] = []
    avg_q_history: List[float] = []
    buffer_size_history: List[int] = []
    lr_history: List[float] = []
    
    # Strategic progress histories
    s_episode_history: List[int] = []
    s_return_history: List[float] = []
    s_actor_loss_mean_history: List[float] = []
    s_critic_loss_mean_history: List[float] = []
    
    def _save_progress_figures(current_episode: int):
        """Save tactical progress figures up to current_episode into tactical_dir.
        Note: Does NOT save per-episode overview images.
        """
        if not episodes_history:
            return
        try:
            # 1) Total return over episodes
            plt.figure(figsize=(7, 4))
            plt.plot(episodes_history, total_return_history, label="Total Return", color="#3b82f6")
            plt.xlabel("Episode")
            plt.ylabel("Total Return")
            plt.title("Total Return per Episode")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(tactical_dir, "total_return.png"))
            plt.close()
            
            # 2) Per-agent returns
            plt.figure(figsize=(7, 4))
            colors = ["#3b82f6", "#ef4444", "#10b981", "#a855f7"]
            for i in range(num_agents):
                plt.plot(episodes_history, per_agent_return_history[i], label=f"Agent {i}", color=colors[i % len(colors)])
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.title("Per-Agent Return per Episode")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(tactical_dir, "per_agent_returns.png"))
            plt.close()
            
            # 3) Epsilon schedule
            plt.figure(figsize=(7, 4))
            plt.plot(episodes_history, epsilon_history, label="Epsilon", color="#f59e0b")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.title("Epsilon over Episodes")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(tactical_dir, "epsilon.png"))
            plt.close()
            
            # 4) Losses
            plt.figure(figsize=(7, 4))
            plt.plot(episodes_history, policy_loss_history, label="Policy Loss", color="#ef4444")
            plt.plot(episodes_history, critic_loss_history, label="Critic Loss", color="#3b82f6")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("Losses over Episodes")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(tactical_dir, "losses.png"))
            plt.close()
            
            # 5) Average Q values
            plt.figure(figsize=(7, 4))
            plt.plot(episodes_history, avg_q_history, label="Avg Q", color="#10b981")
            plt.xlabel("Episode")
            plt.ylabel("Avg Q")
            plt.title("Average Q-Value over Episodes")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(tactical_dir, "avg_q.png"))
            plt.close()
            
            # 6) Buffer size and Learning rate (dual axis)
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(episodes_history, buffer_size_history, label="Buffer Size", color="#6366f1")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Buffer Size", color="#6366f1")
            ax1.tick_params(axis='y', labelcolor="#6366f1")
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(episodes_history, lr_history, label="LR", color="#f97316")
            ax2.set_ylabel("Learning Rate", color="#f97316")
            ax2.tick_params(axis='y', labelcolor="#f97316")
            fig.suptitle("Buffer Size and Learning Rate over Episodes")
            fig.tight_layout()
            fig.savefig(os.path.join(tactical_dir, "buffer_and_lr.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to save progress figures at episode {current_episode}: {e}")
    
    def _save_final_overview():
        """Save a final overview figure combining key plots."""
        if not episodes_history:
            return
        try:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            # Total return
            axs[0, 0].plot(episodes_history, total_return_history, color="#3b82f6")
            axs[0, 0].set_title("Total Return")
            axs[0, 0].grid(True, alpha=0.3)
            # Per-agent (compact)
            for i in range(num_agents):
                axs[0, 1].plot(episodes_history, per_agent_return_history[i], label=f"A{i}")
            axs[0, 1].set_title("Per-Agent Returns")
            axs[0, 1].legend(fontsize=8)
            axs[0, 1].grid(True, alpha=0.3)
            # Losses
            axs[1, 0].plot(episodes_history, policy_loss_history, label="Policy", color="#ef4444")
            axs[1, 0].plot(episodes_history, critic_loss_history, label="Critic", color="#3b82f6")
            axs[1, 0].set_title("Losses")
            axs[1, 0].legend(fontsize=8)
            axs[1, 0].grid(True, alpha=0.3)
            # Epsilon and Avg Q (dual)
            ax_q = axs[1, 1]
            ax_q2 = ax_q.twinx()
            ax_q.plot(episodes_history, epsilon_history, label="Eps", color="#f59e0b")
            ax_q.set_ylabel("Epsilon", color="#f59e0b")
            ax_q.tick_params(axis='y', labelcolor="#f59e0b")
            ax_q2.plot(episodes_history, avg_q_history, label="AvgQ", color="#10b981")
            ax_q2.set_ylabel("Avg Q", color="#10b981")
            ax_q2.tick_params(axis='y', labelcolor="#10b981")
            axs[1, 1].set_title("Epsilon & Avg Q")
            fig.suptitle("Training Overview (Final)")
            fig.tight_layout()
            fig.savefig(os.path.join(tactical_dir, "final_overview.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to save final overview: {e}")
    
    def _save_strategic_figures():
        if not s_episode_history:
            return
        try:
            # Strategic return over strategic episodes
            plt.figure(figsize=(7,4))
            plt.plot(s_episode_history, s_return_history, color="#2563eb")
            plt.xlabel("Strategic Episode (episode//k_update)")
            plt.ylabel("Discounted Return")
            plt.title("Strategic Discounted Return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(strategic_prog_dir, "strategic_return.png"))
            plt.close()
            
            # Strategic losses
            plt.figure(figsize=(7,4))
            if any(not np.isnan(x) for x in s_actor_loss_mean_history):
                plt.plot(s_episode_history, s_actor_loss_mean_history, label="Actor", color="#ef4444")
            if any(not np.isnan(x) for x in s_critic_loss_mean_history):
                plt.plot(s_episode_history, s_critic_loss_mean_history, label="Critic", color="#10b981")
            plt.xlabel("Strategic Episode")
            plt.ylabel("Loss (mean)")
            plt.title("Strategic Losses")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(strategic_prog_dir, "strategic_losses.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: failed to save strategic figures: {e}")
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Progress bar
    pbar = tqdm(range(1, total_episodes + 1), desc="Training", unit="ep")
    
    for episode in pbar:
        obs_all, obs_env, info = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        policy_losses = []
        critic_losses = []
        q_values = []
        
        # Strategic interval accumulators
        interval_reward_local_sum = 0.0
        zeros_before = int(np.sum(env.visitation_table == 0))
        strategic_interval_rewards: List[float] = []
        actor_losses_this_ep: List[float] = []
        critic_losses_this_ep: List[float] = []
        
        # Epsilon schedule
        progress = min(1.0, (episode - 1) / max(1, epsilon_decay_episodes))
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - progress)
        
        # Beta schedule for prioritized replay
        if use_prioritized_replay:
            beta = prioritized_beta_start + (prioritized_beta_end - prioritized_beta_start) * progress
        else:
            beta = 0.0
        
        for step in range(max_steps_per_ep):
            # Strategic action every k_update steps (including step 0)
            if step % k_update == 0:
                strategic_positions_cont = strategic_agent.act(obs_env)  # already rounded ints as floats
                # Apply to env: set each agent's box center (gx,gy)
                strategic_positions_int = []
                for i in range(num_agents):
                    gx = int(strategic_positions_cont[2 * i + 0])
                    gy = int(strategic_positions_cont[2 * i + 1])
                    env.set_agent_grid_position(i, gx, gy)
                    strategic_positions_int.extend([gx, gy])
                # Reset accumulators for new interval
                interval_reward_local_sum = 0.0
                zeros_before = int(np.sum(env.visitation_table == 0))
                obs_env_before = obs_env.copy()
            
            # Select actions for primitive agents
            actions = []
            for i in range(num_agents):
                obs_i = to_tensor(obs_all[i], device)
                action = agents[i].act(obs_i, epsilon)
                actions.append(action)
                q_values.append(agents[i].avg_q_value)
            
            # Environment step
            next_obs_all, next_obs_env, rewards, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            
            # Normalize rewards if enabled
            if isinstance(rewards, (list, tuple, np.ndarray)):
                rewards_array = np.asarray(rewards, dtype=np.float32)
            else:
                rewards_array = np.asarray([rewards], dtype=np.float32)

            if normalize_rewards and reward_normalizer:
                reward_normalizer.update(rewards_array)
                normalized_rewards = reward_normalizer.normalize(rewards_array)
            else:
                normalized_rewards = rewards_array
            
            # Accumulate local rewards for strategic interval (raw)
            interval_reward_local_sum += float(np.sum(rewards_array))
            
            # Store transitions for primitive agents
            for i in range(num_agents):
                obs_i = to_tensor(obs_all[i], device)
                next_obs_i = to_tensor(next_obs_all[i], device)
                reward_i = float(normalized_rewards[i])
                # Reward clipping for stability
                reward_i = float(np.clip(reward_i, -1.0, 1.0))
                buffers[i].push(obs_i, actions[i], reward_i, next_obs_i, float(done))
                episode_rewards[i].append(float(rewards[i] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards))
            
            # Joint transition for critic
            joint_state = torch.cat([to_tensor(obs_all[i], device) for i in range(num_agents)])
            joint_next_state = torch.cat([to_tensor(next_obs_all[i], device) for i in range(num_agents)])
            joint_reward = float(np.sum(normalized_rewards))
            joint_buffer.push(joint_state, joint_reward, joint_next_state, float(done))
            
            obs_all = next_obs_all
            obs_env = next_obs_env
            global_step += 1
            
            # Close strategic interval and store transition
            interval_closed = ((step + 1) % k_update == 0) or done
            if interval_closed:
                strategic_true_episode += 1
                zeros_after = int(np.sum(env.visitation_table == 0))
                newly_visited = max(0, zeros_before - zeros_after)
                global_reward = 0.1 * float(newly_visited)
                interval_total_reward = interval_reward_local_sum + global_reward
                strategic_buffer.push(obs_env_before, np.asarray(strategic_positions_int, dtype=np.float32), interval_total_reward, next_obs_env, float(done))
                strategic_interval_rewards.append(interval_total_reward)
                # Only update strategic network on these intervals (acts like its own episode)
                a_loss_val = float('nan')
                c_loss_val = float('nan')
                if len(strategic_buffer) >= batch_size:
                    batch = strategic_buffer.sample(batch_size)
                    a_loss, c_loss = strategic_agent.learn(batch)
                    strategic_actor_losses.append(a_loss)
                    strategic_critic_losses.append(c_loss)
                    actor_losses_this_ep.append(a_loss)
                    critic_losses_this_ep.append(c_loss)
                    strategic_agent.soft_update(soft_update_tau)
                # Log this strategic interval
                try:
                    with open(strategic_logs_csv, mode="a", newline="") as sf:
                        sw = csv.writer(sf)
                        sw.writerow([
                            strategic_true_episode, global_step, epsilon,
                            interval_total_reward, interval_reward_local_sum,
                            global_reward, newly_visited, a_loss_val, c_loss_val,
                            strategic_agent.avg_q_value
                        ])
                except Exception as e:
                    print(f"Warning: failed to write strategic log: {e}")
            
            # Training for primitive agents and critic
            if global_step >= min_buffer_before_training and global_step % update_every == 0:
                if not training_started:
                    print(f"\nStarting training at step {global_step}")
                    training_started = True
                
                for _ in range(num_updates_per_step):
                    # Train each agent
                    for i in range(num_agents):
                        if len(buffers[i]) >= batch_size:
                            batch, weights, indices = buffers[i].sample(batch_size, beta)
                            loss, td_errors = agents[i].learn(batch, weights if use_prioritized_replay else None)
                            policy_losses.append(loss)
                            if use_prioritized_replay and indices is not None:
                                buffers[i].update_priorities(indices, td_errors + 1e-6)
                    # Train centralized critic
                    if len(joint_buffer) >= batch_size:
                        joint_batch = joint_buffer.sample(batch_size)
                        joint_states = torch.stack([t.state for t in joint_batch]).to(device)
                        joint_next_states = torch.stack([t.next_state for t in joint_batch]).to(device)
                        joint_rewards = torch.tensor([t.reward_total for t in joint_batch], dtype=torch.float32, device=device).unsqueeze(1)
                        joint_dones = torch.tensor([t.done for t in joint_batch], dtype=torch.float32, device=device).unsqueeze(1)
                        
                        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                            next_values = critic_target(joint_next_states)
                            target_values = joint_rewards + (1.0 - joint_dones) * gamma * next_values
                        
                        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                            current_values = critic(joint_states)
                            critic_loss = critic_loss_fn(current_values, target_values)
                        
                        critic_opt.zero_grad(set_to_none=True)
                        critic_scaler.scale(critic_loss).backward()
                        critic_scaler.unscale_(critic_opt)
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
                        critic_scaler.step(critic_opt)
                        critic_scaler.update()
                        
                        critic_losses.append(float(critic_loss.item()))
                for agent in agents:
                    agent.soft_update(soft_update_tau)
                critic_target.load_state_dict(critic.state_dict())
            
            if done:
                break
        
        # Calculate episode statistics
        episode_returns = [sum(rewards) for rewards in episode_rewards]
        total_return = sum(episode_returns)
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        returns_history.append(total_return)
        if total_return > best_return:
            best_return = total_return
            # Save best snapshots
            try:
                for i, agent in enumerate(agents):
                    torch.save(agent.policy_net.state_dict(), os.path.join(best_dir, f"agent_{i}_policy.pt"))
                    torch.save(agent.target_net.state_dict(), os.path.join(best_dir, f"agent_{i}_target.pt"))
                torch.save(critic.state_dict(), os.path.join(best_dir, "critic.pt"))
                torch.save(critic_target.state_dict(), os.path.join(best_dir, "critic_target.pt"))
                # Strategic best
                torch.save(strategic_agent.actor.state_dict(), os.path.join(best_dir, "strategic_actor.pt"))
                torch.save(strategic_agent.critic.state_dict(), os.path.join(best_dir, "strategic_critic.pt"))
            except Exception as e:
                print(f"Warning: failed to save best model snapshot: {e}")
        
        # Update learning rates
        for scheduler in agent_schedulers:
            scheduler.step()
        critic_scheduler.step()
        
        # Logging
        avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_q_value = np.mean(q_values) if q_values else 0.0
        buffer_size = len(buffers[0])
        current_lr = agents[0].optimizer.param_groups[0]['lr']
        
        with open(logs_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [
                episode, global_step, epsilon, total_return, mean_return, std_return
            ] + episode_returns + [
                avg_policy_loss, avg_critic_loss, avg_q_value, buffer_size, current_lr
            ]
            writer.writerow(row)
        
        # Update tactical histories and save figures
        episodes_history.append(episode)
        total_return_history.append(float(total_return))
        for i in range(num_agents):
            per_agent_return_history[i].append(float(episode_returns[i]))
        epsilon_history.append(float(epsilon))
        policy_loss_history.append(float(avg_policy_loss))
        critic_loss_history.append(float(avg_critic_loss))
        avg_q_history.append(float(avg_q_value))
        buffer_size_history.append(int(buffer_size))
        lr_history.append(float(current_lr))
        _save_progress_figures(episode)
        
        # Strategic episodic discounted return over all intervals in this episode
        if strategic_interval_rewards:
            weights = np.power(gamma_str, np.arange(len(strategic_interval_rewards), dtype=np.float32))
            strategic_return = float(np.sum(weights * np.asarray(strategic_interval_rewards, dtype=np.float32)))
            num_intervals = len(strategic_interval_rewards)
        else:
            strategic_return = 0.0
            num_intervals = 0
        # Log every environment episode
        try:
            with open(strategic_logs_csv, mode="a", newline="") as sf:
                sw = csv.writer(sf)
                a_mean = float(np.nanmean(actor_losses_this_ep)) if actor_losses_this_ep else float('nan')
                c_mean = float(np.nanmean(critic_losses_this_ep)) if critic_losses_this_ep else float('nan')
                sw.writerow([episode, strategic_return, num_intervals, a_mean, c_mean, strategic_agent.avg_q_value])
            # Update memory and save figures
            s_episode_history.append(episode)
            s_return_history.append(strategic_return)
            s_actor_loss_mean_history.append(a_mean)
            s_critic_loss_mean_history.append(c_mean)
            _save_strategic_figures()
        except Exception as e:
            print(f"Warning: failed to write strategic episodic log: {e}")
        
        # Update progress bar
        pbar.set_postfix({
            'Return': f'{total_return:.1f}',
            'Epsilon': f'{epsilon:.3f}',
            'Loss': f'{avg_policy_loss:.4f}',
            'LR': f'{current_lr:.6f}'
        })
        
        # Save model checkpoints
        if episode % 100 == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_ep{episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(agent.policy_net.state_dict(), os.path.join(checkpoint_dir, f"agent_{i}_policy.pt"))
                torch.save(agent.target_net.state_dict(), os.path.join(checkpoint_dir, f"agent_{i}_target.pt"))
            torch.save(critic.state_dict(), os.path.join(checkpoint_dir, "critic.pt"))
            torch.save(critic_target.state_dict(), os.path.join(checkpoint_dir, "critic_target.pt"))
            torch.save(strategic_agent.actor.state_dict(), os.path.join(checkpoint_dir, "strategic_actor.pt"))
            torch.save(strategic_agent.critic.state_dict(), os.path.join(checkpoint_dir, "strategic_critic.pt"))
    
    # Final save
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), 
                  os.path.join(final_dir, f"agent_{i}_policy.pt"))
        torch.save(agent.target_net.state_dict(), 
                  os.path.join(final_dir, f"agent_{i}_target.pt"))
    
    torch.save(critic.state_dict(), 
              os.path.join(final_dir, "critic.pt"))
    torch.save(critic_target.state_dict(), 
              os.path.join(final_dir, "critic_target.pt"))
    
    # Save strategic model at same level as final_model
    torch.save(strategic_agent.actor.state_dict(), os.path.join(strategic_dir, "strategic_actor.pt"))
    torch.save(strategic_agent.critic.state_dict(), os.path.join(strategic_dir, "strategic_critic.pt"))
    
    # Training summary
    training_time = time.time() - training_start_time
    print(f"\nTraining completed!")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {global_step}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best return: {best_return:.2f}")
    print(f"Final return: {total_return:.2f}")
    print(f"Results saved to: {output_dir}")
    
    # Save a final overview figure
    try:
        _save_progress_figures(current_episode=episodes_history[-1] if episodes_history else 0)
        _save_final_overview()
    except Exception as e:
        print(f"Warning: failed to save final overview: {e}")
    
    env.close()


if __name__ == "__main__":
    train_hierarchy()
