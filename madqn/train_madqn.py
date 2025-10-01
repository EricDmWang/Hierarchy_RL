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

import gymnasium as gym

# Ensure project root is on sys.path for package imports when running as a script
PROJECT_ROOT = "/home/dongmingwang/project/Expert_RL"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lbf_llm.mlp_model import SmallTabNet
from expert_rl import GridWorld25v0
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))
JointTransition = namedtuple("JointTransition", ("state", "reward_total", "next_state", "done"))


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
        if self.alpha == 0:
            # Uniform sampling
            batch = random.sample(self.buffer, batch_size)
            weights = np.ones(batch_size)
        else:
            # Prioritized sampling
            priorities = np.array(self.priorities) ** self.alpha
            probs = priorities / priorities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            batch = [self.buffer[idx] for idx in indices]
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
        
        return Transition(*zip(*batch)), weights, indices if self.alpha > 0 else None
    
    def update_priorities(self, indices, priorities):
        if self.alpha > 0:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class JointReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        
    def push(self, *args):
        self.buffer.append(JointTransition(*args))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return JointTransition(*zip(*batch))
        
    def __len__(self):
        return len(self.buffer)


class ImprovedDQNNetwork(nn.Module):
    """Enhanced DQN network with dueling architecture option"""
    def __init__(self, obs_dim: int, num_actions: int, hidden_dims: List[int] = [128, 128],
                 dropout: float = 0.0, use_dueling: bool = True, use_noisy: bool = False):
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


class DQNAgent:
    def __init__(self, obs_dim: int, num_actions: int, device: torch.device,
                 lr: float = 5e-4, gamma: float = 0.99, weight_decay: float = 1e-5,
                 grad_clip: float = 10.0, hidden_dims: List[int] = [128, 128],
                 use_dueling: bool = True, use_double_dqn: bool = True):
        self.device = device
        self.gamma = gamma
        self.num_actions = num_actions
        self.grad_clip = grad_clip
        self.use_double_dqn = use_double_dqn
        
        # Policy and target networks
        self.policy_net = ImprovedDQNNetwork(
            obs_dim, num_actions, hidden_dims, dropout=0.0, use_dueling=use_dueling
        ).to(device)
        
        self.target_net = ImprovedDQNNetwork(
            obs_dim, num_actions, hidden_dims, dropout=0.0, use_dueling=use_dueling
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss(reduction='none')  # For importance sampling
        
        # Track training statistics
        self.training_steps = 0
        self.last_loss = 0.0
        self.avg_q_value = 0.0
    
    @torch.no_grad()
    def act(self, obs: torch.Tensor, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        
        q_values = self.policy_net(obs.to(self.device))
        self.avg_q_value = float(q_values.mean().item())
        return int(q_values.argmax(dim=1).item())
    
    def learn(self, batch: Transition, weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        states = torch.stack(batch.state).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
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
        
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.grad_clip)
        
        self.optimizer.step()
        
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


class CentralValueNet(nn.Module):
    """Centralized critic for multi-agent coordination"""
    def __init__(self, total_obs_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = total_obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RewardNormalizer:
    """Running normalization for rewards"""
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean = 0
        self.running_var = 1
        self.count = 0
    
    def update(self, rewards: np.ndarray):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean += delta * batch_count / total_count
        
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.running_var = M2 / total_count
        self.count = total_count
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return rewards
        return (rewards - self.running_mean) / np.sqrt(self.running_var + self.epsilon)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_tensor(x, device):
    t = torch.tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.squeeze(0)


def train_madqn(
    env_id: str = "expert_rl/GridWorld25v0",
    total_episodes: int = 500,
    max_steps_per_ep: int = 200,
    buffer_capacity: int = 100_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 5e-4,
    lr_decay_factor: float = 0.995,
    lr_min: float = 1e-5,
    soft_update_tau: float = 0.005,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 300,
    min_buffer_before_training: int = 1000,
    update_every: int = 4,
    num_updates_per_step: int = 1,
    use_prioritized_replay: bool = False,
    prioritized_alpha: float = 0.6,
    prioritized_beta_start: float = 0.4,
    prioritized_beta_end: float = 1.0,
    normalize_rewards: bool = True,
    use_dueling: bool = True,
    use_double_dqn: bool = True,
    hidden_dims: List[int] = [128, 128],
    grad_clip: float = 10.0,
    seed: int = 42,
    out_dir: str = "/home/dongmingwang/project/Expert_RL/madqn/results"
):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup directories
    ensure_dir(out_dir)
    run_idx = 0
    while True:
        run_dir = os.path.join(out_dir, f"run_{run_idx}")
        if not os.path.exists(run_dir):
            break
        run_idx += 1
    
    ensure_dir(run_dir)
    models_dir = os.path.join(run_dir, "models")
    configs_dir = os.path.join(run_dir, "configs")
    ensure_dir(models_dir)
    ensure_dir(configs_dir)
    
    logs_csv = os.path.join(run_dir, "training_log.csv")
    plot_path = os.path.join(run_dir, "training_returns.png")
    
    # Initialize environment
    env = GridWorld25v0(max_steps=max_steps_per_ep, gamma=gamma, mode="mode_2",seed=1)
    obs_all, _ = env.reset()
    
    num_agents = int(obs_all.shape[0])
    obs_dim = int(obs_all.shape[1])
    num_actions = 6 if not hasattr(env.action_space, "n") else int(env.action_space.n)
    
    print(f"Environment: {num_agents} agents, obs_dim={obs_dim}, actions={num_actions}")
    
    # Initialize agents and buffers
    agents = []
    buffers = []
    
    for i in range(num_agents):
        agent = DQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            device=device,
            lr=lr,
            gamma=gamma,
            weight_decay=1e-5,
            grad_clip=grad_clip,
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
    
    # Reward normalizer
    reward_normalizer = RewardNormalizer(gamma) if normalize_rewards else None
    
    # Learning rate schedulers
    agent_schedulers = [
        optim.lr_scheduler.ExponentialLR(agent.optimizer, gamma=lr_decay_factor)
        for agent in agents
    ]
    critic_scheduler = optim.lr_scheduler.ExponentialLR(critic_opt, gamma=lr_decay_factor)
    
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
    
    # Training variables
    global_step = 0
    training_started = False
    returns_history = []
    best_return = -float('inf')
    
    # Start training timer
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    
    # Progress bar
    pbar = tqdm(range(1, total_episodes + 1), desc="Training", unit="ep")
    
    for episode in pbar:
        obs_all, _ = env.reset()
        episode_rewards = [[] for _ in range(num_agents)]
        policy_losses = []
        critic_losses = []
        q_values = []
        
        # Epsilon schedule
        progress = min(1.0, (episode - 1) / max(1, epsilon_decay_episodes))
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - progress)
        
        # Beta schedule for prioritized replay
        if use_prioritized_replay:
            beta = prioritized_beta_start + (prioritized_beta_end - prioritized_beta_start) * progress
        else:
            beta = 0.0
        
        for step in range(max_steps_per_ep):
            # Select actions
            actions = []
            for i in range(num_agents):
                obs_i = to_tensor(obs_all[i], device)
                action = agents[i].act(obs_i, epsilon)
                actions.append(action)
                q_values.append(agents[i].avg_q_value)
            
            # Environment step
            next_obs_all, rewards, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            
            # Normalize rewards if enabled (handle list, tuple, or np.ndarray uniformly)
            if isinstance(rewards, (list, tuple, np.ndarray)):
                rewards_array = np.asarray(rewards, dtype=np.float32)
            else:
                rewards_array = np.asarray([rewards], dtype=np.float32)

            if normalize_rewards and reward_normalizer:
                reward_normalizer.update(rewards_array)
                normalized_rewards = reward_normalizer.normalize(rewards_array)
            else:
                normalized_rewards = rewards_array
            
            # Store transitions
            for i in range(num_agents):
                obs_i = to_tensor(obs_all[i], device)
                next_obs_i = to_tensor(next_obs_all[i], device)
                reward_i = float(normalized_rewards[i])
                
                buffers[i].push(obs_i, actions[i], reward_i, next_obs_i, float(done))
                episode_rewards[i].append(float(rewards[i] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards))
            
            # Joint transition for critic
            joint_state = torch.cat([to_tensor(obs_all[i], device) for i in range(num_agents)])
            joint_next_state = torch.cat([to_tensor(next_obs_all[i], device) for i in range(num_agents)])
            joint_reward = float(np.sum(normalized_rewards))
            joint_buffer.push(joint_state, joint_reward, joint_next_state, float(done))
            
            obs_all = next_obs_all
            global_step += 1
            
            # Training
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
                            
                            # Update priorities if using prioritized replay
                            if use_prioritized_replay and indices is not None:
                                buffers[i].update_priorities(indices, td_errors + 1e-6)
                    
                    # Train centralized critic
                    if len(joint_buffer) >= batch_size:
                        joint_batch = joint_buffer.sample(batch_size)
                        joint_states = torch.stack(joint_batch.state).to(device)
                        joint_next_states = torch.stack(joint_batch.next_state).to(device)
                        joint_rewards = torch.tensor(joint_batch.reward_total, dtype=torch.float32, device=device).unsqueeze(1)
                        joint_dones = torch.tensor(joint_batch.done, dtype=torch.float32, device=device).unsqueeze(1)
                        
                        with torch.no_grad():
                            next_values = critic_target(joint_next_states)
                            target_values = joint_rewards + (1.0 - joint_dones) * gamma * next_values
                        
                        current_values = critic(joint_states)
                        critic_loss = critic_loss_fn(current_values, target_values)
                        
                        critic_opt.zero_grad(set_to_none=True)
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
                        critic_opt.step()
                        
                        critic_losses.append(float(critic_loss.item()))
                
                # Soft update target networks
                for agent in agents:
                    agent.soft_update(soft_update_tau)
                
                # Soft update critic target
                with torch.no_grad():
                    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                        target_param.data.copy_(soft_update_tau * param.data + (1 - soft_update_tau) * target_param.data)
            
            if done:
                break
        
        # Episode statistics: discounted returns computed here (env rewards are immediate)
        episode_returns = np.array([
            sum((gamma ** t) * r_t for t, r_t in enumerate(episode_rewards[i]))
            for i in range(num_agents)
        ])
        total_return = float(episode_returns.sum())
        mean_return = float(episode_returns.mean())
        std_return = float(episode_returns.std())
        returns_history.append(total_return)
        
        # Update learning rates
        if episode % 10 == 0 and training_started:
            for scheduler in agent_schedulers:
                scheduler.step()
            critic_scheduler.step()
        
        # Ensure minimum learning rate
        for agent in agents:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], lr_min)
        
        # Log to CSV
        current_lr = agents[0].optimizer.param_groups[0]['lr']
        avg_q = np.mean(q_values) if q_values else 0.0
        
        with open(logs_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [
                episode, step + 1, f"{epsilon:.4f}", f"{total_return:.2f}",
                f"{mean_return:.2f}", f"{std_return:.2f}"
            ] + [f"{r:.2f}" for r in episode_returns] + [
                f"{np.mean(policy_losses) if policy_losses else 0:.4f}",
                f"{np.mean(critic_losses) if critic_losses else 0:.4f}",
                f"{avg_q:.2f}",
                len(buffers[0]),
                f"{current_lr:.6f}"
            ]
            writer.writerow(row)
        
        # Save best model
        if total_return > best_return:
            best_return = total_return
            for i, agent in enumerate(agents):
                torch.save({
                    'episode': episode,
                    'state_dict': agent.policy_net.state_dict(),
                    'return': total_return
                }, os.path.join(models_dir, f"best_agent_{i}.pt"))
        
        # Periodic checkpoints
        if episode % 50 == 0:
            for i, agent in enumerate(agents):
                torch.save({
                    'episode': episode,
                    'state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'return': total_return
                }, os.path.join(models_dir, f"checkpoint_agent_{i}_ep{episode}.pt"))
        
        # Update progress bar
        pbar.set_postfix({
            'ret': f"{total_return:.1f}",
            'eps': f"{epsilon:.3f}",
            'lr': f"{current_lr:.1e}",
            'q': f"{avg_q:.1f}"
        })
    
    # Final save
    for i, agent in enumerate(agents):
        torch.save({
            'episode': total_episodes,
            'state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict()
        }, os.path.join(models_dir, f"final_agent_{i}.pt"))
    
    # Save hyperparameters
    try:
        import json
        hparams = {
            "env": env_id,
            "episodes": total_episodes,
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
            "use_prioritized_replay": use_prioritized_replay,
            "normalize_rewards": normalize_rewards,
            "use_dueling": use_dueling,
            "use_double_dqn": use_double_dqn,
            "hidden_dims": hidden_dims,
            "grad_clip": grad_clip,
            "seed": seed,
            "num_agents": num_agents,
            "obs_dim": obs_dim,
            "num_actions": num_actions,
            "best_return": best_return
        }
        
        with open(os.path.join(configs_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)
    except Exception as e:
        print(f"Failed to save hyperparameters: {e}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Returns
        episodes = np.arange(1, len(returns_history) + 1)
        ax1.plot(episodes, returns_history, alpha=0.3, label='Raw')
        
        # Moving average
        window = min(20, len(returns_history) // 10)
        if window > 1:
            moving_avg = np.convolve(returns_history, np.ones(window) / window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, label=f'MA-{window}', linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Return')
        ax1.set_title('Training Returns')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Learning curves from CSV
        import pandas as pd
        df = pd.read_csv(logs_csv)
        
        # Loss curves
        ax2.plot(df['episode'], df['policy_loss_mean'], label='Policy Loss', alpha=0.7)
        ax2.plot(df['episode'], df['critic_loss'], label='Critic Loss', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Curves')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Q-values
        ax3.plot(df['episode'], df['avg_q_value'], color='green', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Q-Value')
        ax3.set_title('Q-Value Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Epsilon and learning rate
        ax4.plot(df['episode'], df['epsilon'], label='Epsilon', color='blue', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.grid(True, alpha=0.3)
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df['episode'], df['learning_rate'], label='LR', color='red', alpha=0.7)
        ax4_twin.set_ylabel('Learning Rate', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.set_title('Exploration & Learning Rate')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training plots to {plot_path}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Calculate training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # Create training log directory
    log_dir = os.path.join(run_dir, "log")
    ensure_dir(log_dir)
    
    # Save simple training log
    training_log = {
        "total_training_time_seconds": float(total_training_time),
        "total_training_time_hours": float(total_training_time / 3600),
        "total_episodes": int(total_episodes),
        "best_return": float(best_return)
    }
    
    log_path = os.path.join(log_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining completed! Best return: {best_return:.2f}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"Results saved to: {run_dir}")
    
    return run_dir, best_return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MADQN on GridWorld25v0")
    parser.add_argument("--env_id", type=str, default="expert_rl/GridWorld25v0", help="Environment ID")
    parser.add_argument("--total_episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--buffer_capacity", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_decay_factor", type=float, default=0.995, help="Learning rate decay factor")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--soft_update_tau", type=float, default=0.005, help="Soft update parameter")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon value")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon value")
    parser.add_argument("--epsilon_decay_episodes", type=int, default=300, help="Epsilon decay episodes")
    parser.add_argument("--min_buffer_before_training", type=int, default=1000, help="Min buffer before training")
    parser.add_argument("--update_every", type=int, default=3, help="Update frequency")
    parser.add_argument("--num_updates_per_step", type=int, default=1, help="Updates per step")
    parser.add_argument("--use_prioritized_replay", action='store_true', default=True, help="Use prioritized replay")
    parser.add_argument("--prioritized_alpha", type=float, default=0.6, help="Prioritized replay alpha")
    parser.add_argument("--prioritized_beta_start", type=float, default=0.4, help="Prioritized replay beta start")
    parser.add_argument("--prioritized_beta_end", type=float, default=1.0, help="Prioritized replay beta end")
    parser.add_argument("--normalize_rewards", action='store_true', default=True, help="Normalize rewards")
    parser.add_argument("--use_dueling", action='store_true', default=True, help="Use dueling DQN")
    parser.add_argument("--use_double_dqn", action='store_true', default=True, help="Use double DQN")
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[128, 128], help="Hidden layer dimensions")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    
    args = parser.parse_args()
    
    # Run with parsed arguments
    train_madqn(
        env_id=args.env_id,
        total_episodes=args.total_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_min=args.lr_min,
        soft_update_tau=args.soft_update_tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        min_buffer_before_training=args.min_buffer_before_training,
        update_every=args.update_every,
        num_updates_per_step=args.num_updates_per_step,
        use_prioritized_replay=args.use_prioritized_replay,
        prioritized_alpha=args.prioritized_alpha,
        prioritized_beta_start=args.prioritized_beta_start,
        prioritized_beta_end=args.prioritized_beta_end,
        normalize_rewards=args.normalize_rewards,
        use_dueling=args.use_dueling,
        use_double_dqn=args.use_double_dqn,
        hidden_dims=args.hidden_dims,
        grad_clip=args.grad_clip,
        seed=args.seed,
        out_dir="/home/dongmingwang/project/Expert_RL/madqn/results",
    )