#!/usr/bin/env python3
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio

# Make project imports available
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lbf_base.gridworld25_env import GridWorld25v0
from hierarchy_RL.train_hierarchy import DQNNetwork, CentralStrategicPolicy


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def load_models(model_dir: Path, num_agents: int, obs_dim: int, num_actions: int, obs_env_dim: int, device: str):
    # Load final models by default if present, else best_model
    final_dir = model_dir / "final_model"
    best_dir = model_dir / "best_model"
    strategic_dir = model_dir / "strategic_model"

    model_root = final_dir if final_dir.exists() else best_dir
    assert model_root.exists(), f"Model directory not found under {model_dir}"

    agents = []
    for i in range(num_agents):
        net = DQNNetwork(obs_dim, num_actions, [128, 128], use_dueling=True).to(device)
        state = torch.load(model_root / f"agent_{i}_policy.pt", map_location=device)
        net.load_state_dict(state)
        net.eval()
        agents.append(net)

    # Strategic actor
    strat_actor = None
    if strategic_dir.exists():
        strat_actor = CentralStrategicPolicy(obs_env_dim, num_agents).to(device)
        state = torch.load(strategic_dir / "strategic_actor.pt", map_location=device)
        strat_actor.load_state_dict(state)
        strat_actor.eval()

    return agents, strat_actor


def save_gif(frames, out_path, fps=8):
    imageio.mimsave(out_path, frames, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Execute a trained Hierarchy RL model and save render outputs")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to results/hierarchy_TIMESTAMP directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of execution episodes")
    parser.add_argument("--max_steps_per_ep", type=int, default=200)
    parser.add_argument("--k_update", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out_root", type=str, default=str(Path(__file__).parent),
                        help="Directory under which to save execution results (default: /exec)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"Run directory not found: {run_dir}"

    # Load config for dims if available
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        num_agents = int(cfg.get("num_agents", 4))
        obs_dim = int(cfg.get("obs_dim", 20))
        num_actions = int(cfg.get("num_actions", 6))
    else:
        num_agents, obs_dim, num_actions = 4, 20, 6
    obs_env_dim = 637

    # Load models
    agents, strat_actor = load_models(run_dir, num_agents, obs_dim, num_actions, obs_env_dim, device)

    # Output dir for this execution under /exec
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = run_dir.name
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"{run_tag}_exec_{timestamp}"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "source_run": str(run_dir),
        "episodes": args.episodes,
        "max_steps_per_ep": args.max_steps_per_ep,
        "k_update": args.k_update,
        "device": device,
        "fps": args.fps,
        "out_dir": str(out_dir)
    }
    with open(out_dir / "exec_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    env = GridWorld25v0(mode="mode_2", seed=args.seed)

    all_episode_returns = []

    for ep in range(1, args.episodes + 1):
        obs_all, obs_env, info = env.reset()
        ep_return = 0.0
        frames = []

        for step in range(args.max_steps_per_ep):
            # Strategic action
            if strat_actor is not None and step % args.k_update == 0:
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    x = torch.tensor(obs_env, dtype=torch.float32, device=device).unsqueeze(0)
                    pos = strat_actor(x)[0]
                    pos = torch.round(pos.clamp(4.0, 20.0)).cpu().numpy()
                for i in range(num_agents):
                    gx = int(pos[2 * i + 0])
                    gy = int(pos[2 * i + 1])
                    env.set_agent_grid_position(i, gx, gy)

            # Primitive actions
            act_list = []
            for i in range(num_agents):
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    q = agents[i](to_tensor(obs_all[i], device))
                    a = int(q.argmax().item())
                act_list.append(a)

            # Step env
            next_obs_all, next_obs_env, rewards, terminated, truncated, info = env.step(act_list)
            done = bool(terminated or truncated)

            # Accumulate full reward signal for this step
            if isinstance(rewards, (list, tuple, np.ndarray)):
                ep_return += float(np.sum(rewards))
            else:
                ep_return += float(rewards)

            # Render frame
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
                Image.fromarray(frame).save(frames_dir / f"ep{ep:03d}_step{step:04d}.png")

            obs_all, obs_env = next_obs_all, next_obs_env
            if done:
                break

        all_episode_returns.append(ep_return)
        # Save GIF per episode
        if frames:
            save_gif(frames, out_dir / f"ep{ep:03d}.gif", fps=args.fps)

    # Save a combined GIF of last episode
    if frames:
        save_gif(frames, out_dir / f"last_episode.gif", fps=args.fps)

    # Save returns
    np.savetxt(out_dir / "episode_returns.csv", np.array(all_episode_returns), delimiter=",", fmt="%.6f")

    env.close()
    print(f"Execution completed. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
