# Hierarchy RL - Detailed Guide

This package implements a complete research and engineering stack for the 25×25 GridWorld (LBF-style) environment with dual observations, multi-agent DQN, a centralized strategic controller, deterministic execution, result aggregation, and plotting. It also ships with a manual policy and a scripted executor for exact, reproducible control.

## Highlights and Improvements vs Classical LBF

Compared to the classic LBF environment, this repository adds:

- Dual observations: per-agent observations plus a global `obs_env` (625 grid + 12 agent-state dims) to enable centralized reasoning.
- Render pipeline that simultaneously displays animation in a window (human) and saves frames/animations (rgb_array) with proper layering.
- Region parameters per agent: each agent owns a 9×9 dotted box; its center is constrained to [4,20] so the box always fits the 25×25 grid.
- Centralized strategic controller: learns or sets the 9×9 centers every k steps (default 5), keeping agents in-region and guiding coverage.
- Region-aware reward shaping: agents are penalized for leaving their 9×9 region (border-safe), and a global coverage reward (0.1 per newly-visited cell) is available for the strategic learner.
- Visitation tracking: a 25×25 table marks visits via 3×3 blocks per step and persists “owner” of cells; this enables coverage metrics, transparent rendering, and post-hoc analysis.
- Advanced rendering: transparent, lighter cell colors for visited status; darker, thicker dotted boxes and paths on top; agent paths unoccluded.
- Deterministic execution tools: deterministic seed usage, exact scripted policies (including box centers) applied verbatim.
- Results aggregation + plotting: utilities to combine multiple runs, compute mean/std, show per-run raw + smoothed curves, and export clear, label-free publication-style figures.

## Repository Structure (key folders)

- `lbf_base/`
  - `gridworld25_env.py`: environment with all extensions (obs_env, 9×9 boxes, visitation, layered rendering, reward shaping).
- `hierarchy_RL/`
  - `train_hierarchy.py`: multi-agent DQN + centralized critic + centralized strategic actor-critic (box centers every k steps).
  - `run_hierarchy.py`: CLI wrapper for training (device, episodes, k_update, gamma_str, etc.).
  - `run_hierarchy.sh` / `run_hierarchy_seeds.sh`: shell scripts for single/multi-seed training.
  - `exec/`
    - `run_exec.py` / `run_exec.sh`: deterministic model execution; saves per-step frames and per-episode GIFs under `/exec`.
  - `tools/`
    - `combine_results.py`: combine multiple runs, smooth, and export mean curves.
    - `plot_graphs.py`: plot per-run raw+smoothed curves with mean±std (no labels), first 300 and full-length, including strategic metrics.
- `scripted_exec/`
  - `run_scripted.py`: build or load exact action/box-center sequences and execute them verbatim; saves frames and a GIF.
  - `examples/box_centers_example.json`: default per-agent script file format (four lists of actions, four lists of box centers).
- `manual_policy/` (optional)
  - `run_manual.py`: a cooperative stripe-based heuristic policy with periodic box-centering; saves frames and GIFs.

## Environment Extensions in Detail

- Observation space:
  - Per-agent: 20-dim vector including last two dims `[gx, gy]` for the agent’s 9×9 box center.
  - Global `obs_env` (shape 637): 625 flattened grid visitation + 12 agent states (x, y, level for 4 agents).
- 9×9 boxes (dotted):
  - Center clamped to `[4,20]` for both axes; guarantees box stays within the grid.
  - Drawn as thick dashed lines with high-contrast color matching the agent; always on top of visited-cell colors.
- Visitation table (25×25):
  - When an agent steps on a cell, a 3×3 block centered there is marked with its ID (1–4) only if cell is unvisited.
  - Used for coverage statistics and rendering.
- Rendering:
  - Visited cells filled with 50%-transparent, lighter agent color; agent paths and boxes stay clearly visible on top.
  - Window (human) and file saving (rgb_array) are both supported.
- Reward shaping:
  - Local region penalty: agents outside their 9×9 receive a small negative reward (border cells are safe).
  - Strategic global reward: +0.1 for each newly-visited cell (used to shape the centralized controller).

## Centralized Strategic Learning

- Observation: uses `obs_env` (637 dims).
- Action: outputs per-agent `[gx, gy]` centers, rounded to integers and clamped to `[4,20]`, every `k_update` steps.
- Architecture: deterministic actor-critic with targets, AMP, grad clipping.
- Strategic episodic return: discounted sum of interval rewards (local-sum + coverage reward) over the k-step intervals.

## Training

Single run:

```bash
./run_hierarchy.sh
```

Custom CLI:

```bash
./run_hierarchy.py \
  --total_episodes 2000 \
  --max_steps_per_ep 200 \
  --k_update 5 \
  --gamma_str 0.95 \
  --batch_size 128 --lr 5e-4 --gamma 0.97 \
  --use_dueling --use_double_dqn --use_prioritized_replay --normalize_rewards
```

Outputs are written under `hierarchy_RL/results/hierarchy_YYYYMMDD_HHMMSS/` and include `training_logs.csv`, `strategic_training_logs.csv`, `final_model/`, `strategic_model/`, `best_model/`, and `tactical_progress/` figures.

## Deterministic Execution of Trained Models

```bash
./exec/run_exec.sh /path/to/results/hierarchy_YYYYMMDD_HHMMSS  # saves to /exec
```

- Produces `frames/*.png`, `epXXX.gif`, `last_episode.gif`, `episode_returns.csv`, `exec_config.json`.
- Uses argmax action selection and rounded box centers for reproducibility.

## Exact Scripted Execution

- Per-agent script format (default): `scripted_exec/examples/box_centers_example.json`
  - `actions`: 4 arrays (agents 0–3), each of length `steps`, containing action IDs (0–5).
  - `box_centers`: 4 arrays (agents 0–3), each of length `steps/k_update`, with `[gx, gy]` integer pairs.

Run:

```bash
python ../scripted_exec/run_scripted.py --k_update 5 --seed 1 --fps 8
```

- If the per-agent JSON exists, it executes those values exactly.
- Otherwise, it auto-generates a stripe coverage plan and saves `actions.json` and `box_centers.json` alongside a `animation.gif` under `scripted_exec/scripted_exec_TIMESTAMP/`.

## Manual Policy (Optional)

```bash
python manual_policy/run_manual.py --episodes 1 --max_steps_per_ep 200 --k_update 5 --seed 3
```

- Disjoint vertical stripes and a lawnmower path per agent.
- Collect-when-adjacent, otherwise pursue stripe food or continue coverage.
- Box centers updated every `k_update` steps to upcoming targets.

## Results Aggregation and Plotting

Combine multiple runs:

```bash
python tools/combine_results.py \
  --runs <run1> <run2> <run3> <run4> <run5> \
  --out_dir hierarchy_RL/results/combined_YYYYMMDD \
  --smooth 10
```

Plot all metrics (first 300 and full-length) with per-run raw+smoothed backgrounds, and mean±std front curve:

```bash
python tools/plot_graphs.py  # defaults to your five most recent runs and final results folder
```

This script generates label-free figures for:

- Training metrics: `return_total`, `epsilon`, `avg_q_value`, `policy_loss_mean`, `critic_loss`, `buffer_size`, `learning_rate`.
- Strategic metrics (every 5 episodes): `strategic_return`, `actor_loss_mean`, `critic_loss_mean`, `avg_q_value`.

For each metric it saves:

- First 300 episodes: `<metric>_0_300.png` (and `.csv` with x, mean, std)
- Full length across runs: `<metric>_full.png` (and `.csv`)
- Special: `combined_return_total.png` (thicker mean curve)

## Reproducibility and Performance

- Deterministic random seeds for Python/NumPy/PyTorch (and CUDA).
- CuDNN deterministic mode by default; TF32 allowed where available.
- AMP for agents, centralized critic, and strategic actor-critic; gradient clipping.

## Tips

- Set `--k_update` via CLI or shell scripts to control how often the strategic box centers update.
- Use `--gamma_str` to adjust strategic episodic discounting over intervals.
- Use the scripted executor to validate exact sequences of actions and box centers verbatim.

## License

This repository is intended for research and education. Please consult your project’s licensing policies for redistribution/use.
