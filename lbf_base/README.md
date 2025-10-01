# Expert RL Custom Environment: GridWorld25v0

This package provides a Gymnasium-style multi-agent grid world tailored to the Little Big Foraging (LBF) observation style while supporting custom neighbor-information rules and training-friendly reward shaping.

## Environment: GridWorld25v0

- Grid size: 25 x 25
- Agents: 4 (fixed)
- Foods: exactly 4 per episode, each with level 1 (drawn as apples)
- Actions per agent (Discrete(6)):
  - 0: No-op
  - 1: Up
  - 2: Down
  - 3: Left
  - 4: Right
  - 5: Collect (move to adjacent food if possible)
- Collection mechanics:
  - **Direct movement**: Agents automatically collect food when they move directly onto it (actions 1-4)
  - **Collect action**: If agent is adjacent to food, action 5 moves them to the food position and collects it
  - Multiple foods can be collected simultaneously by different agents
- Episode termination:
  - terminated=True when all foods are collected
  - truncated=True when max_steps reached without collecting all foods
- Reward:
  - Immediate reward of 1 when food is collected (no discounting in environment)
  - If truncated with remaining foods n, every agent receives an additional terminal penalty of -0.5 * n

## Environment Modes

The environment supports two initialization modes:

### Mode 1: Random Initial Conditions (Default)
```python
env = GridWorld25v0(mode='mode_1')
```
- **Behavior**: Every call to `env.setup()` or `env.reset()` generates new random initial positions for agents and food
- **Use case**: Training with diverse initial conditions, generalization testing
- **Seed effect**: Controls randomness but doesn't fix initial positions

### Mode 2: Fixed Initial Conditions
```python
env = GridWorld25v0(mode='mode_2')
```
- **Behavior**: 
  - First call to `env.setup()` generates and stores fixed initial positions
  - Subsequent calls to `env.setup()` or `env.reset()` restore the same initial state
- **Use case**: Reproducible experiments, debugging, fair algorithm comparison
- **Seed effect**: Different seeds create different fixed initial states

## Observation (per agent, 18-d)

Follows LBF-style layout:

[sx, sy, sL,
 nx, ny, nL,
 f1x, f1y, f1L,
 f2x, f2y, f2L,
 f3x, f3y, f3L,
 f4x, f4y, f4L]

- (sx, sy, sL): self position and level (level fixed to 1)
- (nx, ny, nL): neighbor information by fixed mapping (always-on information exchange):
  - Agent 1 gets info for Agents 2 and 3 (first slot uses Agent 2)
  - Agent 2 gets info for Agents 1 and 4 (first slot uses Agent 1)
  - Agent 3 gets info for Agents 1 and 4 (first slot uses Agent 1)
  - Agent 4 gets info for Agents 2 and 3 (first slot uses Agent 2)
- f1..f4: agent-specific 3-of-4 food visibility with one hidden slot zeroed. Mapping (1-based):
  - Agent 1 sees foods 1,2,3; Agent 2 sees 2,3,4; Agent 3 sees 3,4,1; Agent 4 sees 4,1,2

## Usage Examples

### Basic Usage (Mode 1 - Random Initial Conditions)
```python
import sys
sys.path.insert(0, "/home/dongmingwang/project/Expert_RL")

from expert_rl import GridWorld25v0

env = GridWorld25v0(max_steps=200, gamma=0.95)  # Default mode_1
obs, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    actions = [env.action_space.sample() for _ in range(4)]
    obs, rewards, terminated, truncated, info = env.step(actions)

env.close()
```

### Fixed Initial Conditions (Mode 2)
```python
from expert_rl import GridWorld25v0

# Create environment with fixed initial conditions
env = GridWorld25v0(mode='mode_2', seed=42, max_steps=200, gamma=0.95)

# First setup generates and stores fixed initial state
obs, info = env.setup()
print(f"Initial agent positions: {env.agent_positions}")
print(f"Initial food positions: {env.food_positions}")

# Run episode
for step in range(50):
    actions = [0, 0, 0, 0]  # All agents stay still
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    if terminated or truncated:
        break

# Reset restores the same initial state
obs, info = env.reset()
print(f"Reset agent positions: {env.agent_positions}")  # Same as initial
print(f"Reset food positions: {env.food_positions}")    # Same as initial

env.close()
```

## Rendering Features

The environment provides enhanced visualization capabilities:

### Visual Elements
- **Grid**: 25x25 grid with coordinate axes
- **Agents**: Colored robot icons with agent IDs (0-3)
- **Food**: Red apple icons representing food items
- **Communication Links**: Light-green dashed lines showing agent communication pairs
- **Coordinates**: X/Y axis labels for precise positioning
- **Typography**: Black text with white outlines for maximum readability

### Object Labels
- **Agent Labels**: Show format `A{i}(x,y)` above each agent
  - Example: `A0(12,8)` means Agent 0 is at position (12, 8)
  - Black text with white outline for maximum visibility
- **Food Labels**: Show format `F{i}(x,y)` below each food item
  - Example: `F1(5,20)` means Food 1 is at position (5, 20)
  - Black text with white outline for maximum visibility

### Output Files
- **PNG Frames**: Individual frames saved as `frame_XXXXXX.png`
- **Animation GIF**: Complete episode animation as `animation.gif`
- **Location**: All files saved to `expert_rl/results/execution_i/`

### Example Rendering Output
```
Grid with axes:
Y
^
|
|  A0(12,8)    F1(5,20)
|     🤖         🍎
|                  F1(5,20)
|  A1(3,15)   A2(18,2)
|     🤖         🤖
|              F0(20,18)
|                  🍎
+-------------------> X
  0  5  10 15 20 25
```

## Notes

- **Movement constraints**: Agents cannot move outside the grid; out-of-bounds moves result in staying still.
- **Action validation**: Illegal actions (< 0 or > 5) are automatically mapped to no-op (action 0).
- **Collision avoidance**: Agent position swaps are blocked; multiple agents cannot occupy the same cell.
- **Food collection**: Two methods supported - direct movement onto food or collect action when adjacent.
- **Rendering**: Communication links shown as light-green dashed lines; foods as apples; agents as small robots.
- **Enhanced Visualization**: 
  - Grid axes with X/Y coordinate labels
  - Object position labels showing exact coordinates
  - Agent labels: `A0(x,y)`, `A1(x,y)`, etc.
  - Food labels: `F0(x,y)`, `F1(x,y)`, etc.
- **Frame saving**: Renders saved to `expert_rl/results/execution_i/frame_XXXXXX.png`; animation GIF created on `env.close()`.
- **Training integration**: Use list of 4 actions; observations shaped as (4, 18) for multi-agent compatibility.
- **Reward discounting**: Environment provides immediate rewards; apply discounting in training loop for episodic returns.
