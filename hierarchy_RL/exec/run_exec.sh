#!/bin/bash

# Execution script for Hierarchy RL models
# Usage: ./exec/run_exec.sh [RUN_DIR] [EPISODES] [MAX_STEPS] [K_UPDATE] [SEED] [FPS]
# Defaults:
#   RUN_DIR   = /home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_005110
#   EPISODES  = 3
#   MAX_STEPS = 200
#   K_UPDATE  = 5
#   SEED      = 1
#   FPS       = 8

RUN_DIR=${1:-/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_013011}
EPISODES=${2:-3}
MAX_STEPS=${3:-200}
K_UPDATE=${4:-5}
SEED=${5:-1}
FPS=${6:-8}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT="$SCRIPT_DIR"

python "$SCRIPT_DIR/run_exec.py" \
  --run_dir "$RUN_DIR" \
  --episodes "$EPISODES" \
  --max_steps_per_ep "$MAX_STEPS" \
  --k_update "$K_UPDATE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --out_root "$OUT_ROOT"
