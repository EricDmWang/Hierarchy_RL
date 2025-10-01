#!/bin/bash

# Hierarchy Training Script for Hierarchy RL
# This script runs Hierarchy training with default parameters

echo "Starting Hierarchy training for Hierarchy RL..."
echo "=========================================="

# Set default parameters
TOTAL_EPISODES=300
MAX_STEPS_PER_EP=200
BATCH_SIZE=128
LR=0.0005
GAMMA=0.97
HIDDEN_DIMS="128 128"
BUFFER_CAPACITY=100000
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY_EPISODES=300
MIN_BUFFER_BEFORE_TRAINING=1000
UPDATE_EVERY=3
SOFT_UPDATE_TAU=0.005
GRAD_CLIP=10.0
LR_DECAY_FACTOR=0.995
SEED=1
DEVICE="auto"
K_UPDATE=5
LAMBDA_STR=0.95

# Create results directory
mkdir -p results

# Run training
./run_hierarchy.py \
    --total_episodes $TOTAL_EPISODES \
    --max_steps_per_ep $MAX_STEPS_PER_EP \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --gamma $GAMMA \
    --hidden_dims $HIDDEN_DIMS \
    --buffer_capacity $BUFFER_CAPACITY \
    --epsilon_start $EPSILON_START \
    --epsilon_end $EPSILON_END \
    --epsilon_decay_episodes $EPSILON_DECAY_EPISODES \
    --min_buffer_before_training $MIN_BUFFER_BEFORE_TRAINING \
    --update_every $UPDATE_EVERY \
    --soft_update_tau $SOFT_UPDATE_TAU \
    --grad_clip $GRAD_CLIP \
    --lr_decay_factor $LR_DECAY_FACTOR \
    --seed $SEED \
    --device $DEVICE \
    --k_update $K_UPDATE \
    --lambda_str $LAMBDA_STR \
    --use_dueling \
    --use_double_dqn \
    --use_prioritized_replay \
    --normalize_rewards

echo "Training completed!"
echo "Results saved in results/ directory"
