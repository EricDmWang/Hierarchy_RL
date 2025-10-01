#!/bin/bash

# MADQN Training Script with Multiple Random Seeds
# This script runs MADQN training with different random seeds for reproducibility testing

# Activate conda environment
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MADQN Training with Multiple Seeds..."
echo "=============================================="

# Configuration parameters
TOTAL_EPISODES=2000
MAX_STEPS_PER_EP=200
SEED_START=1
SEED_END=5

# Run training with different seeds
for SEED in $(seq $SEED_START $SEED_END); do
    echo ""
    echo "Running MADQN with seed $SEED ($((SEED - SEED_START + 1))/$((SEED_END - SEED_START + 1)))..."
    echo "----------------------------------------"
    
    # Run training with current seed
    python madqn/train_madqn.py \
        --total_episodes $TOTAL_EPISODES \
        --max_steps_per_ep $MAX_STEPS_PER_EP \
        --seed $SEED \
        --lr 5e-4 \
        --gamma 0.97 \
        --epsilon_decay_episodes 300 \
        --batch_size 128 \
        --buffer_capacity 100000
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for seed $SEED"
    else
        echo "✗ Training failed for seed $SEED"
        exit 1
    fi
done

echo ""
echo "=============================================="
echo "All training runs completed successfully!"
echo "Results saved in: madqn/results/run_*"
echo "=============================================="
