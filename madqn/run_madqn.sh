#!/bin/bash

# MADQN Training Script
# This script provides examples of how to run MADQN with different configurations

# Activate conda environment (adjust if your environment name is different)
source ~/.bashrc
conda activate marl_env

# Change to project directory
cd /home/dongmingwang/project/Expert_RL

echo "Starting MADQN Training..."

# Example 1: Default configuration
echo "Running with default configuration..."
python madqn/train_madqn.py

# Example 2: Quick training run (fewer episodes)
# echo "Running quick training (500 episodes)..."
# python madqn/train_madqn.py --total_episodes 500 --epsilon_decay_episodes 100

# Example 3: High learning rate configuration
# echo "Running with high learning rate..."
# python madqn/train_madqn.py --lr 0.001 --total_episodes 1500

# Example 4: Conservative configuration (slower learning)
# echo "Running conservative configuration..."
# python madqn/train_madqn.py --lr 0.0001 --epsilon_decay_episodes 500 --total_episodes 3000

# Example 5: Large network configuration
# echo "Running with larger network..."
# python madqn/train_madqn.py --hidden_dims 256 256 128 --total_episodes 2500

# Example 6: Fast exploration
# echo "Running with fast exploration..."
# python madqn/train_madqn.py --epsilon_decay_episodes 150 --total_episodes 1500

# Example 7: Training with specific seed
# echo "Running with seed 42..."
# python madqn/train_madqn.py --seed 42 --total_episodes 1000

# Example 8: Multiple seeds (uncomment to run)
# for seed in {1..5}; do
#     echo "Running with seed $seed..."
#     python madqn/train_madqn.py --seed $seed --total_episodes 500
# done

echo "Training completed!"
echo ""
echo "To run with multiple random seeds (1-30), use:"
echo "  ./madqn/run_madqn_seeds.sh"
