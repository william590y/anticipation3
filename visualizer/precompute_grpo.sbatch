#!/bin/bash
#SBATCH --job-name=viz-grpo
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/viz_grpo_%A_%a.out
#SBATCH --error=logs/viz_grpo_%A_%a.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -u visualizer/precompute_grpo_rollouts.py \
    --data visualizer/data.js \
    --grpo-checkpoint run_grpo_acc_reward/checkpoint-250 \
    --seed-counts 1,2,3,4,5 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --num-shards 4 \
    --output "visualizer/grpo_shards/shard_0${SLURM_ARRAY_TASK_ID}.json"
