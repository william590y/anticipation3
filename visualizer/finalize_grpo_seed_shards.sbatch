#!/bin/bash
#SBATCH --job-name=viz-grpo-final
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=visualizer/logs/viz_grpo_final_%A_%a.out
#SBATCH --error=visualizer/logs/viz_grpo_final_%A_%a.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

idx="${SLURM_ARRAY_TASK_ID:?}"
mkdir -p visualizer/grpo_final_shards visualizer/logs
python -u visualizer/finalize_grpo_seed_shard.py \
  --data visualizer/data.js \
  --input "visualizer/grpo_shards/shard_0${idx}.json" \
  --output "visualizer/grpo_final_shards/shard_0${idx}.json" \
  --shard-index "${idx}" \
  --grpo-checkpoint "${GRPO_CHECKPOINT:-run_grpo_acc_reward/checkpoint-250}"
