#!/bin/bash
#SBATCH --job-name=viz-seed-fill
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%4
#SBATCH --output=visualizer/logs/viz_seed_fill_%A_%a.out
#SBATCH --error=visualizer/logs/viz_seed_fill_%A_%a.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

task="${SLURM_ARRAY_TASK_ID:?}"
shard=$(( task % 4 ))
model_index=$(( task / 4 ))
groups=(rollouts_lora rollouts_valloss rollouts_lora_valloss)
tags=(lora valloss lora_valloss)
group="${groups[$model_index]}"
tag="${tags[$model_index]}"
mkdir -p visualizer/seed_backfill_shards visualizer/logs
python -u visualizer/precompute_seed_backfill.py \
  --data visualizer/data.js \
  --group "${group}" \
  --shard-index "${shard}" \
  --output "visualizer/seed_backfill_shards/${tag}_shard_0${shard}.json"
