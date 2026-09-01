#!/bin/bash
#SBATCH --job-name=viz-seed-publish
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=visualizer/logs/viz_seed_publish_%j.out
#SBATCH --error=visualizer/logs/viz_seed_publish_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
python -u visualizer/publish_seed_pipeline.py \
  --data visualizer/data.js \
  --grpo-shards visualizer/grpo_final_shards \
  --backfill-shards visualizer/seed_backfill_shards \
  --grpo-checkpoint "${GRPO_CHECKPOINT:-run_grpo_acc_reward/checkpoint-250}" \
  --html visualizer/visualizer.html \
  --cache-version "${DATA_CACHE_VERSION:-20260814020000}"
