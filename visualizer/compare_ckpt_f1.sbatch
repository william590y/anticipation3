#!/bin/bash
#SBATCH --job-name=viz-ckpt-f1
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/viz_ckpt_f1_%A_%a.out
#SBATCH --error=logs/viz_ckpt_f1_%A_%a.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

NUM_SHARDS=7
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?}"
OUT="visualizer/ckpt_f1_shards/shard_$(printf '%02d' "$SHARD_INDEX").json"

mkdir -p logs visualizer/ckpt_f1_shards
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)  shard=$SHARD_INDEX/$NUM_SHARDS"
echo "Start: $(date)"

python visualizer/compare_ckpt_f1.py \
  --data visualizer/data.js \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --output "$OUT"

echo "End: $(date)"
