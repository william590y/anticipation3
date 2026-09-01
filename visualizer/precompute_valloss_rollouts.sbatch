#!/bin/bash
#SBATCH --job-name=viz-valloss
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/viz_valloss_%A_%a.out
#SBATCH --error=logs/viz_valloss_%A_%a.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

NUM_SHARDS=7
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?}"
OUT="visualizer/valloss_shards/shard_$(printf '%02d' "$SHARD_INDEX").json"
CHECKPOINT="${CHECKPOINT:-run_paper_split_v2/checkpoint-2500}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-run_paper_split_lora_r512/checkpoint-10000}"

mkdir -p logs visualizer/valloss_shards
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname) shard=$SHARD_INDEX/$NUM_SHARDS"
echo "Base: $CHECKPOINT  LoRA: $LORA_CHECKPOINT"
echo "Start: $(date)"

python visualizer/precompute_valloss_rollouts.py \
  --data visualizer/data.js \
  --checkpoint "$CHECKPOINT" \
  --lora-checkpoint "$LORA_CHECKPOINT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --output "$OUT"

echo "End: $(date)"
