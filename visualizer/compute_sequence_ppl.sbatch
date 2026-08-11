#!/bin/bash
#SBATCH --job-name=viz-seq-ppl
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/viz_seq_ppl_%A_%a.out
#SBATCH --error=logs/viz_seq_ppl_%A_%a.out

# Score generated / beam perplexity in 7 parallel shards on thickstun.
#   sbatch visualizer/compute_sequence_ppl.sbatch
# Then merge:
#   sbatch --dependency=afterok:<array_jobid> visualizer/merge_ppl_shards.sbatch
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

DATA="${DATA:-visualizer/data.js}"
NUM_SHARDS="${NUM_SHARDS:-7}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?missing array task id}"
OUT="visualizer/ppl_shards/shard_$(printf '%02d' "$SHARD_INDEX").json"

mkdir -p logs visualizer/ppl_shards
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Data: $DATA"
echo "Shard: $SHARD_INDEX / $NUM_SHARDS -> $OUT"
echo "Start: $(date)"

python visualizer/compute_sequence_ppl.py \
  --data "$DATA" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --output "$OUT"

echo "End: $(date)"
