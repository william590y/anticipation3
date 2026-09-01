#!/bin/bash
#SBATCH --job-name=base-score-ppl
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/base_score_ppl_%A_%a.out
#SBATCH --error=logs/base_score_ppl_%A_%a.out

# GT-score vs base-model-generated-score perplexity under the UNTUNED base
# model (eval_base_score_ppl.py), sharded over the thickstun GPUs: each array
# task takes stripe i::8 of the (stride-subsampled) val AND test paper-split
# files on one GPU.
#
# Submit:              sbatch submit_base_score_ppl.sbatch
# Adjust coverage:     sbatch --export=ALL,STRIDE=4 submit_base_score_ppl.sbatch
# Merge when done:     python eval_base_score_ppl.py --merge 'base_score_ppl_results/*_shard*.json'
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?missing array task id}"
STRIDE="${STRIDE:-1}"
DECODE_BS="${DECODE_BS:-96}"
OUTDIR="${OUTDIR:-base_score_ppl_results}"

mkdir -p logs "$OUTDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "Host: $(hostname)  GPU: $(nvidia-smi -L | head -1)"
echo "Shard: $SHARD_INDEX / $NUM_SHARDS   stride: $STRIDE   batch: $DECODE_BS"
echo "Start: $(date)"

for SPLIT in val test; do
  echo "=== split: $SPLIT ==="
  python eval_base_score_ppl.py \
    --token-file "data/${SPLIT}_paper.txt" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --stride "$STRIDE" \
    --decode-batch-size "$DECODE_BS" \
    --output "$OUTDIR/${SPLIT}_shard$(printf '%02d' "$SHARD_INDEX").json"
done

echo "End: $(date)"
