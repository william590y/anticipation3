#!/bin/bash
#SBATCH --job-name=rrsample-viz
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/rerank_sample_%A_%a.out
#SBATCH --error=logs/rerank_sample_%A_%a.out

# Sample-then-rerank decode (native N-best selection: 8 sampled + 1 greedy,
# fitted z-scored objective) on the 24 viz windows, 4 shards on the 4 ellis
# GPUs. Merge afterwards:
#   python visualizer/merge_rerank_rollouts.py --group rollouts_rerank_sample \
#       --shards visualizer/rerank_sample_shards/shard_*.json
#   python visualizer/compute_f1.py --data visualizer/data.js
#   python visualizer/compute_sequence_ppl.py --data visualizer/data.js --checkpoint run_paper_split_v2/checkpoint-7500
#   python visualizer/split_visualizer_payload.py
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

WEIGHTS="${WEIGHTS:-nbest_data/decode_weights_unfiltered.json}"
CKPT="${CKPT:-run_paper_split_v2/checkpoint-2500}"
VARIANT="${VARIANT:-raw}"
test -f "$WEIGHTS" || { echo "missing $WEIGHTS"; exit 1; }

KEYS=$(python -c "
import json, sys
raw = open('visualizer/data_slim.js').read()
d = json.loads(raw[raw.index('=')+1:].strip().rstrip(';'))
order = d['example_order']
print(','.join(order[int('$SLURM_ARRAY_TASK_ID')::4]))
")
echo "shard $SLURM_ARRAY_TASK_ID keys: $KEYS"
mkdir -p visualizer/rerank_sample_shards
python visualizer/rerank_sample_viz.py \
  --example-keys "$KEYS" --weights "$WEIGHTS" \
  --checkpoint "$CKPT" --variant "$VARIANT" \
  --output "visualizer/rerank_sample_shards/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").json"
